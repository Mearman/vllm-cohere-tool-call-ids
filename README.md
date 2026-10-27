# vllm-cohere-tool-call-ids

A reproduction of a vLLM bug: the `cohere_command3` and `cohere_command4` parsers return the model's own `tool_call_id` as the OpenAI tool call id, so ids repeat within a conversation and the model's chat template then drops or misattributes tool results.

The Cohere Command models write a `tool_call_id` for each call, and vLLM constrains it to digits (`"pattern": "^[0-9]+$"` in [`vllm/parser/cohere_command.py` L135](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/parser/cohere_command.py#L135)). That value is a small index local to the prompt, and it reaches the client unchanged. The non-streaming path builds `FunctionCall(id=tc.id, ...)` ([L602-L605](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/parser/cohere_command.py#L602-L605)) and the streaming path sets `delta.id = resolved_id` ([L673-L674](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/parser/cohere_command.py#L673-L674), [L686-L688](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/parser/cohere_command.py#L686-L688)), where `tc.id` is the text melody copied out of the model's `"tool_call_id": "..."` ([melody v0.15.3 `action_filter.rs` L274-L282](https://github.com/cohere-ai/melody/blob/v0.15.3/src/parsing/action_filter.rs#L274-L282)). The chat completion handler only falls back to a generated id when that value is empty (`ToolCall(id=tc.id or make_tool_call_id(), ...)`, [`serving.py` L1066](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/entrypoints/openai/chat_completion/serving.py#L1066)), and the parser is built afresh for every request, so nothing makes the id unique across turns.

When a client sends the conversation back, the model's chat template ([`chat_template.jinja`](https://huggingface.co/CohereLabs/North-Mini-Code-1.0/blob/d11e61a842617a22dc328552fa5bb86231ee4f37/chat_template.jinja)) renumbers the calls by position (`regen_tool_call_ids` defaults to true) and links each tool result to the first call whose id matches (`tool_call_id_to_int`, L42-L60). A repeated id therefore links a later result to an earlier call, and a tool message whose id was already seen in an earlier group of results is skipped altogether (L248). The template never needs the client's id to be digits, only unique.

## Running

vLLM 0.32.0 ships Linux wheels only. No GPU or model weights are needed: the test drives the real parser class with a stub tokenizer and downloads the model's chat template at a pinned revision.

```bash
uv run pytest
```

On macOS, run it in a container:

```bash
docker run --rm -v "$PWD":/w -v vllm-cohere-ids-venv:/w/.venv -w /w \
  ghcr.io/astral-sh/uv:0.8.22-python3.12-bookworm-slim uv run --frozen pytest
```

`pyproject.toml` pins `vllm==0.32.0` and `cohere-melody==0.15.3`, and takes torch from the CPU index.

## Output on vLLM 0.32.0

The model output fed to the parser is two assistant turns that each emit `"tool_call_id": "1"`, and separately a turn with two parallel calls (`"0"`, `"1"`) followed by a turn emitting `"1"`.

```
test_tool_call_ids.py::test_ids_unique_across_turns[unary] FAILED
test_tool_call_ids.py::test_ids_unique_across_turns[streaming] FAILED
test_tool_call_ids.py::test_ids_unique_after_parallel_calls[unary] FAILED
test_tool_call_ids.py::test_ids_unique_after_parallel_calls[streaming] FAILED
test_tool_call_ids.py::test_template_links_each_result_to_its_call[across_turns] FAILED
test_tool_call_ids.py::test_template_links_each_result_to_its_call[after_parallel_calls] FAILED

E       AssertionError: duplicate tool call ids: ['1', '1']
E       AssertionError: duplicate tool call ids: ['0', '1', '1']

across_turns:
expected result -> call position: {'contents of a.txt': '0', 'contents of b.txt': '1'}
rendered result -> call position: {'contents of a.txt': '0', 'contents of b.txt': '0'}

after_parallel_calls:
expected result -> call position: {'contents of a.txt': '0', 'contents of b.txt': '1', 'contents of c.txt': '2'}
rendered result -> call position: {'contents of a.txt': '0', 'contents of b.txt': '1'}
```

In the first conversation the result for `b.txt` is attributed to the call that read `a.txt`. In the second the result for `c.txt` is missing from the prompt.

With the parser's ids replaced by unique values (digits or not), all six tests pass.

## Licence

MIT
