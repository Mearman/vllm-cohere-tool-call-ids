"""Tool call ids from vLLM's cohere_command4 parser repeat across a conversation.

The parser hands the model's ``tool_call_id`` (a prompt-local index the chat template numbers from zero) straight to the client as the OpenAI tool call id. Nothing makes it unique, so two turns that both emit ``"1"`` give two tool calls with the id ``"1"``, and the model's chat template then drops or misattributes the second tool result when the conversation is sent back.
"""

from __future__ import annotations

import json

import pytest
from huggingface_hub import hf_hub_download
from transformers.utils.chat_template_utils import render_jinja_template

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager

PARSER = "cohere_command4"
MODEL = "CohereLabs/North-Mini-Code-1.0"
# Pinned so the rendered prompt cannot change under the test.
MODEL_REVISION = "d11e61a842617a22dc328552fa5bb86231ee4f37"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def action(*calls: tuple[str, str]) -> str:
    """Model output for one assistant turn: ``(tool_call_id, path)`` per call."""
    body = ",\n".join(
        json.dumps(
            {"tool_call_id": call_id, "tool_name": "read_file", "parameters": {"path": path}}
        )
        for call_id, path in calls
    )
    return f"<|START_THINKING|>read it<|END_THINKING|><|START_ACTION|>[\n{body}\n]<|END_ACTION|>"


# Two assistant turns of one conversation, each emitting tool_call_id "1", as observed from the model.
TURNS = [action(("1", "a.txt")), action(("1", "b.txt"))]
# One assistant turn with two parallel calls, then a later turn reusing "1".
PARALLEL_THEN_SINGLE = [action(("0", "a.txt"), ("1", "b.txt")), action(("1", "c.txt"))]


class StubTokenizer:
    """The parser only asks the tokenizer for the thinking marker ids."""

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|START_THINKING|>": 1, "<|END_THINKING|>": 2}.get(token, 0)

    def get_vocab(self) -> dict[str, int]:
        return {}


def new_parser():
    """A fresh parser, as the server builds one per request."""
    cls = ParserManager.get_parser(PARSER, PARSER, enable_auto_tools=True)
    return cls(StubTokenizer(), TOOLS, model_config=None, chat_template_kwargs={})


def new_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[], model=MODEL, tools=TOOLS, tool_choice="auto")


def unary_calls(model_output: str) -> list[tuple[str, str]]:
    """``(id, arguments)`` of each tool call the non-streaming path returns."""
    _, _, tool_calls = new_parser().parse(model_output, new_request(), enable_auto_tools=True)
    return [(tc.id, tc.arguments) for tc in tool_calls or []]


def streamed_calls(model_output: str) -> list[tuple[str, str]]:
    """``(id, arguments)`` of each tool call accumulated from streamed deltas."""
    parser, request = new_parser(), new_request()
    chunks = [model_output[i : i + 3] for i in range(0, len(model_output), 3)]
    slots: dict[int, dict[str, str]] = {}
    for i, chunk in enumerate(chunks):
        delta = parser.parse_delta(chunk, [], request, finished=i == len(chunks) - 1)
        for tc in (delta.tool_calls if delta else None) or []:
            slot = slots.setdefault(tc.index, {"id": "", "arguments": ""})
            slot["id"] = tc.id or slot["id"]
            if tc.function and tc.function.arguments:
                slot["arguments"] += tc.function.arguments
    return [(s["id"], s["arguments"]) for _, s in sorted(slots.items())]


PATHS = pytest.mark.parametrize("calls_of", [unary_calls, streamed_calls], ids=["unary", "streaming"])


def conversation_ids(calls_of, turns: list[str]) -> list[str]:
    ids = [call_id for turn in turns for call_id, _ in calls_of(turn)]
    print(f"\ntool call ids across the conversation: {ids}")
    return ids


@PATHS
def test_ids_unique_across_turns(calls_of):
    ids = conversation_ids(calls_of, TURNS)
    assert len(set(ids)) == len(ids), f"duplicate tool call ids: {ids}"


@PATHS
def test_ids_unique_after_parallel_calls(calls_of):
    ids = conversation_ids(calls_of, PARALLEL_THEN_SINGLE)
    assert len(set(ids)) == len(ids), f"duplicate tool call ids: {ids}"


@pytest.mark.parametrize(
    "turns", [TURNS, PARALLEL_THEN_SINGLE], ids=["across_turns", "after_parallel_calls"]
)
def test_template_links_each_result_to_its_call(turns):
    """Send the parser's ids back the way a client does and render the model's own chat template: every tool result must appear, linked to its own call."""
    messages: list[dict] = [{"role": "user", "content": "read the files"}]
    for turn in turns:
        calls = unary_calls(turn)
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": arguments}}
                    for call_id, arguments in calls
                ],
            }
        )
        for call_id, arguments in calls:
            path = json.loads(arguments)["path"]
            messages.append({"role": "tool", "tool_call_id": call_id, "content": f"contents of {path}"})

    template_path = hf_hub_download(MODEL, "chat_template.jinja", revision=MODEL_REVISION)
    with open(template_path, encoding="utf-8") as f:
        template = f.read()
    rendered, _ = render_jinja_template(
        conversations=[messages], chat_template=template, tools=TOOLS, add_generation_prompt=True
    )
    prompt = rendered[0]

    # The template numbers calls by position across the conversation and links each result back by the client's id.
    links: dict[str, str] = {}
    for result_block in prompt.split("<|START_TOOL_RESULT|>")[1:]:
        for entry in json.loads(result_block.split("<|END_TOOL_RESULT|>")[0]):
            links[entry["results"]["0"]["content"]] = entry["tool_call_id"]
    paths = [
        json.loads(call["function"]["arguments"])["path"]
        for message in messages
        for call in message.get("tool_calls", [])
    ]
    expected = {f"contents of {path}": str(position) for position, path in enumerate(paths)}
    print(f"\nexpected result -> call position: {expected}\nrendered result -> call position: {links}")
    assert links == expected
