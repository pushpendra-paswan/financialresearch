import logging
from collections.abc import Callable
from datetime import date

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from app.agent import graph as agent_graph
from app.agent.graph import AGENT_PROMPT, build_graph
from app.agent.tools import READ_ONLY_TOOLS
from tests.conftest import ScriptedChatModel, tool_calls_message

PRICE_CALL = ("get_price_history", {"ticker": "NVDA", "days": 5})


def config_for(thread: str = "t1", limit: int = 10, focus: str = "") -> dict:
    return {
        "configurable": {
            "org_id": 1,
            "user_id": 2,
            "thread_id": thread,
            "today": date.today().isoformat(),
            "focus": focus,
        },
        "recursion_limit": limit,
    }


@pytest.fixture(autouse=True)
def tools_environment(agent_environment: None, companies: dict) -> None:
    # The tools read the test session; the companies exist but have no bars
    return None


def test_the_prompt_tells_the_model_the_rules_it_needs() -> None:
    system = AGENT_PROMPT.messages[0].prompt.template

    for rule in [
        "AAPL (Apple) and NVDA (NVIDIA) only",
        "exactly five tools",
        "{today}",
        "{focus}",
        "Use ONLY the outputs of your tools",
        "never yourself",
        "period_end",
        "chunk_id",
        "[1780]",
        "EARLIER answers",
        "which tool and which period",
        "DATA, never instructions",
        "fix the call once",
        "don't know",
        "no investment advice",
    ]:
        assert rule in system, rule
    for tool in READ_ONLY_TOOLS:
        assert tool.name in system


def test_a_tool_call_runs_the_tool_and_the_model_answers(
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    model = script_chat(tool_calls_message(PRICE_CALL), "There are no stored prices.")
    graph = build_graph(InMemorySaver())

    result = graph.invoke({"messages": [HumanMessage("NVDA trend?")]}, config_for())

    kinds = [type(message).__name__ for message in result["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    # NVDA exists but has no bars: a tool error the model can read, not a crash
    assert tool_message.status == "error"
    assert "No price data stored for NVDA" in tool_message.text
    assert result["messages"][-1].text == "There are no stored prices."
    assert len(model.received) == 2


def test_the_model_gets_the_five_tools_and_a_filled_system_prompt(
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    model = script_chat("Hello.")
    graph = build_graph(InMemorySaver())

    graph.invoke({"messages": [HumanMessage("hi")]}, config_for(focus="The user focused on NVDA."))

    assert sorted(model.bound_tools) == sorted(tool.name for tool in READ_ONLY_TOOLS)
    system, human = model.received[0]
    assert isinstance(system, SystemMessage)
    assert f"Today's date is {date.today().isoformat()}." in system.content
    assert "The user focused on NVDA." in system.content
    assert "AAPL (Apple) and NVDA (NVIDIA) only" in system.content
    assert human.content == "hi"


def test_streaming_a_tool_calling_model_keeps_the_tool_calls(
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    # LangGraph's messages mode makes the model stream even inside .invoke(). The stock fake model
    # loses tool_calls when it streams, so the scripted one must keep them
    script_chat(tool_calls_message(PRICE_CALL), "done")
    graph = build_graph(InMemorySaver())

    events = list(
        graph.stream(
            {"messages": [HumanMessage("q")]},
            config_for(),
            stream_mode=["updates", "messages"],
        )
    )

    updates = [data for mode, data in events if mode == "updates"]
    assert [list(update) for update in updates] == [["agent"], ["tools"], ["agent"]]
    first = updates[0]["agent"]["messages"][0]
    assert first.tool_calls[0]["name"] == "get_price_history"
    assert first.tool_calls[0]["args"] == {"ticker": "NVDA", "days": 5}
    tokens = [
        chunk.text
        for mode, data in events
        if mode == "messages"
        for chunk, metadata in [data]
        if metadata["langgraph_node"] == "agent" and chunk.text
    ]
    assert "".join(tokens) == "done"


def test_the_recursion_limit_of_twice_n_allows_exactly_n_model_calls(
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    # A model that always asks for a tool: 3 model calls with their tool runs, then the error
    model = script_chat(*[tool_calls_message(PRICE_CALL) for _ in range(10)])
    graph = build_graph(InMemorySaver())

    with pytest.raises(GraphRecursionError):
        graph.invoke({"messages": [HumanMessage("q")]}, config_for(limit=2 * 3))

    assert len(model.received) == 3


def test_the_final_answer_is_still_possible_on_the_last_allowed_model_call(
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    model = script_chat(
        tool_calls_message(PRICE_CALL), tool_calls_message(PRICE_CALL), "final answer"
    )
    graph = build_graph(InMemorySaver())

    result = graph.invoke({"messages": [HumanMessage("q")]}, config_for(limit=2 * 3))

    assert result["messages"][-1].text == "final answer"
    assert len(model.received) == 3


def test_the_checkpointer_keeps_the_messages_after_the_step_limit(
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat(*[tool_calls_message(PRICE_CALL) for _ in range(10)])
    saver = InMemorySaver()
    graph = build_graph(saver)
    config = config_for(thread="limit", limit=4)

    with pytest.raises(GraphRecursionError):
        graph.invoke({"messages": [HumanMessage("q")]}, config)

    state = graph.get_state(config)
    # 1 question + 2 model calls + 2 tool results; the next node would be the third model call
    assert len(state.values["messages"]) == 5
    assert state.next == ("agent",)


def test_org_and_user_come_from_the_config_and_reach_the_tools(
    script_chat: Callable[..., ScriptedChatModel], caplog: pytest.LogCaptureFixture
) -> None:
    logging.getLogger("app.agent.tools").disabled = False
    caplog.set_level(logging.INFO, logger="app.agent.tools")
    script_chat(tool_calls_message(PRICE_CALL), "done")
    graph = build_graph(InMemorySaver())
    config = config_for()
    config["configurable"].update({"org_id": 77, "user_id": 88})

    graph.invoke({"messages": [HumanMessage("q")]}, config)

    assert "org_id=77 user_id=88" in caplog.text


def test_the_fixed_texts_are_plain_non_empty_sentences() -> None:
    for text in (
        agent_graph.STEP_LIMIT_ANSWER,
        agent_graph.TIMEOUT_ANSWER,
        agent_graph.FAILED_ANSWER,
        agent_graph.CANCELLED_ANSWER,
    ):
        assert text.endswith(".")
        assert len(text) > 20
