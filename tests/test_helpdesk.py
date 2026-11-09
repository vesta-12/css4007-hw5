
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

import part2_agent
from part1_graph import build_graph
from helpdesk_tools import (
    find_instructions,
    check_service_status,
    prepare_ticket,
)
from part3_memory import (
    preference_db,
    get_preference,
    save_preference,
)


@pytest.mark.parametrize(
    "message,variant,expected",
    [
        ("I cannot connect to Wi-Fi.", "original", "wifi"),
        ("I forgot my PASSWORD.", "original", "password"),
        ("My laptop screen is broken.", "original", "other"),
        ("My Wi-Fi password does not work.", "original", "wifi"),
        (
            "My Wi-Fi password does not work.",
            "with_clarification",
            "both",
        ),
    ],
)
def test_routing(message, variant, expected):
    graph = build_graph(variant)
    result = graph.invoke({
        "message": message,
        "category": "",
        "reply": "",
    })
    assert result["category"] == expected
    assert result["reply"]


def test_instructions():
    assert find_instructions("wifi")["instruction_id"] == "GUIDE-WIFI"
    assert (
        find_instructions("password")["instruction_id"]
        == "GUIDE-PASSWORD"
    )
    assert find_instructions("Wi-Fi troubleshooting")["topic"] == "wifi"


@pytest.mark.parametrize(
    "profile,expected",
    [
        ("normal", "operational"),
        ("wifi_outage", "outage"),
        ("status_unavailable", "unknown"),
    ],
)
def test_wifi_status(profile, expected):
    result = check_service_status("wifi", profile)
    assert result["status"] == expected


def test_recovery_status():
    result = check_service_status("password", "recovery_outage")
    assert result["service"] == "account_recovery"
    assert result["status"] == "outage"


def test_ticket_missing_details(tmp_path):
    result = prepare_ticket(
        problem_description="",
        steps_tried=[],
        category="wifi",
        user_id="demo-user-a",
        thread_id="test",
        tickets_dir=tmp_path,
    )

    assert result["ok"] is False
    assert len(result["missing_fields"]) == 2
    assert not list(tmp_path.glob("*.json"))


def test_ticket_created(tmp_path):
    result = prepare_ticket(
        problem_description="Wi-Fi does not connect.",
        steps_tried=["Enabled Wi-Fi", "Tried reconnecting"],
        category="wifi",
        user_id="demo-user-a",
        thread_id="test",
        service_status="operational",
        tickets_dir=tmp_path,
    )

    assert result["ok"] is True
    assert result["ticket"]["status"] == "draft"

    saved = json.loads(
        (tmp_path / (result["ticket"]["ticket_id"] + ".json"))
        .read_text(encoding="utf-8")
    )
    assert saved["category"] == "wifi"
    assert saved["user_id"] == "demo-user-a"


def test_ticket_rejects_explicit_secret(tmp_path):
    result = prepare_ticket(
        problem_description="Password: secret123",
        steps_tried=["Tried recovery"],
        category="password",
        user_id="demo-user-a",
        thread_id="test",
        tickets_dir=tmp_path,
    )

    assert result["ok"] is False
    assert not list(tmp_path.glob("*.json"))


def test_user_preference_survives_reopen(tmp_path):
    connection = preference_db(tmp_path)
    save_preference(connection, "demo-user-a", "Kazakh")
    connection.close()

    reopened = preference_db(tmp_path)
    assert get_preference(reopened, "demo-user-a") == "Kazakh"
    assert get_preference(reopened, "demo-user-b") is None
    reopened.close()


class FakeChatModel:
    def __init__(self, **kwargs):
        pass

    def bind_tools(self, tools, parallel_tool_calls=False):
        return self

    def invoke(self, messages):
        user_messages = [
            message.content
            for message in messages
            if isinstance(message, HumanMessage)
        ]
        return AIMessage(
            content=f"Remembered {len(user_messages)} user messages."
        )


def test_conversation_persistence_and_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(part2_agent, "ChatOpenAI", FakeChatModel)

    database = str(tmp_path / "checkpoints.sqlite")

    first = {
        "configurable": {"thread_id": "demo-user-a/thread-1"}
    }
    new_thread = {
        "configurable": {"thread_id": "demo-user-a/thread-2"}
    }
    other_user = {
        "configurable": {"thread_id": "demo-user-b/thread-1"}
    }

    with SqliteSaver.from_conn_string(database) as saver:
        graph, _ = part2_agent.create_agent(
            "demo-user-a", "thread-1", "normal",
            checkpointer=saver,
        )
        graph.invoke(
            {"messages": [HumanMessage(content="Wi-Fi failed.")]},
            first,
        )

    with SqliteSaver.from_conn_string(database) as saver:
        graph, _ = part2_agent.create_agent(
            "demo-user-a", "thread-1", "normal",
            checkpointer=saver,
        )

        old_messages = graph.get_state(first).values["messages"]
        assert any(
            isinstance(msg, HumanMessage)
            and msg.content == "Wi-Fi failed."
            for msg in old_messages
        )

        assert not graph.get_state(new_thread).values
        assert not graph.get_state(other_user).values

        result = graph.invoke(
            {"messages": [HumanMessage(content="What happened?")]},
            first,
        )

        human_messages = [
            msg for msg in result["messages"]
            if isinstance(msg, HumanMessage)
        ]
        assert len(human_messages) == 2
