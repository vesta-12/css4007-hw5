import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from helpdesk_tools import (
    find_instructions as local_find_instructions,
    check_service_status as local_check_service_status,
    prepare_ticket as local_prepare_ticket,
    normalize_topic,
)


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"
MAX_GRAPH_STEPS = 15

SYSTEM_PROMPT = """
You are a university IT helpdesk assistant using fictional demo data.

You help users with Wi-Fi, password recovery, service status,
and local support ticket drafts.

Rules:
1. Use find_instructions when troubleshooting steps are requested.
2. Use check_service_status when service status is requested.
3. Prefer canonical topics: "wifi" or "password".
4. For password recovery, check the account_recovery service.
5. For a conditional ticket request, check the service first.
6. If the user requested a ticket only if the service is operational,
   do not create it when the status is outage or unknown.
   Explain that the user's condition was not met.
7. If a service is down, explain the outage without inventing an ETA.
8. If status is unknown, do not prepare a conditional ticket.
9. A ticket requires a problem description and steps already tried.
10. Ask for missing information. Use only troubleshooting actions
    explicitly reported by the user. You may paraphrase them,
    but never add unreported actions.
11. Never request or store passwords, API keys, or recovery codes.
12. Do not ask for an internal category if the issue is clear.
13. Tickets are local drafts, never submitted to a real helpdesk.
14. Use previous messages to understand follow-up requests.
15. Never invent tool results, ticket IDs, or filenames.
16. Report tool errors honestly.
17. For steps_tried, use short verbatim phrases from the user's
messages. Never add actions mentioned only in instructions.
18. Only prepare a ticket when the user explicitly requested it.
    Reporting an unresolved problem is not permission to create one.
19. Distinguish attempted troubleshooting actions from their outcomes.
    A failed reconnection is a result, not a separate action.
20. Do not suggest preparing a ticket unless the user requests
    one or explicitly asks about escalation.
21. Do not ask the user to repeat information already available
    in the conversation history.

After using tools, provide a clear and concise final answer.
"""


def as_json(value):
    return json.dumps(value, ensure_ascii=False)


def is_conditional_ticket_request(message):
    text = message.lower()

    conditional = bool(
        re.search(r"\b(if|only if|provided that)\b", text)
    )
    operating = bool(
        re.search(
            r"\b(operating|operational|working|available|up)\b",
            text,
        )
    )
    return conditional and operating


def create_agent(
    user_id,
    thread_id,
    status_profile,
    checkpointer=None,
    preferred_language=None,
):
    turn = {
        "conditional_ticket": False,
        "checked_statuses": {},
        "reference_ids": set(),
        "current_message": "",
        "user_messages": [],
        "ticket_requested": False,
    }

    def begin_turn(message):
        turn["current_message"] = message
        turn["conditional_ticket"] = (
            is_conditional_ticket_request(message)
        )
        turn["checked_statuses"] = {}
        turn["user_messages"].append(message)

    @tool("find_instructions")
    def instructions_tool(topic: str) -> str:
        """Retrieve Wi-Fi or password instructions from local data."""
        try:
            result = local_find_instructions(topic)
            turn["reference_ids"].add(result["instruction_id"])
            return as_json(result)
        except ValueError as error:
            return as_json({"error": str(error)})

    @tool("check_service_status")
    def status_tool(topic: str) -> str:
        """Check the demo status of Wi-Fi or password recovery."""
        try:
            result = local_check_service_status(
                topic, status_profile
            )
            turn["checked_statuses"][result["topic"]] = (
                result["status"]
            )
            return as_json(result)
        except ValueError as error:
            return as_json({"error": str(error)})

    @tool("prepare_ticket")
    def ticket_tool(
        problem_description: str = "",
        steps_tried: Optional[list[str]] = None,
        category: str = "",
    ) -> str:
        """Prepare a local ticket draft using user-provided details."""
        if not turn["ticket_requested"]:
            return as_json({
                "ok": False,
                "message": "No ticket was requested. Draft not created.",
            })
        try:
            steps = steps_tried or []
            description = problem_description.strip()

            missing = []
            if not description:
                missing.append("problem description")
            if not steps:
                missing.append("steps already tried")

            if missing:
                return as_json({
                    "ok": False,
                    "missing_fields": missing,
                    "message": "Please provide: " + ", ".join(missing) + ".",
                })

            checked_topics = list(turn["checked_statuses"].keys())

            if category:
                canonical = normalize_topic(category)
            elif len(checked_topics) == 1:
                canonical = checked_topics[0]
            else:
                source_text = (
                    description + " " + turn["current_message"]
                )
                try:
                    canonical = normalize_topic(source_text)
                except ValueError:
                    return as_json({
                        "ok": False,
                        "message": (
                            "Please clarify whether the issue "
                            "concerns Wi-Fi or password recovery."
                        ),
                    })

            checked_status = turn["checked_statuses"].get(canonical)

            if (
                turn["conditional_ticket"]
                and checked_status != "operational"
            ):
                return as_json({
                    "ok": False,
                    "message": (
                        "Conditional ticket blocked. The service "
                        "was not confirmed operational."
                    ),
                    "checked_status": checked_status or "not_checked",
                })

            result = local_prepare_ticket(
                problem_description=description,
                steps_tried=steps,
                category=canonical,
                user_id=user_id,
                thread_id=thread_id,
                status_profile=status_profile,
                service_status=checked_status,
                reference_ids=sorted(turn["reference_ids"]),
            )
            return as_json(result)

        except ValueError as error:
            return as_json({"ok": False, "error": str(error)})

    tools = [instructions_tool, status_tool, ticket_tool]

    model = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        timeout=45,
        max_retries=1,
    )
    model_with_tools = model.bind_tools(
        tools, parallel_tool_calls=False
    )

    def call_model(state: MessagesState):
        language_rule = (
            f"The saved user language preference is {preferred_language}. "
            f"Reply in {preferred_language}."
            if preferred_language
            else (
                "No saved language preference is available. "
                "Do not claim the user previously selected a language."
            )
        )

        user_messages = [
            msg.content
            for msg in state["messages"]
            if isinstance(msg, HumanMessage)
            and isinstance(msg.content, str)
        ]

        turn["ticket_requested"] = any(
            re.search(
                r"\b(?:prepare|create|make|open)\b.{0,60}\bticket\b",
                message,
                re.IGNORECASE,
            )
            for message in user_messages
        )

        response = model_with_tools.invoke(
            [
                SystemMessage(
                    content=SYSTEM_PROMPT + "\n" + language_rule
                )
            ] + state["messages"]
        )
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model",
        tools_condition,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "model")

    return builder.compile(checkpointer=checkpointer), begin_turn


def run_scenario(scenario):
    user_events = [
        event
        for event in scenario["events"]
        if event["action"] == "message"
    ]

    if not user_events:
        raise ValueError("Scenario contains no user messages.")

    user_id = user_events[0]["user_id"]
    thread_id = user_events[0]["thread_id"]
    profile = scenario["status_profile"]

    graph, begin_turn = create_agent(
        user_id, thread_id, profile
    )

    history = []
    trace = []
    ticket_paths = []
    loop_limit_reached = False

    print(f"\n{'=' * 55}")
    print(f"SCENARIO: {scenario['id']}")
    print(f"PROFILE: {profile}")
    print(f"MODEL: {MODEL_NAME}")
    print(f"{'=' * 55}")

    for event in user_events:
        message = event["message"]
        begin_turn(message)

        history.append(HumanMessage(content=message))
        print(f"\nUSER: {message}")

        try:
            stream = graph.stream(
                {"messages": list(history)},
                config={"recursion_limit": MAX_GRAPH_STEPS},
                stream_mode="updates",
            )

            for update in stream:
                for node_name, node_update in update.items():
                    messages = node_update.get("messages", [])

                    for msg in messages:
                        history.append(msg)

                        record = {
                            "node": node_name,
                            "message_type": msg.type,
                            "message_count_after": len(history),
                        }

                        if isinstance(msg, AIMessage):
                            calls = [
                                {
                                    "name": call["name"],
                                    "args": call["args"],
                                }
                                for call in msg.tool_calls
                            ]

                            record["tool_requests"] = calls
                            record["content"] = msg.content

                            for call in calls:
                                print(
                                    "MODEL REQUESTS:",
                                    call["name"],
                                    call["args"],
                                )

                            if not calls:
                                print("ASSISTANT:", msg.content)

                        elif isinstance(msg, ToolMessage):
                            record["tool_name"] = msg.name
                            record["tool_result"] = msg.content

                            print(
                                "TOOL RESULT:",
                                msg.name,
                                msg.content,
                            )

                            if msg.name == "prepare_ticket":
                                try:
                                    data = json.loads(msg.content)
                                    if data.get("ok") and data.get("path"):
                                        ticket_paths.append(data["path"])
                                except (ValueError, TypeError):
                                    pass

                        trace.append(record)

        except GraphRecursionError:
            loop_limit_reached = True
            print("LOOP LIMIT REACHED:", MAX_GRAPH_STEPS)
            break

    final_answer = ""
    if history and isinstance(history[-1], AIMessage):
        final_answer = str(history[-1].content)

    result = {
        "scenario": scenario["id"],
        "model": MODEL_NAME,
        "provider": "OpenAI",
        "status_profile": profile,
        "user_id": user_id,
        "thread_id": thread_id,
        "source_files": [
            "data/helpdesk.json",
            "data/service_status.json",
            "data/ticket_template.json",
        ],
        "trace": trace,
        "final_answer": final_answer,
        "ticket_paths": ticket_paths,
        "loop_limit_reached": loop_limit_reached,
    }

    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{scenario['id']}_part2.json"
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nTRACE SAVED: {path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default="S06",
        help="Part 2 scenario, e.g. S06, S07, S08",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Configure your .env file."
        )

    scenarios_path = BASE_DIR / "data" / "scenarios.json"
    data = json.loads(
        scenarios_path.read_text(encoding="utf-8")
    )

    available = {
        item["id"]: item
        for item in data["scenarios"]
        if item["part"] == 2
    }

    if args.scenario not in available:
        parser.error(
            "Choose one of: " + ", ".join(sorted(available))
        )

    run_scenario(available[args.scenario])
