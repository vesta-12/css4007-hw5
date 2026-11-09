
import json
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

with open(DATA_DIR / "helpdesk.json", encoding="utf-8") as file:
    HELPDESK = json.load(file)


# Shared graph state
class HelpdeskState(TypedDict):
    message: str
    category: str
    reply: str


def get_instructions(topic: str) -> str:
    """Read the supplied instructions from helpdesk.json."""
    guide = HELPDESK["instructions"][topic]
    steps = "\n".join(
        f"{number}. {step}"
        for number, step in enumerate(guide["steps"], start=1)
    )
    return f"{guide['title']} ({guide['instruction_id']})\n{steps}"


# Help nodes
def wifi_help(state: HelpdeskState) -> dict:
    return {"reply": get_instructions("wifi")}


def password_help(state: HelpdeskState) -> dict:
    return {"reply": get_instructions("password")}


def other_help(state: HelpdeskState) -> dict:
    return {
        "reply": "Is your problem related to Wi-Fi or a password?"
    }


def clarify_problem(state: HelpdeskState) -> dict:
    return {
        "reply": (
            "Would you like to solve the Wi-Fi connection "
            "problem or the password problem first?"
        )
    }


def build_graph(variant: str = "original"):
    """Build the original or improved routing graph."""
    if variant not in ("original", "with_clarification"):
        raise ValueError(f"Unknown graph variant: {variant}")

    def read_request(state: HelpdeskState) -> dict:
        message = state["message"].lower()

        has_wifi = "wifi" in message or "wi-fi" in message
        has_password = "password" in message

        if (
            variant == "with_clarification"
            and has_wifi
            and has_password
        ):
            category = "both"
        elif has_wifi:
            category = "wifi"
        elif has_password:
            category = "password"
        else:
            category = "other"

        return {"category": category}

    def choose_next_node(state: HelpdeskState) -> str:
        return state["category"]

    graph = StateGraph(HelpdeskState)

    graph.add_node("read_request", read_request)
    graph.add_node("wifi_help", wifi_help)
    graph.add_node("password_help", password_help)
    graph.add_node("other_help", other_help)

    routes = {
        "wifi": "wifi_help",
        "password": "password_help",
        "other": "other_help",
    }

    if variant == "with_clarification":
        graph.add_node("clarify_problem", clarify_problem)
        routes["both"] = "clarify_problem"

    graph.add_edge(START, "read_request")
    graph.add_conditional_edges(
        "read_request",
        choose_next_node,
        routes,
    )

    for node in routes.values():
        graph.add_edge(node, END)

    return graph.compile()


def run_scenario(scenario: dict) -> dict:
    """Execute a Part 1 scenario and record its actual trace."""
    variant = scenario.get("graph_variant", "original")
    graph = build_graph(variant)

    event = next(
        event
        for event in scenario["events"]
        if event["action"] == "message"
    )

    state = {
        "message": event["message"],
        "category": "",
        "reply": "",
    }

    initial_state = state.copy()
    steps = []

    for chunk in graph.stream(state, stream_mode="updates"):
        for node_name, update in chunk.items():
            state.update(update)

            steps.append({
                "node": node_name,
                "update": update,
                "state_after": state.copy(),
            })

    route = ["START"] + [
        step["node"] for step in steps
    ] + ["END"]

    result = {
        "scenario": scenario["id"],
        "graph_variant": variant,
        "status_profile": scenario["status_profile"],
        "user_id": event["user_id"],
        "thread_id": event["thread_id"],
        "reference_file": "data/helpdesk.json",
        "initial_state": initial_state,
        "route": route,
        "steps": steps,
        "final_state": state,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / f"{scenario['id']}_part1.json"

    output_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{scenario['id']} ({variant})")
    print("Route:", " -> ".join(route))
    print("Category:", state["category"])
    print("Reply:", state["reply"])
    print("Saved:", output_file.relative_to(BASE_DIR))

    return result


if __name__ == "__main__":
    with open(DATA_DIR / "scenarios.json", encoding="utf-8") as file:
        scenarios_data = json.load(file)

    for scenario in scenarios_data["scenarios"]:
        if scenario["part"] == 1:
            run_scenario(scenario)
