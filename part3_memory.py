import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from part2_agent import create_agent, MAX_GRAPH_STEPS


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "local-storage"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_scenario(scenario_id):
    data = json.loads(
        (DATA_DIR / "scenarios.json").read_text(encoding="utf-8")
    )
    for scenario in data["scenarios"]:
        if scenario["id"] == scenario_id and scenario["part"] == 3:
            return scenario
    raise ValueError(f"Unknown Part 3 scenario: {scenario_id}")


def preference_db(folder):
    path = folder / "preferences.sqlite"
    connection = sqlite3.connect(str(path))
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT PRIMARY KEY,
            language TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def get_preference(connection, user_id):
    row = connection.execute(
        "SELECT language FROM user_preferences WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    return row[0] if row else None


def save_preference(connection, user_id, language):
    connection.execute(
        """
        INSERT INTO user_preferences (user_id, language)
        VALUES (?, ?)
        ON CONFLICT(user_id)
        DO UPDATE SET language = excluded.language
        """,
        (user_id, language),
    )
    connection.commit()


def detect_preference(message):
    match = re.search(
        r"\b(?:i (?:prefer|want) (?:replies|responses|answers) in"
        r"|please (?:reply|respond) in)\s+"
        r"(kazakh|russian|english)\b",
        message,
        re.IGNORECASE,
    )
    return match.group(1).capitalize() if match else None


def run_batch(scenario, indices):
    scenario_id = scenario["id"]
    folder = STORAGE_DIR / scenario_id
    folder.mkdir(parents=True, exist_ok=True)

    batch_results = []

    with preference_db(folder) as preferences:
        with SqliteSaver.from_conn_string(
            str(folder / "checkpoints.sqlite")
        ) as checkpointer:

            for index in indices:
                event = scenario["events"][index]
                user_id = event["user_id"]
                thread_id = event["thread_id"]
                message = event["message"]

                saved_before = get_preference(preferences, user_id)
                detected = detect_preference(message)

                if detected:
                    save_preference(preferences, user_id, detected)

                language = get_preference(preferences, user_id)

                graph, begin_turn = create_agent(
                    user_id=user_id,
                    thread_id=thread_id,
                    status_profile=scenario["status_profile"],
                    checkpointer=checkpointer,
                    preferred_language=language,
                )

                config = {
                    "configurable": {
                        "thread_id": f"{user_id}/{thread_id}"
                    },
                    "recursion_limit": MAX_GRAPH_STEPS,
                }

                previous_state = graph.get_state(config)
                previous_messages = previous_state.values.get(
                    "messages", []
                )

                previous_user_messages = [
                    item.content
                    for item in previous_messages
                    if isinstance(item, HumanMessage)
                ]

                print(f"\nSCENARIO: {scenario_id}")
                print(f"USER ID: {user_id}")
                print(f"THREAD ID: {thread_id}")
                print("CONVERSATION MEMORY:", previous_user_messages)
                print("SAVED USER MEMORY:", saved_before)
                print("ACTIVE LANGUAGE:", language)
                print("USER:", message)

                begin_turn(message)
                trace = []

                for update in graph.stream(
                    {"messages": [HumanMessage(content=message)]},
                    config=config,
                    stream_mode="updates",
                ):
                    for node, node_update in update.items():
                        for item in node_update.get("messages", []):
                            record = {
                                "node": node,
                                "message_type": item.type,
                            }

                            if isinstance(item, AIMessage):
                                record["tool_calls"] = item.tool_calls
                                record["content"] = item.content

                                if item.tool_calls:
                                    print(
                                        "MODEL REQUESTS:",
                                        item.tool_calls,
                                    )
                                else:
                                    print("ASSISTANT:", item.content)

                            elif isinstance(item, ToolMessage):
                                record["tool_name"] = item.name
                                record["tool_result"] = item.content
                                print(
                                    "TOOL RESULT:",
                                    item.name,
                                    item.content,
                                )

                            trace.append(record)

                current_state = graph.get_state(config)
                current_messages = current_state.values.get(
                    "messages", []
                )

                batch_results.append({
                    "event_index": index,
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "user_message": message,
                    "conversation_memory_before": previous_user_messages,
                    "user_memory_before": saved_before,
                    "user_memory_written": detected,
                    "user_memory_after": language,
                    "message_count_after": len(current_messages),
                    "trace": trace,
                })

    batch_dir = OUTPUT_DIR / "part3_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    path = batch_dir / f"{scenario_id}_{indices[0]}.json"
    path.write_text(
        json.dumps(batch_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_scenario(scenario):
    scenario_id = scenario["id"]
    folder = STORAGE_DIR / scenario_id

    if scenario.get("reset_before"):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    batches = []
    current_batch = []
    restart_indices = []

    for index, event in enumerate(scenario["events"]):
        if event["action"] == "restart_process":
            if current_batch:
                batches.append(current_batch)
                current_batch = []
            restart_indices.append(index)
        elif event["action"] == "message":
            current_batch.append(index)

    if current_batch:
        batches.append(current_batch)

    all_results = []

    for batch_index, indices in enumerate(batches):
        if batch_index > 0:
            print("\n=== RESTARTING PYTHON PROCESS ===\n", flush=True)

        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--scenario",
                scenario_id,
                "--worker",
                "--indices",
                ",".join(map(str, indices)),
            ],
            check=True,
        )

        path = (
            OUTPUT_DIR
            / "part3_batches"
            / f"{scenario_id}_{indices[0]}.json"
        )

        all_results.extend(
            json.loads(path.read_text(encoding="utf-8"))
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{scenario_id}_part3.json"

    output_path.write_text(
        json.dumps({
            "scenario": scenario_id,
            "restart_events": restart_indices,
            "storage": str(folder),
            "runs": all_results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nTRACE SAVED: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--indices", default="")
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)

    if args.worker:
        indices = [
            int(index)
            for index in args.indices.split(",")
            if index
        ]
        run_batch(scenario, indices)
    else:
        run_scenario(scenario)
