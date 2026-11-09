
import copy
import json
import re
import uuid
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_TICKETS_DIR = BASE_DIR / "outputs" / "tickets"


def load_json(filename):
    with open(DATA_DIR / filename, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_topic(topic):
    """Normalize topic names and common natural-language variants."""
    if not isinstance(topic, str):
        raise ValueError("Topic must be a string.")

    value = topic.strip().lower()

    if value in ("account_recovery", "account recovery"):
        return "password"

    helpdesk = load_json("helpdesk.json")

    for canonical, aliases in helpdesk["topic_aliases"].items():
        if value == canonical or value in aliases:
            return canonical

    has_wifi = any(
        word in value
        for word in ("wifi", "wi-fi", "campus-demo")
    )

    has_password = any(
        word in value
        for word in ("password", "account recovery")
    )

    if has_wifi and not has_password:
        return "wifi"

    if has_password and not has_wifi:
        return "password"

    raise ValueError(
        f"Unsupported or ambiguous topic: {topic}"
    )


# TOOL 1: Find instructions

def find_instructions(topic):
    """Find Wi-Fi or password instructions in local data."""
    canonical = normalize_topic(topic)
    helpdesk = load_json("helpdesk.json")
    guide = helpdesk["instructions"][canonical]

    return {
        "topic": canonical,
        "instruction_id": guide["instruction_id"],
        "title": guide["title"],
        "steps": guide["steps"],
    }


# TOOL 2: Check service status

def check_service_status(topic, status_profile="normal"):
    """Check a service using the selected fictional snapshot."""
    canonical = normalize_topic(topic)

    helpdesk = load_json("helpdesk.json")
    profiles = load_json("service_status.json")["profiles"]

    if status_profile not in profiles:
        raise ValueError(
            f"Unknown status profile: {status_profile}"
        )

    service = helpdesk["topic_services"][canonical]
    snapshot = profiles[status_profile][service]

    return {
        "topic": canonical,
        "service": service,
        "status": snapshot["status"],
        "message": snapshot["message"],
        "status_profile": status_profile,
    }


# TOOL 3: Prepare a local ticket

def prepare_ticket(
    problem_description,
    steps_tried,
    category,
    *,
    user_id,
    thread_id,
    status_profile="normal",
    service_status=None,
    reference_ids=None,
    tickets_dir=None,
):
    """Validate and save a local support ticket draft."""

    # The application supplies these identifiers, not the LLM.
    if not user_id or not thread_id:
        raise ValueError("user_id and thread_id are required.")

    profiles = load_json("service_status.json")["profiles"]
    if status_profile not in profiles:
        raise ValueError("Unknown status profile.")

    if isinstance(steps_tried, str):
        steps = [steps_tried.strip()] if steps_tried.strip() else []
    elif isinstance(steps_tried, list):
        steps = [
            step.strip()
            for step in steps_tried
            if isinstance(step, str) and step.strip()
        ]
    else:
        steps = []

    description = (
        problem_description.strip()
        if isinstance(problem_description, str)
        else ""
    )

    missing = []

    if not description:
        missing.append("problem description")

    if not steps:
        missing.append("steps already tried")

    if missing:
        return {
            "ok": False,
            "missing_fields": missing,
            "message": (
                "Please provide: " + ", ".join(missing) + "."
            ),
            "ticket": None,
        }

    # Reject explicit secret assignments rather than saving them.
    secret_pattern = (
        r"\b(?:password|api[\s_-]*key|recovery[\s_-]*code|"
        r"verification[\s_-]*code)\s*(?::|=|\bis\b)\s*\S+"
    )

    for text in [description] + steps:
        if re.search(secret_pattern, text, re.IGNORECASE):
            return {
                "ok": False,
                "missing_fields": [],
                "message": (
                    "Remove passwords, API keys or recovery "
                    "codes before preparing a ticket."
                ),
                "ticket": None,
            }

    canonical = normalize_topic(category) if category else None

    if service_status not in (
        None, "operational", "outage", "unknown"
    ):
        raise ValueError("Invalid service status.")

    template = load_json("ticket_template.json")
    ticket = copy.deepcopy(template["ticket"])

    ticket_id = "DRAFT-" + uuid.uuid4().hex[:12]

    ticket.update({
        "ticket_id": ticket_id,
        "status": "draft",
        "user_id": user_id,
        "thread_id": thread_id,
        "category": canonical,
        "summary": description,
        "steps_tried": steps,
        "service_status": service_status,
        "status_profile": status_profile,
        "reference_ids": list(reference_ids or []),
    })

    folder = (
        Path(tickets_dir)
        if tickets_dir is not None
        else DEFAULT_TICKETS_DIR
    )
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / f"{ticket_id}.json"
    path.write_text(
        json.dumps(ticket, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "ok": True,
        "missing_fields": [],
        "message": "Local ticket draft saved. Nothing was sent.",
        "ticket": ticket,
        "path": str(path),
    }
