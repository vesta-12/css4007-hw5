
from pathlib import Path

from helpdesk_tools import (
    find_instructions,
    check_service_status,
    prepare_ticket,
)


print("\n=== TEST 1: FIND INSTRUCTIONS ===")

wifi_guide = find_instructions("Wi-Fi")
password_guide = find_instructions("password")

assert wifi_guide["instruction_id"] == "GUIDE-WIFI"
assert password_guide["instruction_id"] == "GUIDE-PASSWORD"

print("Wi-Fi:", wifi_guide["title"])
print("Steps:", len(wifi_guide["steps"]))
print("Password:", password_guide["title"])
print("PASS")


print("\n=== TEST 2: SERVICE STATUS ===")

expected = {
    "normal": "operational",
    "wifi_outage": "outage",
    "status_unavailable": "unknown",
}

for profile, expected_status in expected.items():
    result = check_service_status("wifi", profile)
    assert result["status"] == expected_status
    print(profile, "->", result["status"])

recovery = check_service_status(
    "password", "recovery_outage"
)

assert recovery["service"] == "account_recovery"
assert recovery["status"] == "outage"

print("recovery_outage ->", recovery["status"])
print("PASS")


print("\n=== TEST 3: MISSING INFORMATION ===")

missing = prepare_ticket(
    problem_description="",
    steps_tried=[],
    category=None,
    user_id="demo-user-a",
    thread_id="manual-test",
)

assert missing["ok"] is False
assert len(missing["missing_fields"]) == 2

print(missing["message"])
print("PASS")


print("\n=== TEST 4: CREATE TICKET DRAFT ===")

normal_status = check_service_status("wifi", "normal")

draft = prepare_ticket(
    problem_description=(
        "Campus-Demo Wi-Fi still does not connect."
    ),
    steps_tried=[
        "Enabled Wi-Fi",
        "Forgot the saved network",
        "Tried reconnecting",
    ],
    category="wifi",
    user_id="demo-user-a",
    thread_id="manual-test",
    status_profile="normal",
    service_status=normal_status["status"],
    reference_ids=[wifi_guide["instruction_id"]],
)

assert draft["ok"] is True
assert draft["ticket"]["status"] == "draft"
assert draft["ticket"]["category"] == "wifi"
assert draft["ticket"]["service_status"] == "operational"
assert Path(draft["path"]).exists()

print("Ticket ID:", draft["ticket"]["ticket_id"])
print("Saved to:", draft["path"])
print("PASS")


print("\nALL MANUAL TOOL TESTS PASSED")
