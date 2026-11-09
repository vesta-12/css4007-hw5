
# CSS-4007 HW5 — University IT Helpdesk with LangGraph

**Student:** Sofiya Serbina  
**Course:** CSS-4007 Artificial Intelligence  
**Implementation:** Python 3.9, LangGraph, OpenAI `gpt-4o-mini`, SQLite

This repository contains a terminal-based demonstration of an IT helpdesk for a **fictional university**. It retrieves Wi-Fi and password instructions, checks simulated service status, creates local support ticket drafts, and preserves conversation context and user language preferences.

No real university services are queried, and no tickets are submitted.

## Project Structure

| File or folder | Purpose |
| --- | --- |
| `part1_graph.py` | Deterministic LangGraph routing for S01–S05 |
| `helpdesk_tools.py` | Three reusable local tools |
| `test_tools_manual.py` | Direct tool demonstrations |
| `part2_agent.py` | LLM agent, tool-calling loop and S06–S09, S14 |
| `part3_memory.py` | Persistent conversation and user memory, S10–S13 |
| `tests/test_helpdesk.py` | Automated tests using a fake model |
| `.github/workflows/tests.yml` | GitHub Actions continuous integration |
| `requirements.lock.txt` | Pinned dependencies |
| `data/` | Supplied fictional helpdesk data and scenarios |
| `outputs/` | Saved execution traces |
| `SUBMISSION.md` | Graphs, explanations, experimental results |

## 1. Environment Setup

The project was developed with Python 3.9. The GitHub Actions workflow uses Python 3.9 on Ubuntu 24.04.

Clone the repository:

```bash
git clone https://github.com/vesta-12/css4007-hw5.git
cd css4007-hw5
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the pinned dependencies:

```bash
python -m pip install -r requirements.lock.txt
```

Check the environment:

```bash
python --version
python -m pip check
```

### API Configuration

Part 1 and the automated tests do not require an API key.

Parts 2 and 3 use a real OpenAI model. Copy the example environment file:

```bash
cp .env.example .env
```

Configure the following variables in `.env`:

```dotenv
OPENAI_API_KEY=YOUR_PRIVATE_KEY
MODEL_NAME=gpt-4o-mini
LANGSMITH_TRACING=false
```

An OpenAI API key with available API credits is required for the real model demonstrations.

Never commit `.env` or expose private API keys.

## 2. Run Part 1 — Graph Routing

Part 1 uses LangGraph without an LLM.

Run all five routing scenarios:

```bash
python part1_graph.py
```

The application demonstrates:

- Wi-Fi keyword routing.
- Case-insensitive password routing.
- Fallback for unsupported problems.
- Priority when Wi-Fi and password both appear.
- An improved graph with a clarification node.

Results are saved to:

```text
outputs/S01_part1.json
outputs/S02_part1.json
outputs/S03_part1.json
outputs/S04_part1.json
outputs/S05_part1.json
```

Each trace records the execution route, state updates and final state.

## 3. Run Part 2 — Model and Tools

Part 2 adds three reusable tools:

| Tool | Function |
| --- | --- |
| `find_instructions` | Retrieves local Wi-Fi or password instructions |
| `check_service_status` | Checks the supplied service-status snapshot |
| `prepare_ticket` | Validates and saves a local ticket draft |

The model decides which tools to request. LangGraph executes the tool loop and returns tool results to the model.

First, test the tools independently:

```bash
python test_tools_manual.py
```

Run the model demonstrations:

```bash
python part2_agent.py --scenario S06
python part2_agent.py --scenario S07
python part2_agent.py --scenario S08
python part2_agent.py --scenario S09
python part2_agent.py --scenario S14
```

These commands perform real model calls and may consume API credits.

The status profiles demonstrate different conditional behavior:

- `normal`: Wi-Fi is operational.
- `wifi_outage`: Wi-Fi has a known outage.
- `status_unavailable`: Wi-Fi status is unknown.
- `recovery_outage`: Account recovery has a known outage.

A conditional ticket is prepared only when the requested service has been confirmed operational.

Ticket drafts are saved under:

```text
outputs/tickets/
```

They are local JSON files and are never submitted to a real helpdesk.

Model execution traces are saved to `outputs/S##_part2.json`.

## 4. Run Part 3 — Persistent Memory

Part 3 implements two types of memory.

**Conversation memory:** LangGraph `SqliteSaver` checkpoints the conversation state, including previous user and assistant messages.

**User memory:** A separate SQLite database stores the preferred response language for each invented user.

Run:

```bash
python part3_memory.py --scenario S10
python part3_memory.py --scenario S11
python part3_memory.py --scenario S12
python part3_memory.py --scenario S13
```

These scenarios demonstrate:

- S10: Recalling previous troubleshooting actions.
- S11: Restoring conversation and user memory after a real process restart.
- S12: Remembering language preferences in a new conversation without copying the previous conversation history.
- S13: Keeping different users' memory separate.

The scenario runner uses the supplied user identifiers, thread identifiers, messages and reset/restart events from `data/scenarios.json`.

It launches a new Python process for `restart_process` events rather than simulating a restart in the same application instance.

### Memory Storage

Persistent files are stored in:

```text
local-storage/
└── <scenario-id>/
    ├── checkpoints.sqlite
    └── preferences.sqlite
```

`checkpoints.sqlite` stores conversation history, indexed by the composite LangGraph thread identifier `user_id/thread_id`.

`preferences.sqlite` stores language preferences indexed by `user_id`.

A new conversation for the same user can retrieve the saved language without inheriting another conversation's messages.

### Clearing Memory

Close the program before deleting the database files.

To clear all local memory on macOS or Linux:

```bash
rm -rf local-storage
```

To clear local ticket drafts:

```bash
rm -rf outputs/tickets
```

The Part 3 runner also respects the `reset_before` setting in the supplied scenarios.

Execution traces are stored separately under `outputs/`.

## 5. Automated Tests

Run:

```bash
python -m pytest tests/ -v
```

The tests cover:

- Original and improved graph routing.
- Instruction lookup.
- Service-status profiles.
- Ticket creation and validation.
- Rejection of some explicitly supplied secrets.
- Persistent user preferences.
- Conversation restoration from SQLite.
- Isolation between users and conversations.

The automated tests use a clearly labelled `FakeChatModel`.

This allows the LangGraph model node to be tested without making OpenAI API calls or providing a private key.

Real model behavior is demonstrated separately through the Part 2 and Part 3 scenarios.

## 6. Continuous Integration

GitHub Actions is configured in:

```text
.github/workflows/tests.yml
```

The workflow runs automatically on pushes and pull requests.

It uses:

- Ubuntu 24.04.
- Python 3.9.
- Pinned dependencies from `requirements.lock.txt`.
- `pytest` without a private model API key.

The GitHub Actions workflow has been successfully tested.

## 7. Execution Logs

The project includes the following scenario groups:

| Part | Scenarios | Output |
| --- | --- | --- |
| Part 1 | S01–S05 | `outputs/S##_part1.json` |
| Part 2 | S06–S09, S14 | `outputs/S##_part2.json` |
| Part 3 | S10–S13 | `outputs/S##_part3.json` |

The saved JSON files contain actual execution traces.

Part 1 logs include graph nodes, state updates and routing decisions.

Part 2 logs include model tool requests, tool results and final answers.

Part 3 logs include conversation memory, user preferences and results of persistent-memory demonstrations.

Model-generated text and ticket identifiers may differ between runs.

## 8. Security and Limitations

The project is an educational demonstration using fictional university data.

- No real IT service is contacted.
- Service statuses are supplied snapshots, not real-time monitoring.
- Tickets are local drafts and are not submitted externally.
- API keys are stored in environment variables or an ignored `.env` file.
- Local SQLite databases and generated ticket drafts must not be committed.
- The model is instructed not to invent troubleshooting actions or request passwords and recovery codes.
- The local validator blocks certain explicit secret formats but is not a complete secret-redaction system.
- The preference parser recognizes limited English phrases for Kazakh, Russian and English.
- The agent has a graph recursion limit of 15 to prevent unlimited execution.
- The application is a scenario-driven terminal demonstration rather than a general-purpose interactive chat interface.

For diagrams, design decisions, memory explanations and results of all 14 scenarios, see [SUBMISSION.md](SUBMISSION.md).
