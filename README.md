# customer-genai

Multi-agent banking support demo (capstone Part 1) with:
- Classifier agent: routes to `positive_feedback`, `negative_feedback`, or `query`.
- CrewAI scaffold: dispatches to role-specific agents and tasks.
- Ticket persistence: minimal SQLite store (`data/support.db`) for ticket creation and lookup.

## Project layout
- `src/main/classifier.py`: classification logic returning labels and routes.
- `src/main/crew_scaffold.py`: CrewAI entrypoint; wires classifier → feedback/query agents.
- `src/main/openai_client_factory.py`: shared OpenAI client.
- `src/main/openai_llm_adapter.py`: adapts the OpenAI client to CrewAI’s `BaseLLM`.
- `src/main/support_store.py`: SQLite-backed ticket store (`init_db`, `create_ticket`, `get_ticket_status`).

## Setup
1) Python env: use the `cust-ai` conda env or your own.
2) Install deps:
```bash
pip install -r requirements.txt
```
3) Env vars (`.env` is read by Pydantic settings):
```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1   # optional if default
PYTHONPATH=.
```
Export them before running (e.g., `export $(cat .env | xargs)`).

## Running the scaffold
From repo root:
```bash
python -m src.main.crew_scaffold
```
Sample flow:
- Positive feedback → warm thank-you.
- Negative feedback → ticket number generated + stored; empathetic response with ticket.
- Query → ticket number extracted; status returned from SQLite or “not found” prompt.

Programmatic use:
```python
from src.main.crew_scaffold import handle_message
print(handle_message("My debit card replacement still hasn't arrived."))
print(handle_message("Could you check the status of ticket 650932?"))
```

## Notes
- Data persistence: tickets live in `data/support.db` (ignored by git). The DB is created on first run.
- Model: defaults to `gpt-4o-mini`; override via `handle_message(..., model="gpt-3.5-turbo")` or adjust in `_build_llm`.
- Keep your API keys private; do not commit `.env` or `data/`.
