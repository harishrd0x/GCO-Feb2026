# GCO-Feb 2026 — Tri-Tier Chatbot (CLI)

## Run

```bash
python3 chatbot.py
```

Type `exit` or `quit` to leave the loop.

## Data sources

- **Knowledge base**: `knowledge_base.txt`
- **Inventory DB**: `./inventory.db` (SQLite)
  - If missing/corrupt, the app rebuilds it using `inventory_setup.sql`.

## Optional AI responses (tool-calling)

By default the chatbot runs offline with deterministic routing and answers grounded in the KB/DB.

If you want more open-ended, UK-English responses using an LLM **while still querying the DB/KB**, copy `.env.example` to `.env` and fill in your key (the app will auto-load `.env`):

```bash
cp .env.example .env
```

Then set:

- `CHATBOT_USE_LLM=1`
- **Azure OpenAI** (preferred if set): `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_API_VERSION`, and optionally `AZURE_OPENAI_DEPLOYMENT`
- or **OpenAI-compatible**: `OPENAI_API_KEY`, and optionally `OPENAI_MODEL`, `OPENAI_BASE_URL`

The fallback message for unknown questions remains:

`I'm sorry, I cannot answer your query at the moment.`

## Tests

```bash
python3 run_tests.py
```
