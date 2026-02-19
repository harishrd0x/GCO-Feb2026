# GCO-Feb 2026

## Setup

- Copy `.env.example` to `.env`
- Fill in your Azure OpenAI settings in `.env`
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_KEY`
  - `AZURE_API_VERSION`
  - `AZURE_OPENAI_MODEL` (this is your **Azure deployment name**; it can be `gpt-4o-mini` if that’s how you deployed it)

Notes:

- `.env` is ignored by git (secrets stay local).
- The bot answers using `knowledge_base.txt` and the local SQLite inventory database (generated from `inventory_setup.sql`).

## Run

```bash
python chatbot.py
```

Enable open-ended AI-wrapped responses (grounded in KB + DB facts):

```bash
CHATBOT_USE_AI=1 python chatbot.py
```

## Tests

```bash
python run_tests.py
```
