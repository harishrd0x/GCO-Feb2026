# GCO-Feb 2026

## Setup

- Copy `.env.example` to `.env`
- Fill in your Azure OpenAI settings in `.env`
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_KEY`
  - `AZURE_API_VERSION`
  - `AZURE_OPENAI_MODEL`

## Run

```bash
py chatbot.py
```

Enable open-ended AI-wrapped responses (grounded in KB + DB facts):

```bash
CHATBOT_USE_AI=1 python chatbot.py
```

## Tests

```bash
python run_tests.py
```
