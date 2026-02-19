# TechGear UK – Tri-Tier Chatbot (CLI)

A console-based customer service chatbot for TechGear UK that intelligently routes queries across three tiers:

1. **Knowledge Base** – Answers company information questions (address, hours, delivery, returns, contact) from `knowledge_base.txt`.
2. **Database** – Answers product inventory questions (stock, pricing, availability) by querying `inventory.db` via LLM function calling.
3. **Fallback** – Politely declines queries that fall outside both data sources.

## Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up the database (only needed if inventory.db is missing)
sqlite3 inventory.db < inventory_setup.sql

# 3. Export your OpenAI API key
export OPENAI_API_KEY="sk-..."

# 4. (Optional) Choose a model – defaults to gpt-4o-mini
export OPENAI_MODEL="gpt-4o-mini"
```

## Usage

```bash
python chatbot.py
```

The chatbot runs in a continuous terminal loop. Type your question and press Enter. Type `quit` or `exit` to leave.

### Example

```
============================================================
  TechGear UK – Customer Service Chatbot
============================================================
Type your question below. Type 'quit' or 'exit' to leave.

You: What is the office address?
Bot: Our office is located at 124 High Street, London, EC1A 1BB.

You: Is the Waterproof Commuter Jacket available in XL?
Bot: Yes, the Waterproof Commuter Jacket in XL is available with 3 in stock, priced at £85.00.

You: What is the capital of France?
Bot: I'm sorry, I cannot answer your query at the moment.
```

## Project Structure

```
├── chatbot.py             # Main application
├── knowledge_base.txt     # Static company information
├── inventory.db           # SQLite product inventory database
├── inventory_setup.sql    # SQL script to recreate the database
├── requirements.txt       # Python dependencies
└── README.md
```

## Environment Variables

| Variable         | Required | Default        | Description                |
|------------------|----------|----------------|----------------------------|
| `OPENAI_API_KEY` | Yes      | –              | Your OpenAI API key        |
| `OPENAI_MODEL`   | No       | `gpt-4o-mini`  | OpenAI model to use        |
