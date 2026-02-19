"""
Tri-Tier Chatbot CLI Application for TechGear UK.

Routes user queries to one of three tiers:
  1. Knowledge Base – static company information from knowledge_base.txt
  2. Database       – product inventory lookups via SQLite + LLM function calling
  3. Fallback       – polite decline for anything else
"""

import json
import os
import sqlite3
import sys

from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "knowledge_base.txt")
DB_PATH = os.path.join(BASE_DIR, "inventory.db")

FALLBACK_MESSAGE = "I'm sorry, I cannot answer your query at the moment."


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------

def load_knowledge_base() -> str:
    with open(KB_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Run: sqlite3 inventory.db < inventory_setup.sql"
        )
    return sqlite3.connect(DB_PATH)


def query_inventory(item_name: str | None = None, size: str | None = None) -> str:
    """Query product_inventory and return a JSON string of matching rows."""
    conn = _get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    clauses: list[str] = []
    params: list[str] = []

    if item_name:
        clauses.append("item_name LIKE ?")
        params.append(f"%{item_name}%")
    if size:
        clauses.append("UPPER(size) = ?")
        params.append(size.strip().upper())

    sql = "SELECT item_name, size, stock_count, price_gbp FROM product_inventory"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)

    cursor.execute(sql, params)
    rows = []
    for row in cursor.fetchall():
        record = dict(row)
        record["price_gbp"] = f"£{record['price_gbp']:.2f}"
        rows.append(record)
    conn.close()

    return json.dumps(rows, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool definitions for function calling
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_inventory",
            "description": (
                "Search the product inventory database for stock levels, "
                "prices, and availability. Use this for any question about "
                "products, stock, or pricing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Product name (or partial name) to search for.",
                    },
                    "size": {
                        "type": "string",
                        "description": "Size to filter by.",
                        "enum": ["S", "M", "L", "XL"],
                    },
                },
                "required": [],
            },
        },
    }
]

TOOL_MAP = {
    "query_inventory": query_inventory,
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    kb_content = load_knowledge_base()
    return (
        "You are a helpful customer service assistant for TechGear UK.\n"
        "You have access to two data sources:\n\n"
        "1. **Knowledge Base** – company information shown below.\n"
        "2. **Product Inventory Database** – accessed via the `query_inventory` tool.\n\n"
        "## Knowledge Base\n"
        f"{kb_content}\n\n"
        "## Rules\n"
        "- For company-related questions (address, hours, delivery, returns, contact), "
        "answer directly from the Knowledge Base above.\n"
        "- For product, stock, price, or availability questions, call the "
        "`query_inventory` tool with the relevant parameters.\n"
        "- For anything else that is not covered by the Knowledge Base or the "
        "inventory database, respond EXACTLY with:\n"
        f'  "{FALLBACK_MESSAGE}"\n'
        "- Always use UK English spelling and conventions.\n"
        "- Display all monetary values in GBP using the £ symbol.\n"
        "- When reporting stock for a specific item and size, clearly state the "
        "stock count. If stock_count is 0, say the item is out of stock.\n"
        "- Keep responses concise and helpful.\n"
    )


# ---------------------------------------------------------------------------
# Chat engine
# ---------------------------------------------------------------------------

class ChatBot:
    def __init__(self):
        self.client = OpenAI()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.system_prompt = build_system_prompt()
        self.messages: list[dict] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def _call_llm(self) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        return response.choices[0].message

    def _handle_tool_calls(self, assistant_message) -> str | None:
        """Execute any tool calls and return the final assistant reply."""
        if not assistant_message.tool_calls:
            return assistant_message.content

        self.messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            fn = TOOL_MAP.get(fn_name)
            if fn is None:
                result = json.dumps({"error": f"Unknown function: {fn_name}"})
            else:
                result = fn(**fn_args)

            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        follow_up = self._call_llm()
        return self._handle_tool_calls(follow_up)

    def ask(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        assistant_message = self._call_llm()
        reply = self._handle_tool_calls(assistant_message)

        if reply:
            self.messages.append({"role": "assistant", "content": reply})

        return reply or FALLBACK_MESSAGE


# ---------------------------------------------------------------------------
# CLI loop
# ---------------------------------------------------------------------------

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is not set.\n"
            "Export it before running: export OPENAI_API_KEY='sk-...'"
        )
        sys.exit(1)

    print("=" * 60)
    print("  TechGear UK – Customer Service Chatbot")
    print("=" * 60)
    print("Type your question below. Type 'quit' or 'exit' to leave.\n")

    bot = ChatBot()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        try:
            reply = bot.ask(user_input)
            print(f"Bot: {reply}\n")
        except Exception as exc:
            print(f"Bot: {FALLBACK_MESSAGE}\n")
            print(f"[Debug] Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
