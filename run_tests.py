from __future__ import annotations

import json
import os
from pathlib import Path

from chatbot import Chatbot, FALLBACK_MESSAGE


def main() -> None:
    # Ensure deterministic offline behaviour for the test suite.
    os.environ["CHATBOT_USE_LLM"] = "0"

    base_dir = Path(__file__).resolve().parent
    suite_path = base_dir / "test_suite.json"
    tests = json.loads(suite_path.read_text(encoding="utf-8"))

    bot = Chatbot(base_dir=base_dir)

    failures: list[str] = []
    for t in tests:
        q = t["q"]
        expected = t["target"]
        ttype = t["type"]

        got = bot.get_response(q)
        if ttype == "Fallback":
            if got != FALLBACK_MESSAGE:
                failures.append(f'#{t["id"]} expected fallback, got: {got!r}')
            continue

        if expected not in got:
            failures.append(
                f'#{t["id"]} expected substring {expected!r} in response, got: {got!r}'
            )

    if failures:
        for f in failures:
            print(f)
        raise SystemExit(1)

    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()

