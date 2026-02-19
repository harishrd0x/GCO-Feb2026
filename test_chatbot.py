"""
Offline tests for the chatbot's knowledge base and database layers.
These validate that the data sources return the expected information
without requiring an OpenAI API key.
"""

import json
import chatbot


def test_knowledge_base_loaded():
    kb = chatbot.load_knowledge_base()
    assert "124 High Street" in kb
    assert "EC1A 1BB" in kb
    assert "09:00 - 18:00" in kb
    assert "£5.99" in kb
    assert "30 days" in kb
    assert "support@techgear.co.uk" in kb
    print("[PASS] Knowledge base loaded with all expected data")


def test_db_jacket_xl():
    result = json.loads(chatbot.query_inventory("Waterproof Commuter Jacket", "XL"))
    assert len(result) == 1
    assert result[0]["stock_count"] == 3
    assert result[0]["price_gbp"] == "£85.00"
    print("[PASS] Jacket XL: 3 in stock at £85.00")


def test_db_jacket_m_out_of_stock():
    result = json.loads(chatbot.query_inventory("Waterproof Commuter Jacket", "M"))
    assert len(result) == 1
    assert result[0]["stock_count"] == 0
    print("[PASS] Jacket M: out of stock (0)")


def test_db_hoodie_m():
    result = json.loads(chatbot.query_inventory("Tech-Knit Hoodie", "M"))
    assert len(result) == 1
    assert result[0]["stock_count"] == 10
    print("[PASS] Hoodie M: 10 in stock")


def test_db_running_tee_price():
    result = json.loads(chatbot.query_inventory("Dry-Fit Running Tee"))
    assert len(result) == 2
    for row in result:
        assert row["price_gbp"] == "£25.00"
    print("[PASS] Running Tee: £25.00 for all sizes")


def test_db_all_products():
    result = json.loads(chatbot.query_inventory())
    assert len(result) == 8
    print("[PASS] Full inventory: 8 rows returned")


if __name__ == "__main__":
    test_knowledge_base_loaded()
    test_db_jacket_xl()
    test_db_jacket_m_out_of_stock()
    test_db_hoodie_m()
    test_db_running_tee_price()
    test_db_all_products()
    print("\nAll offline tests passed.")
