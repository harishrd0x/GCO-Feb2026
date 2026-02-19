BEGIN TRANSACTION;
DROP TABLE IF EXISTS product_inventory;
CREATE TABLE product_inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_name TEXT NOT NULL,
    size TEXT NOT NULL,
    stock_count INTEGER NOT NULL,
    price_gbp DECIMAL(10, 2) NOT NULL
);

INSERT INTO product_inventory (item_name, size, stock_count, price_gbp) VALUES
('Waterproof Commuter Jacket', 'S', 5, 85.00),
('Waterproof Commuter Jacket', 'M', 0, 85.00),
('Waterproof Commuter Jacket', 'L', 12, 85.00),
('Waterproof Commuter Jacket', 'XL', 3, 85.00),
('Tech-Knit Hoodie', 'M', 10, 45.00),
('Tech-Knit Hoodie', 'S', 0, 45.00),
('Dry-Fit Running Tee', 'L', 20, 25.00),
('Dry-Fit Running Tee', 'M', 15, 25.00);
COMMIT;