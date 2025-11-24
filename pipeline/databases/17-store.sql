-- 17-store.sql
DROP TRIGGER IF EXISTS store;

CREATE TRIGGER store
AFTER INSERT ON orders
FOR EACH ROW
    UPDATE items
    SET quantity = quantity - NEW.number
    WHERE name = NEW.item_name;
