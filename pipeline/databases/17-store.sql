-- 17-store.sql
DELIMITER $$

CREATE TRIGGER decrease_quantity
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    UPDATE items
    SET quantity = quantity - orders.number
    WHERE name = orders.item_name;
END$$

DELIMITER ;