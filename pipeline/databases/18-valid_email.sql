-- 18-valid_email
DROP TRIGGER IF EXISTS valid_email;
DELIMITER $$

CREATE TRIGGER valid_email
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    IF NOT (OLD.email <=> NEW.email) THEN
        SET NEW.valid_email = 0;
    END IF;
END$$

DELIMITER ;
