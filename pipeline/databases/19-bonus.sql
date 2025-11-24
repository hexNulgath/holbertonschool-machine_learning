-- 19-bonus
DROP PROCEDURE IF EXISTS AddBonus;
DELIMITER $$

CREATE PROCEDURE AddBonus(
    user_id int,
    project_name varchar(255),
    score int 
)
BEGIN
    DECLARE proj_id INT;
    SELECT id INTO proj_id FROM projects WHERE name = project_name;
    IF proj_id IS NOT NULL THEN
        INSERT INTO bonuses (user_id, project_id, score) VALUES (user_id, proj_id, score);
    END IF;
    INSERT INTO bonuses (user_id, project_id, score) VALUES (user_id, NULL, score);
END$$

DELIMITER ;