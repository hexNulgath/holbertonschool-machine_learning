-- 103-average_weighted_score.sql
DROP PROCEDURE IF EXISTS ComputeAverageWeightedScoreForUser;
DELIMITER $$

CREATE PROCEDURE ComputeAverageWeightedScoreForUser(IN p_user_id INT)
BEGIN
    DECLARE v_total_weighted DECIMAL(10,2);
    DECLARE v_count INT;
    DECLARE v_average DECIMAL(10,2);

    -- Total weighted sum
    SELECT SUM(c.score * p.weight)
    INTO v_total_weighted
    FROM corrections c
    JOIN projects p ON c.project_id = p.id
    WHERE c.user_id = p_user_id;

    -- Count number of corrections
    SELECT COUNT(*)
    INTO v_count
    FROM corrections
    WHERE user_id = p_user_id;

    -- Avoid division by zero
    IF v_count = 0 OR v_total_weighted IS NULL THEN
        SET v_average = 0;
    ELSE
        SET v_average = v_total_weighted / v_count;
    END IF;

    -- Update users table
    UPDATE users
    SET average_score = v_average
    WHERE id = p_user_id;
END$$

DELIMITER ;


