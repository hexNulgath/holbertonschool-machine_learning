-- 103-average_weighted_score.sql
DROP PROCEDURE IF EXISTS ComputeAverageWeightedScoreForUser;
DELIMITER $$

CREATE PROCEDURE ComputeAverageWeightedScoreForUser(IN p_user_id INT)
BEGIN
    DECLARE v_weighted_sum DECIMAL(10,2);
    DECLARE v_total_weight DECIMAL(10,2);
    DECLARE v_average DECIMAL(10,2);

    -- Calculate weighted sum and total weight
    SELECT 
        SUM(c.score * p.weight),
        SUM(p.weight)
    INTO v_weighted_sum, v_total_weight
    FROM corrections c
    JOIN projects p ON c.project_id = p.id
    WHERE c.user_id = p_user_id;

    -- Handle no corrections case
    IF v_total_weight IS NULL OR v_total_weight = 0 THEN
        SET v_average = 0;
    ELSE
        SET v_average = v_weighted_sum / v_total_weight;
    END IF;

    -- Update the user table
    UPDATE users
    SET average_score = v_average
    WHERE id = p_user_id;

END$$

DELIMITER ;



