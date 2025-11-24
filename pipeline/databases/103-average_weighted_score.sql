-- 103-average_weighted_score.sql
DROP PROCEDURE IF EXISTS ComputeAverageWeightedScoreForUser;
DELIMITER $$

CREATE PROCEDURE ComputeAverageWeightedScoreForUser(IN p_user_id INT)
BEGIN
    DECLARE v_total_score DECIMAL(10,2);

    -- Calculate weighted score sum
    SELECT SUM(c.score * p.weight)
    INTO v_total_score
    FROM corrections c
    JOIN projects p ON c.project_id = p.id
    WHERE c.user_id = p_user_id;

    -- If null (no corrections), set to 0
    SET v_total_score = IFNULL(v_total_score, 0);

    -- Update the users table
    UPDATE users
    SET average_score = v_total_score
    WHERE id = p_user_id;
END$$

DELIMITER ;

