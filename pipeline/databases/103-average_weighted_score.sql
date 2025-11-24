-- 103-average_weighted_score.sql
DROP PROCEDURE IF EXISTS ComputeAverageWeightedScoreForUser;
DELIMITER $$

CREATE PROCEDURE ComputeAverageWeightedScoreForUser(IN p_user_id INT)
BEGIN
    DECLARE total_weighted DECIMAL(30,6) DEFAULT 0;
    DECLARE total_weight DECIMAL(30,6) DEFAULT 0;
    DECLARE avg_score DECIMAL(10,4);

    SELECT
        COALESCE(SUM(s.score * s.weight), 0),
        COALESCE(SUM(s.weight), 0)
    INTO total_weighted, total_weight
    FROM corrections s
    WHERE s.user_id = p_user_id;

    IF total_weight = 0 THEN
        SET avg_score = 0;
    ELSE
        SET avg_score = total_weighted / total_weight;
    END IF;

    UPDATE users
    SET average_score = avg_score
    WHERE id = p_user_id;
END$$

DELIMITER ;

