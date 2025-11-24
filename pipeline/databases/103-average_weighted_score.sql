-- 103-average_weighted_score.sql
DROP PROCEDURE IF EXISTS ComputeAverageWeightedScoreForUser;

DELIMITER $$

CREATE PROCEDURE ComputeAverageWeightedScoreForUser(IN user_id INT)
BEGIN
    DECLARE total_weighted DECIMAL(30,6) DEFAULT 0;
    DECLARE total_weight DECIMAL(30,6) DEFAULT 0;
    DECLARE avg_score DECIMAL(10,2);

    SELECT
        COALESCE(SUM(score * weight), 0),
        COALESCE(SUM(weight), 0)
    INTO total_weighted, total_weight
    FROM scores
    WHERE scores.user_id = user_id;

    IF total_weight = 0 THEN
        SET avg_score = NULL;
    ELSE
        SET avg_score = ROUND(total_weighted / total_weight, 2);
    END IF;

    INSERT INTO average_weighted_scores (user_id, average_score)
    VALUES (user_id, avg_score)
    ON DUPLICATE KEY UPDATE average_score = avg_score;
END$$

DELIMITER ;

