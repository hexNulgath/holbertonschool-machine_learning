-- 103-average_weighted_score.sql
DROP PROCEDURE IF EXISTS ComputeAverageWeightedScoreForUser;
CREATE TABLE IF NOT EXISTS average_weighted_scores (
    user_id INT PRIMARY KEY,
    average_weighted_score DECIMAL(10,4) NULL
);

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
    FROM scores s
    WHERE s.user_id = p_user_id;

    IF total_weight = 0 THEN
        SET avg_score = NULL;
    ELSE
        SET avg_score = total_weighted / total_weight;
    END IF;

    INSERT INTO average_weighted_scores (user_id, average_weighted_score)
    VALUES (p_user_id, avg_score)
    ON DUPLICATE KEY UPDATE average_weighted_score = VALUES(average_weighted_score);
END$$
DELIMITER ;