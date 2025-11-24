-- 6-avg_temperatures.sql
SELECT city, AVG(value) AS avg_temperature
FROM temperatures
GROUP BY city
ORDER BY avg_temperature DESC;