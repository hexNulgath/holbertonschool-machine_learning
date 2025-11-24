-- 16-glam_rock.sql
SELECT band_name,
       (IFNULL(split, 2020) - formed) AS lifespan
FROM metal_bands
WHERE style LIKE '%glam rock%'
ORDER BY lifespan DESC;