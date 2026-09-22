CREATE TABLE IF NOT EXISTS raw.demographics (
    customer_id VARCHAR(50) PRIMARY KEY,
    count INTEGER,
    gender VARCHAR(20),
    age INTEGER,
    under_30 VARCHAR(10),
    senior_citizen VARCHAR(10),
    married VARCHAR(10),
    dependents VARCHAR(10),
    number_of_dependents INTEGER
);