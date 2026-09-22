CREATE TABLE IF NOT EXISTS raw.population (
    id INTEGER PRIMARY KEY,
    zip_code VARCHAR(10) UNIQUE NOT NULL,
    population INTEGER
);