CREATE TABLE IF NOT EXISTS raw.location (
    location_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) UNIQUE NOT NULL,
    count INTEGER,
    country VARCHAR(100),
    state VARCHAR(100),
    city VARCHAR(100),
    zip_code VARCHAR(10) NOT NULL,
    lat_long VARCHAR(50),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,

    CONSTRAINT fk_location_customer
        FOREIGN KEY (customer_id)
        REFERENCES raw.demographics (customer_id),

    CONSTRAINT fk_location_zip_code
        FOREIGN KEY (zip_code)
        REFERENCES raw.population (zip_code)
);