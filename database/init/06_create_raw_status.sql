CREATE TABLE IF NOT EXISTS raw.status (
    status_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) UNIQUE NOT NULL,
    count INTEGER,
    quarter VARCHAR(10),
    satisfaction_score INTEGER,
    customer_status VARCHAR(20),
    churn_label VARCHAR(10),
    churn_value INTEGER,
    churn_score INTEGER,
    cltv INTEGER,
    churn_category VARCHAR(50),
    churn_reason VARCHAR(100),

    CONSTRAINT fk_status_customer
        FOREIGN KEY (customer_id)
        REFERENCES raw.demographics (customer_id)
);