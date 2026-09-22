CREATE TABLE IF NOT EXISTS ml.customer_churn_scores (
    customer_id VARCHAR(50) PRIMARY KEY,
    churn_probability NUMERIC(8, 6) NOT NULL,
    risk_percentile NUMERIC(6, 2) NOT NULL,
    risk_decile INTEGER NOT NULL,
    risk_band VARCHAR(20) NOT NULL,
    prediction_threshold NUMERIC(4, 2) NOT NULL,
    predicted_churn INTEGER NOT NULL,
    actual_churn_value INTEGER,
    scored_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_score_customer
        FOREIGN KEY (customer_id)
        REFERENCES raw.demographics (customer_id),

    CONSTRAINT chk_probability
        CHECK (churn_probability BETWEEN 0 AND 1),

    CONSTRAINT chk_risk_decile
        CHECK (risk_decile BETWEEN 1 AND 10),

    CONSTRAINT chk_risk_band
        CHECK (risk_band IN ('Low', 'Medium', 'High', 'Critical')),

    CONSTRAINT chk_predicted_churn
        CHECK (predicted_churn IN (0, 1)),

    CONSTRAINT chk_actual_churn
        CHECK (actual_churn_value IS NULL OR actual_churn_value IN (0, 1))
);