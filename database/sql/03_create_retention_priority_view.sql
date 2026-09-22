CREATE OR REPLACE VIEW analytics.retention_priority AS
SELECT
    s.customer_id,
    s.churn_probability,
    s.risk_percentile,
    s.risk_decile,
    s.risk_band,
    s.predicted_churn,

    a.age,
    a.senior_citizen,
    a.dependents,
    a.tenure_in_months,
    a.contract,
    a.internet_service,
    a.internet_type,
    a.monthly_charge,
    a.total_revenue,
    a.referred_a_friend,
    a.number_of_referrals,

    CASE
        WHEN s.risk_band = 'Critical' THEN 1
        WHEN s.risk_band = 'High' THEN 2
        WHEN s.risk_band = 'Medium' THEN 3
        ELSE 4
    END AS priority_order,

    CASE
        WHEN s.risk_band = 'Critical'
            THEN 'Immediate retention action'
        WHEN s.risk_band = 'High'
            THEN 'Proactive retention contact'
        WHEN s.risk_band = 'Medium'
            THEN 'Monitor and engage'
        ELSE
            'Standard relationship'
    END AS recommended_action

FROM ml.customer_churn_scores AS s

INNER JOIN analytics.customer_churn_dataset AS a
    ON s.customer_id = a.customer_id;