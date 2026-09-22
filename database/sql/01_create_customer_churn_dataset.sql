CREATE OR REPLACE VIEW analytics.customer_churn_dataset AS
SELECT
    -- Customer identifier
    d.customer_id,

    -- Demographics
    d.gender,
    d.age,
    d.under_30,
    d.senior_citizen,
    d.married,
    d.dependents,
    d.number_of_dependents,

    -- Location
    l.country,
    l.state,
    l.city,
    l.zip_code,
    l.latitude,
    l.longitude,
    p.population,

    -- Services and customer relationship
    s.referred_a_friend,
    s.number_of_referrals,
    s.tenure_in_months,
    s.offer,
    s.phone_service,
    s.avg_monthly_long_distance_charges,
    s.multiple_lines,
    s.internet_service,
    s.internet_type,
    s.avg_monthly_gb_download,
    s.online_security,
    s.online_backup,
    s.device_protection_plan,
    s.premium_tech_support,
    s.streaming_tv,
    s.streaming_movies,
    s.streaming_music,
    s.unlimited_data,
    s.contract,
    s.paperless_billing,
    s.payment_method,

    -- Financial metrics
    s.monthly_charge,
    s.total_charges,
    s.total_refunds,
    s.total_extra_data_charges,
    s.total_long_distance_charges,
    s.total_revenue,

    -- Machine Learning target
    st.churn_value

FROM raw.demographics AS d

INNER JOIN raw.location AS l
    ON d.customer_id = l.customer_id

INNER JOIN raw.population AS p
    ON l.zip_code = p.zip_code

INNER JOIN raw.services AS s
    ON d.customer_id = s.customer_id

INNER JOIN raw.status AS st
    ON d.customer_id = st.customer_id;