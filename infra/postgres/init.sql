CREATE SCHEMA IF NOT EXISTS agent_system;

CREATE TABLE agent_system.experiments (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hypothesis_text TEXT,
    generated_code TEXT,
    status VARCHAR(20) DEFAULT 'pending', -- pending, running, completed, failed
    k8s_job_name VARCHAR(100),
    mlflow_run_id VARCHAR(100),
    best_cv_score FLOAT,
    error_log TEXT
);
