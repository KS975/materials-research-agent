-- Execute ONLY in a separately approved Agent Runtime database.
-- Never run this in `materials`.

CREATE TABLE IF NOT EXISTS agent_run (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    run_id VARCHAR(64) NOT NULL UNIQUE,
    user_id VARCHAR(255) NOT NULL,
    company_id VARCHAR(255) NOT NULL,
    user_message TEXT NOT NULL,
    status VARCHAR(32) NOT NULL,
    intent VARCHAR(128) NULL,
    tool_name VARCHAR(128) NULL,
    tool_result_json JSON NULL,
    answer LONGTEXT NULL,
    error_message LONGTEXT NULL,
    created_at DATETIME NOT NULL,
    finished_at DATETIME NULL,
    INDEX idx_agent_run_company_created (company_id, created_at),
    INDEX idx_agent_run_user_created (user_id, created_at)
);
