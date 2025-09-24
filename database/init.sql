CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(255) UNIQUE NOT NULL,
    filename VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'processing',
    error_message TEXT,
    total_pages INTEGER,
    total_chunks INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chat sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Query runs with telemetry
CREATE TABLE IF NOT EXISTS query_runs (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    retrieved_chunks JSONB,
    similarity_scores JSONB,
    embedding_time FLOAT,
    retrieval_time FLOAT,
    llm_time FLOAT,
    total_time FLOAT,
    sources_found BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_documents_doc_id ON documents(doc_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_query_runs_session_id ON query_runs(session_id);
