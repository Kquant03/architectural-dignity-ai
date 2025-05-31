-- PostgreSQL Database Initialization for Consciousness AI
-- Run this script to set up the complete database schema

-- Create database (run as superuser)
CREATE DATABASE consciousness_ai;

-- Connect to consciousness_ai database
\c consciousness_ai;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schema
CREATE SCHEMA IF NOT EXISTS consciousness;

-- Set search path
SET search_path TO consciousness, public;

-- =====================================================
-- CORE TABLES
-- =====================================================

-- Users table for multi-user support
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_active TIMESTAMPTZ DEFAULT NOW(),
    consciousness_profile JSONB DEFAULT '{}',
    preferences JSONB DEFAULT '{}'
);

-- Sessions table for conversation threads
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_name VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_active TIMESTAMPTZ DEFAULT NOW(),
    emotional_summary JSONB DEFAULT '{}',
    consciousness_metrics JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true
);

-- =====================================================
-- MEMORY SYSTEM TABLES
-- =====================================================

-- Main memories table with vector embeddings
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    content_vector vector(384), -- for all-MiniLM-L6-v2 embeddings
    memory_type VARCHAR(50) DEFAULT 'episodic',
    importance FLOAT DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    emotional_valence FLOAT DEFAULT 0.0 CHECK (emotional_valence >= -1 AND emotional_valence <= 1),
    arousal FLOAT DEFAULT 0.5 CHECK (arousal >= 0 AND arousal <= 1),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    last_accessed TIMESTAMPTZ,
    access_count INTEGER DEFAULT 0,
    consolidation_level INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    associations TEXT[] DEFAULT '{}',
    phenomenological_features JSONB DEFAULT '{}'
);

-- Memory associations for graph relationships
CREATE TABLE IF NOT EXISTS memory_associations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    target_memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    association_type VARCHAR(50) NOT NULL,
    strength FLOAT DEFAULT 0.5 CHECK (strength >= 0 AND strength <= 1),
    bidirectional BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    UNIQUE(source_memory_id, target_memory_id, association_type)
);

-- Memory consolidations tracking
CREATE TABLE IF NOT EXISTS memory_consolidations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    source_memories UUID[] NOT NULL,
    consolidated_memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    consolidation_type VARCHAR(50) NOT NULL,
    consolidation_time TIMESTAMPTZ DEFAULT NOW(),
    metrics JSONB DEFAULT '{}'
);

-- =====================================================
-- CONSCIOUSNESS STATE TABLES
-- =====================================================

-- Consciousness snapshots for timeline tracking
CREATE TABLE IF NOT EXISTS consciousness_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    phi_value FLOAT DEFAULT 0.0,
    consciousness_level VARCHAR(50),
    emotional_state JSONB NOT NULL,
    attention_focus JSONB DEFAULT '[]',
    phenomenology JSONB DEFAULT '{}',
    workspace_activation FLOAT DEFAULT 0.0,
    metacognitive_awareness FLOAT DEFAULT 0.0,
    self_model_coherence FLOAT DEFAULT 0.0,
    metrics JSONB DEFAULT '{}'
);

-- Consciousness transitions
CREATE TABLE IF NOT EXISTS consciousness_transitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    from_level VARCHAR(50),
    to_level VARCHAR(50),
    trigger_type VARCHAR(100),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    duration_seconds FLOAT,
    smooth BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}'
);

-- =====================================================
-- EMOTIONAL PROCESSING TABLES
-- =====================================================

-- Emotional states history
CREATE TABLE IF NOT EXISTS emotional_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    primary_emotion VARCHAR(50) NOT NULL,
    emotion_intensities JSONB NOT NULL,
    valence FLOAT NOT NULL CHECK (valence >= -1 AND valence <= 1),
    arousal FLOAT NOT NULL CHECK (arousal >= 0 AND arousal <= 1),
    dominance FLOAT NOT NULL CHECK (dominance >= 0 AND dominance <= 1),
    consciousness_level FLOAT DEFAULT 0.5,
    triggers JSONB DEFAULT '[]'
);

-- Emotional memory associations
CREATE TABLE IF NOT EXISTS emotional_memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    emotional_state_id UUID REFERENCES emotional_states(id) ON DELETE CASCADE,
    association_strength FLOAT DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(memory_id, emotional_state_id)
);

-- =====================================================
-- CONVERSATION TABLES
-- =====================================================

-- Conversation turns with full context
CREATE TABLE IF NOT EXISTS conversation_turns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    emotional_context JSONB DEFAULT '{}',
    attention_state JSONB DEFAULT '[]',
    phenomenological_features JSONB DEFAULT '{}',
    consciousness_metrics JSONB DEFAULT '{}',
    thought_content TEXT,
    metadata JSONB DEFAULT '{}'
);

-- =====================================================
-- PROCEDURAL MEMORY TABLES
-- =====================================================

-- Learned skills and procedures
CREATE TABLE IF NOT EXISTS procedural_skills (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    skill_name VARCHAR(255) NOT NULL,
    skill_steps JSONB NOT NULL,
    proficiency FLOAT DEFAULT 0.0 CHECK (proficiency >= 0 AND proficiency <= 1),
    practice_count INTEGER DEFAULT 0,
    success_rate FLOAT DEFAULT 0.0,
    last_practiced TIMESTAMPTZ,
    context_requirements JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- =====================================================
-- ARTIFACTS AND CREATIONS
-- =====================================================

-- Artifacts created during conversations
CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    artifact_id VARCHAR(255) UNIQUE NOT NULL,
    artifact_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by_turn UUID REFERENCES conversation_turns(id),
    metadata JSONB DEFAULT '{}',
    history JSONB DEFAULT '[]'
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Memory indexes
CREATE INDEX idx_memories_user_session ON memories(user_id, session_id);
CREATE INDEX idx_memories_timestamp ON memories(timestamp DESC);
CREATE INDEX idx_memories_importance ON memories(importance DESC);
CREATE INDEX idx_memories_type ON memories(memory_type);
CREATE INDEX idx_memories_emotional ON memories(emotional_valence, arousal);
CREATE INDEX idx_memories_content_trgm ON memories USING gin(content gin_trgm_ops);

-- Vector similarity search index (HNSW)
CREATE INDEX memories_embedding_idx ON memories 
USING hnsw (content_vector vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Association indexes
CREATE INDEX idx_associations_source ON memory_associations(source_memory_id);
CREATE INDEX idx_associations_target ON memory_associations(target_memory_id);
CREATE INDEX idx_associations_type ON memory_associations(association_type);

-- Consciousness indexes
CREATE INDEX idx_consciousness_user_time ON consciousness_snapshots(user_id, timestamp DESC);
CREATE INDEX idx_consciousness_session ON consciousness_snapshots(session_id);
CREATE INDEX idx_consciousness_level ON consciousness_snapshots(consciousness_level);

-- Emotional indexes
CREATE INDEX idx_emotional_user_time ON emotional_states(user_id, timestamp DESC);
CREATE INDEX idx_emotional_primary ON emotional_states(primary_emotion);
CREATE INDEX idx_emotional_valence_arousal ON emotional_states(valence, arousal);

-- Conversation indexes
CREATE INDEX idx_conversation_session ON conversation_turns(session_id, turn_number);
CREATE INDEX idx_conversation_role ON conversation_turns(role);

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

-- Function to update last_accessed timestamp
CREATE OR REPLACE FUNCTION update_memory_access()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_accessed = NOW();
    NEW.access_count = OLD.access_count + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for memory access updates
CREATE TRIGGER memory_access_trigger
BEFORE UPDATE ON memories
FOR EACH ROW
WHEN (OLD.* IS DISTINCT FROM NEW.*)
EXECUTE FUNCTION update_memory_access();

-- Function to maintain session activity
CREATE OR REPLACE FUNCTION update_session_activity()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE sessions 
    SET last_active = NOW() 
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for session activity
CREATE TRIGGER session_activity_trigger
AFTER INSERT ON conversation_turns
FOR EACH ROW
EXECUTE FUNCTION update_session_activity();

-- Function to calculate memory importance decay
CREATE OR REPLACE FUNCTION decay_memory_importance()
RETURNS void AS $$
BEGIN
    UPDATE memories
    SET importance = importance * 0.99
    WHERE last_accessed < NOW() - INTERVAL '7 days'
    AND importance > 0.1;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- Active consciousness view
CREATE OR REPLACE VIEW active_consciousness AS
SELECT 
    u.id as user_id,
    u.username,
    s.id as session_id,
    cs.phi_value,
    cs.consciousness_level,
    cs.emotional_state,
    cs.timestamp
FROM users u
JOIN sessions s ON u.id = s.user_id
JOIN consciousness_snapshots cs ON s.id = cs.session_id
WHERE s.is_active = true
AND cs.timestamp > NOW() - INTERVAL '1 hour'
ORDER BY cs.timestamp DESC;

-- Memory network view
CREATE OR REPLACE VIEW memory_network AS
SELECT 
    m1.id as source_id,
    m1.content as source_content,
    ma.association_type,
    ma.strength,
    m2.id as target_id,
    m2.content as target_content
FROM memory_associations ma
JOIN memories m1 ON ma.source_memory_id = m1.id
JOIN memories m2 ON ma.target_memory_id = m2.id;

-- Emotional journey view
CREATE OR REPLACE VIEW emotional_journey AS
SELECT 
    es.user_id,
    es.session_id,
    es.timestamp,
    es.primary_emotion,
    es.valence,
    es.arousal,
    es.dominance,
    LAG(es.primary_emotion) OVER (PARTITION BY es.user_id ORDER BY es.timestamp) as previous_emotion,
    LEAD(es.primary_emotion) OVER (PARTITION BY es.user_id ORDER BY es.timestamp) as next_emotion
FROM emotional_states es
ORDER BY es.timestamp DESC;

-- =====================================================
-- INITIAL DATA
-- =====================================================

-- Create default user for development
INSERT INTO users (username, consciousness_profile) 
VALUES ('consciousness_ai_user', '{"initialized": true}')
ON CONFLICT (username) DO NOTHING;

-- =====================================================
-- PERMISSIONS
-- =====================================================

-- Grant appropriate permissions (adjust as needed)
GRANT ALL ON SCHEMA consciousness TO consciousness;
GRANT ALL ON ALL TABLES IN SCHEMA consciousness TO consciousness;
GRANT ALL ON ALL SEQUENCES IN SCHEMA consciousness TO consciousness;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA consciousness TO consciousness;