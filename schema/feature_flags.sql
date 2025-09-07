-- Feature flags and A/B testing schema

-- Feature flags table
CREATE TABLE IF NOT EXISTS feature_flags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'disabled',
    description TEXT,
    rollout_percentage DECIMAL(5,2) DEFAULT 0.0,
    ab_test_enabled BOOLEAN DEFAULT false,
    ab_test_variants JSONB DEFAULT '{}',
    target_users TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- A/B test assignments table
CREATE TABLE IF NOT EXISTS ab_test_assignments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    feature_name VARCHAR(100) NOT NULL REFERENCES feature_flags(name),
    variant VARCHAR(50) NOT NULL,
    assigned_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, feature_name)
);

-- A/B test results table
CREATE TABLE IF NOT EXISTS ab_test_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    feature_name VARCHAR(100) NOT NULL REFERENCES feature_flags(name),
    variant VARCHAR(50) NOT NULL,
    interaction_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- A/B test conversions table
CREATE TABLE IF NOT EXISTS ab_test_conversions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    feature_name VARCHAR(100) NOT NULL REFERENCES feature_flags(name),
    variant VARCHAR(50) NOT NULL,
    conversion_value DECIMAL(10,2) DEFAULT 1.0,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Feature analytics table
CREATE TABLE IF NOT EXISTS feature_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_name VARCHAR(100) NOT NULL REFERENCES feature_flags(name),
    user_id UUID NOT NULL,
    interaction_type VARCHAR(50) NOT NULL,
    count INTEGER DEFAULT 1,
    last_interaction TIMESTAMP DEFAULT NOW(),
    UNIQUE(feature_name, user_id, interaction_type)
);

-- Performance tracking table
CREATE TABLE IF NOT EXISTS feature_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_name VARCHAR(100) NOT NULL REFERENCES feature_flags(name),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    variant VARCHAR(50),
    timestamp TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Rollback events table
CREATE TABLE IF NOT EXISTS feature_rollbacks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_name VARCHAR(100) NOT NULL REFERENCES feature_flags(name),
    previous_status VARCHAR(20) NOT NULL,
    new_status VARCHAR(20) NOT NULL,
    reason TEXT NOT NULL,
    triggered_by UUID,
    rollback_type VARCHAR(50) NOT NULL, -- 'manual', 'automatic', 'emergency'
    timestamp TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_feature_flags_name ON feature_flags(name);
CREATE INDEX IF NOT EXISTS idx_feature_flags_status ON feature_flags(status);
CREATE INDEX IF NOT EXISTS idx_ab_test_assignments_user_feature ON ab_test_assignments(user_id, feature_name);
CREATE INDEX IF NOT EXISTS idx_ab_test_results_feature_timestamp ON ab_test_results(feature_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_ab_test_conversions_feature_timestamp ON ab_test_conversions(feature_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_feature_analytics_feature_user ON feature_analytics(feature_name, user_id);
CREATE INDEX IF NOT EXISTS idx_feature_performance_feature_timestamp ON feature_performance(feature_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_feature_rollbacks_feature_timestamp ON feature_rollbacks(feature_name, timestamp);

-- Row Level Security (RLS) policies
ALTER TABLE feature_flags ENABLE ROW LEVEL SECURITY;
ALTER TABLE ab_test_assignments ENABLE ROW LEVEL SECURITY;
ALTER TABLE ab_test_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE ab_test_conversions ENABLE ROW LEVEL SECURITY;
ALTER TABLE feature_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE feature_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE feature_rollbacks ENABLE ROW LEVEL SECURITY;

-- Policies for feature flags (admin access only)
CREATE POLICY "Admin can manage feature flags" ON feature_flags
    FOR ALL USING (auth.jwt() ->> 'role' = 'admin');

-- Policies for user-specific data
CREATE POLICY "Users can view their own A/B test data" ON ab_test_assignments
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can view their own test results" ON ab_test_results
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can view their own conversions" ON ab_test_conversions
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can view their own analytics" ON feature_analytics
    FOR SELECT USING (user_id = auth.uid());

-- Service role can insert/update all data
CREATE POLICY "Service role can manage all data" ON ab_test_assignments
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

CREATE POLICY "Service role can manage results" ON ab_test_results
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

CREATE POLICY "Service role can manage conversions" ON ab_test_conversions
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

CREATE POLICY "Service role can manage analytics" ON feature_analytics
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

CREATE POLICY "Service role can manage performance" ON feature_performance
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

CREATE POLICY "Service role can manage rollbacks" ON feature_rollbacks
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');