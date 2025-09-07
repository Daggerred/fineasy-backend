-- Usage analytics and performance monitoring schema

-- Usage analytics table
CREATE TABLE IF NOT EXISTS usage_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    user_id UUID NOT NULL,
    business_id UUID,
    variant VARCHAR(50),
    properties JSONB DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Performance alerts table
CREATE TABLE IF NOT EXISTS performance_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    current_value DECIMAL(15,4) NOT NULL,
    threshold_value DECIMAL(15,4) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    resolved BOOLEAN DEFAULT false,
    resolved_at TIMESTAMP,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_usage_analytics_feature_timestamp ON usage_analytics(feature_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_analytics_user_timestamp ON usage_analytics(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_analytics_event_type ON usage_analytics(event_type);
CREATE INDEX IF NOT EXISTS idx_usage_analytics_variant ON usage_analytics(variant) WHERE variant IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_performance_alerts_feature_timestamp ON performance_alerts(feature_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_alerts_severity ON performance_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_performance_alerts_resolved ON performance_alerts(resolved);

-- Row Level Security (RLS) policies
ALTER TABLE usage_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_alerts ENABLE ROW LEVEL SECURITY;

-- Policies for usage analytics
CREATE POLICY "Users can view their own analytics" ON usage_analytics
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Service role can manage analytics" ON usage_analytics
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- Policies for performance alerts (admin only)
CREATE POLICY "Admin can view performance alerts" ON performance_alerts
    FOR SELECT USING (auth.jwt() ->> 'role' = 'admin');

CREATE POLICY "Service role can manage alerts" ON performance_alerts
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- Functions for analytics aggregation
CREATE OR REPLACE FUNCTION get_feature_usage_summary(
    p_feature_name VARCHAR(100),
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    total_events BIGINT,
    unique_users BIGINT,
    event_breakdown JSONB,
    daily_usage JSONB
) AS $$
BEGIN
    RETURN QUERY
    WITH event_stats AS (
        SELECT 
            COUNT(*) as total_events,
            COUNT(DISTINCT user_id) as unique_users,
            jsonb_object_agg(event_type, event_count) as event_breakdown
        FROM (
            SELECT 
                event_type,
                COUNT(*) as event_count
            FROM usage_analytics 
            WHERE feature_name = p_feature_name 
                AND timestamp >= NOW() - INTERVAL '1 day' * p_days
                AND user_id != 'system'
            GROUP BY event_type
        ) event_counts
    ),
    daily_stats AS (
        SELECT 
            jsonb_object_agg(
                date_trunc('day', timestamp)::date::text, 
                daily_count
            ) as daily_usage
        FROM (
            SELECT 
                date_trunc('day', timestamp) as day,
                COUNT(*) as daily_count
            FROM usage_analytics 
            WHERE feature_name = p_feature_name 
                AND timestamp >= NOW() - INTERVAL '1 day' * p_days
                AND user_id != 'system'
            GROUP BY date_trunc('day', timestamp)
            ORDER BY day
        ) daily_counts
    )
    SELECT 
        es.total_events,
        es.unique_users,
        es.event_breakdown,
        ds.daily_usage
    FROM event_stats es
    CROSS JOIN daily_stats ds;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function for A/B test analysis
CREATE OR REPLACE FUNCTION get_ab_test_analysis(
    p_feature_name VARCHAR(100),
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    variant VARCHAR(50),
    unique_users BIGINT,
    total_interactions BIGINT,
    total_conversions BIGINT,
    conversion_rate DECIMAL(5,4)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ua.variant,
        COUNT(DISTINCT ua.user_id) as unique_users,
        COUNT(CASE WHEN ua.event_type = 'feature_interaction' THEN 1 END) as total_interactions,
        COUNT(CASE WHEN ua.event_type = 'feature_conversion' THEN 1 END) as total_conversions,
        CASE 
            WHEN COUNT(CASE WHEN ua.event_type = 'feature_interaction' THEN 1 END) > 0 THEN
                COUNT(CASE WHEN ua.event_type = 'feature_conversion' THEN 1 END)::DECIMAL / 
                COUNT(CASE WHEN ua.event_type = 'feature_interaction' THEN 1 END)::DECIMAL
            ELSE 0
        END as conversion_rate
    FROM usage_analytics ua
    WHERE ua.feature_name = p_feature_name 
        AND ua.timestamp >= NOW() - INTERVAL '1 day' * p_days
        AND ua.variant IS NOT NULL
        AND ua.user_id != 'system'
    GROUP BY ua.variant
    ORDER BY conversion_rate DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function for performance monitoring
CREATE OR REPLACE FUNCTION get_performance_summary(
    p_feature_name VARCHAR(100),
    p_hours INTEGER DEFAULT 24
)
RETURNS TABLE (
    metric_name VARCHAR(50),
    avg_value DECIMAL(15,4),
    min_value DECIMAL(15,4),
    max_value DECIMAL(15,4),
    alert_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fp.metric_name,
        AVG(fp.metric_value) as avg_value,
        MIN(fp.metric_value) as min_value,
        MAX(fp.metric_value) as max_value,
        COUNT(pa.id) as alert_count
    FROM feature_performance fp
    LEFT JOIN performance_alerts pa ON pa.feature_name = fp.feature_name 
        AND pa.metric_type = fp.metric_name
        AND pa.timestamp >= NOW() - INTERVAL '1 hour' * p_hours
    WHERE fp.feature_name = p_feature_name 
        AND fp.timestamp >= NOW() - INTERVAL '1 hour' * p_hours
    GROUP BY fp.metric_name
    ORDER BY fp.metric_name;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger for automatic cleanup of old analytics data
CREATE OR REPLACE FUNCTION cleanup_old_analytics()
RETURNS TRIGGER AS $$
BEGIN
    -- Delete analytics data older than 90 days
    DELETE FROM usage_analytics 
    WHERE timestamp < NOW() - INTERVAL '90 days';
    
    -- Delete resolved alerts older than 30 days
    DELETE FROM performance_alerts 
    WHERE resolved = true 
        AND resolved_at < NOW() - INTERVAL '30 days';
    
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to run cleanup daily
CREATE OR REPLACE FUNCTION schedule_analytics_cleanup()
RETURNS void AS $$
BEGIN
    -- This would be called by a scheduled job
    PERFORM cleanup_old_analytics();
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;