-- =============================================================================
-- AI Backend Database Schema Extensions
-- =============================================================================
-- This file contains database schema extensions for AI analysis results
-- Run this after your main Supabase schema is set up

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- AI Analysis Results Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS ai_analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID NOT NULL REFERENCES businesses(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL CHECK (analysis_type IN (
        'fraud_detection',
        'predictive_analytics', 
        'compliance_check',
        'nlp_invoice',
        'pattern_analysis',
        'risk_assessment'
    )),
    entity_id UUID NOT NULL, -- ID of the entity being analyzed (transaction, invoice, etc.)
    entity_type VARCHAR(50) NOT NULL CHECK (entity_type IN (
        'transaction',
        'invoice', 
        'customer',
        'supplier',
        'business'
    )),
    results JSONB NOT NULL, -- Analysis results in JSON format
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    processing_time_ms INTEGER,
    model_version VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE -- For automatic cleanup
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_ai_analysis_business_id ON ai_analysis_results(business_id);
CREATE INDEX IF NOT EXISTS idx_ai_analysis_type ON ai_analysis_results(analysis_type);
CREATE INDEX IF NOT EXISTS idx_ai_analysis_entity ON ai_analysis_results(entity_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_ai_analysis_created_at ON ai_analysis_results(created_at);
CREATE INDEX IF NOT EXISTS idx_ai_analysis_expires_at ON ai_analysis_results(expires_at) WHERE expires_at IS NOT NULL;

-- =============================================================================
-- Fraud Alerts Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS fraud_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID NOT NULL REFERENCES businesses(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL CHECK (alert_type IN (
        'duplicate_invoice',
        'payment_mismatch',
        'suspicious_pattern',
        'duplicate_transaction',
        'unusual_amount',
        'frequency_anomaly'
    )),
    severity VARCHAR(20) NOT NULL DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    evidence JSONB, -- Supporting evidence for the alert
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    related_entity_id UUID, -- ID of related transaction/invoice
    related_entity_type VARCHAR(50),
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'acknowledged', 'resolved', 'dismissed')),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by UUID REFERENCES auth.users(id),
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by UUID REFERENCES auth.users(id),
    resolution_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for fraud alerts
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_business_id ON fraud_alerts(business_id);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_status ON fraud_alerts(status);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_severity ON fraud_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_type ON fraud_alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_created_at ON fraud_alerts(created_at);

-- =============================================================================
-- Business Insights Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS business_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID NOT NULL REFERENCES businesses(id) ON DELETE CASCADE,
    insight_type VARCHAR(50) NOT NULL CHECK (insight_type IN (
        'cash_flow_prediction',
        'customer_analysis',
        'revenue_trend',
        'expense_pattern',
        'working_capital',
        'seasonal_trend',
        'growth_opportunity',
        'risk_warning'
    )),
    category VARCHAR(30) NOT NULL DEFAULT 'general' CHECK (category IN (
        'financial',
        'operational', 
        'strategic',
        'compliance',
        'general'
    )),
    priority VARCHAR(20) NOT NULL DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    recommendations JSONB, -- Array of recommended actions
    metrics JSONB, -- Supporting metrics and data
    impact_score DECIMAL(3,2) CHECK (impact_score >= 0 AND impact_score <= 1),
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    valid_until TIMESTAMP WITH TIME ZONE, -- When this insight expires
    viewed_at TIMESTAMP WITH TIME ZONE,
    viewed_by UUID REFERENCES auth.users(id),
    dismissed_at TIMESTAMP WITH TIME ZONE,
    dismissed_by UUID REFERENCES auth.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for business insights
CREATE INDEX IF NOT EXISTS idx_business_insights_business_id ON business_insights(business_id);
CREATE INDEX IF NOT EXISTS idx_business_insights_type ON business_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_business_insights_category ON business_insights(category);
CREATE INDEX IF NOT EXISTS idx_business_insights_priority ON business_insights(priority);
CREATE INDEX IF NOT EXISTS idx_business_insights_valid_until ON business_insights(valid_until) WHERE valid_until IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_business_insights_created_at ON business_insights(created_at);

-- =============================================================================
-- ML Models Metadata Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL CHECK (model_type IN (
        'fraud_detection',
        'anomaly_detection',
        'time_series_forecast',
        'classification',
        'clustering',
        'nlp_extraction',
        'pattern_recognition',
        'prediction'
    )),
    business_id UUID REFERENCES businesses(id) ON DELETE CASCADE,
    description TEXT,
    training_data_hash VARCHAR(64), -- Hash of training data for versioning
    accuracy_metrics JSONB, -- Model performance metrics
    training_config JSONB, -- Training configuration used
    model_path VARCHAR(500), -- Path to model file
    status VARCHAR(20) NOT NULL DEFAULT 'training' CHECK (status IN (
        'training',
        'validating', 
        'deployed',
        'deprecated',
        'failed'
    )),
    trained_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deployed_at TIMESTAMP WITH TIME ZONE,
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(model_name, model_version)
);

-- Index for ML models
CREATE INDEX IF NOT EXISTS idx_ml_models_name_version ON ml_models(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_ml_models_type ON ml_models(model_type);
CREATE INDEX IF NOT EXISTS idx_ml_models_status ON ml_models(status);
CREATE INDEX IF NOT EXISTS idx_ml_models_business_id ON ml_models(business_id);
CREATE INDEX IF NOT EXISTS idx_ml_models_deployed ON ml_models(status) WHERE status = 'deployed';

-- =============================================================================
-- Model Performance Tracking Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS model_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20),
    accuracy DECIMAL(5,4) DEFAULT 0.0,
    precision_score DECIMAL(5,4) DEFAULT 0.0,
    recall DECIMAL(5,4) DEFAULT 0.0,
    f1_score DECIMAL(5,4) DEFAULT 0.0,
    cross_val_mean DECIMAL(5,4) DEFAULT 0.0,
    cross_val_std DECIMAL(5,4) DEFAULT 0.0,
    anomaly_ratio DECIMAL(5,4) DEFAULT 0.0,
    sample_size INTEGER DEFAULT 0,
    evaluation_period_start TIMESTAMP WITH TIME ZONE,
    evaluation_period_end TIMESTAMP WITH TIME ZONE,
    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for model performance
CREATE INDEX IF NOT EXISTS idx_model_performance_name ON model_performance(model_name);
CREATE INDEX IF NOT EXISTS idx_model_performance_evaluated_at ON model_performance(evaluated_at);
CREATE INDEX IF NOT EXISTS idx_model_performance_name_date ON model_performance(model_name, evaluated_at);

-- =============================================================================
-- Model Feedback Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS model_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    prediction_id VARCHAR(100) NOT NULL,
    predicted_outcome BOOLEAN,
    actual_outcome BOOLEAN,
    confidence_score DECIMAL(3,2),
    user_feedback TEXT,
    business_id UUID REFERENCES businesses(id) ON DELETE CASCADE,
    feedback_type VARCHAR(30) DEFAULT 'correction' CHECK (feedback_type IN (
        'correction',
        'confirmation',
        'improvement_suggestion',
        'false_positive',
        'false_negative'
    )),
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    recorded_by UUID REFERENCES auth.users(id),
    
    UNIQUE(model_name, prediction_id)
);

-- Indexes for model feedback
CREATE INDEX IF NOT EXISTS idx_model_feedback_name ON model_feedback(model_name);
CREATE INDEX IF NOT EXISTS idx_model_feedback_business_id ON model_feedback(business_id);
CREATE INDEX IF NOT EXISTS idx_model_feedback_recorded_at ON model_feedback(recorded_at);
CREATE INDEX IF NOT EXISTS idx_model_feedback_type ON model_feedback(feedback_type);

-- =============================================================================
-- Model Alerts Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS model_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    alert_type VARCHAR(50) NOT NULL CHECK (alert_type IN (
        'performance_degradation',
        'training_failed',
        'deployment_failed',
        'data_drift',
        'accuracy_drop',
        'high_error_rate'
    )),
    severity VARCHAR(20) NOT NULL DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    message TEXT NOT NULL,
    details JSONB,
    current_accuracy DECIMAL(5,4),
    historical_accuracy DECIMAL(5,4),
    degradation_amount DECIMAL(5,4),
    threshold_breached DECIMAL(5,4),
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'acknowledged', 'resolved')),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by UUID REFERENCES auth.users(id),
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by UUID REFERENCES auth.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for model alerts
CREATE INDEX IF NOT EXISTS idx_model_alerts_name ON model_alerts(model_name);
CREATE INDEX IF NOT EXISTS idx_model_alerts_type ON model_alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_model_alerts_severity ON model_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_model_alerts_status ON model_alerts(status);
CREATE INDEX IF NOT EXISTS idx_model_alerts_created_at ON model_alerts(created_at);

-- =============================================================================
-- Model Training Jobs Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS model_training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    business_id UUID REFERENCES businesses(id) ON DELETE CASCADE,
    training_config JSONB NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'queued' CHECK (status IN (
        'queued',
        'running',
        'completed',
        'failed',
        'cancelled'
    )),
    progress DECIMAL(5,2) DEFAULT 0.0 CHECK (progress >= 0 AND progress <= 100),
    training_data_size INTEGER,
    training_duration_seconds INTEGER,
    error_message TEXT,
    result_model_version VARCHAR(20),
    performance_metrics JSONB,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID REFERENCES auth.users(id)
);

-- Indexes for model training jobs
CREATE INDEX IF NOT EXISTS idx_model_training_jobs_name ON model_training_jobs(model_name);
CREATE INDEX IF NOT EXISTS idx_model_training_jobs_business_id ON model_training_jobs(business_id);
CREATE INDEX IF NOT EXISTS idx_model_training_jobs_status ON model_training_jobs(status);
CREATE INDEX IF NOT EXISTS idx_model_training_jobs_created_at ON model_training_jobs(created_at);

-- =============================================================================
-- AI Processing Logs Table (for debugging and monitoring)
-- =============================================================================
CREATE TABLE IF NOT EXISTS ai_processing_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID REFERENCES businesses(id) ON DELETE CASCADE,
    operation_type VARCHAR(50) NOT NULL,
    operation_id VARCHAR(100), -- Unique identifier for the operation
    input_data_hash VARCHAR(64), -- Hash of input data (for privacy)
    processing_time_ms INTEGER,
    memory_usage_mb INTEGER,
    status VARCHAR(20) NOT NULL CHECK (status IN ('started', 'completed', 'failed', 'timeout')),
    error_message TEXT,
    model_used VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for processing logs
CREATE INDEX IF NOT EXISTS idx_ai_processing_logs_business_id ON ai_processing_logs(business_id);
CREATE INDEX IF NOT EXISTS idx_ai_processing_logs_operation ON ai_processing_logs(operation_type);
CREATE INDEX IF NOT EXISTS idx_ai_processing_logs_status ON ai_processing_logs(status);
CREATE INDEX IF NOT EXISTS idx_ai_processing_logs_created_at ON ai_processing_logs(created_at);

-- =============================================================================
-- Row Level Security (RLS) Policies
-- =============================================================================

-- Enable RLS on all AI tables
ALTER TABLE ai_analysis_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE fraud_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE business_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_processing_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_training_jobs ENABLE ROW LEVEL SECURITY;

-- AI Analysis Results Policies
CREATE POLICY "Users can view their business AI analysis results" ON ai_analysis_results
    FOR SELECT USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Service role can manage AI analysis results" ON ai_analysis_results
    FOR ALL USING (auth.role() = 'service_role');

-- Fraud Alerts Policies
CREATE POLICY "Users can view their business fraud alerts" ON fraud_alerts
    FOR SELECT USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Users can update their business fraud alerts" ON fraud_alerts
    FOR UPDATE USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Service role can manage fraud alerts" ON fraud_alerts
    FOR ALL USING (auth.role() = 'service_role');

-- Business Insights Policies
CREATE POLICY "Users can view their business insights" ON business_insights
    FOR SELECT USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Users can update their business insights" ON business_insights
    FOR UPDATE USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Service role can manage business insights" ON business_insights
    FOR ALL USING (auth.role() = 'service_role');

-- AI Processing Logs Policies
CREATE POLICY "Users can view their business processing logs" ON ai_processing_logs
    FOR SELECT USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Service role can manage processing logs" ON ai_processing_logs
    FOR ALL USING (auth.role() = 'service_role');

-- ML Models Policies
CREATE POLICY "Users can view models for their business" ON ml_models
    FOR SELECT USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        ) OR business_id IS NULL
    );

CREATE POLICY "Service role can manage ML models" ON ml_models
    FOR ALL USING (auth.role() = 'service_role');

-- Model Performance Policies
CREATE POLICY "Service role can manage model performance" ON model_performance
    FOR ALL USING (auth.role() = 'service_role');

-- Model Feedback Policies
CREATE POLICY "Users can view feedback for their business models" ON model_feedback
    FOR SELECT USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Users can create feedback for their business models" ON model_feedback
    FOR INSERT WITH CHECK (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Service role can manage model feedback" ON model_feedback
    FOR ALL USING (auth.role() = 'service_role');

-- Model Alerts Policies
CREATE POLICY "Service role can manage model alerts" ON model_alerts
    FOR ALL USING (auth.role() = 'service_role');

-- Model Training Jobs Policies
CREATE POLICY "Users can view their business training jobs" ON model_training_jobs
    FOR SELECT USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Users can create training jobs for their business" ON model_training_jobs
    FOR INSERT WITH CHECK (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Service role can manage training jobs" ON model_training_jobs
    FOR ALL USING (auth.role() = 'service_role');

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- Function to clean up expired analysis results
CREATE OR REPLACE FUNCTION cleanup_expired_ai_data()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete expired analysis results
    DELETE FROM ai_analysis_results 
    WHERE expires_at IS NOT NULL AND expires_at < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Delete old processing logs (keep only last 30 days)
    DELETE FROM ai_processing_logs 
    WHERE created_at < NOW() - INTERVAL '30 days';
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at columns
CREATE TRIGGER update_ai_analysis_results_updated_at
    BEFORE UPDATE ON ai_analysis_results
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_fraud_alerts_updated_at
    BEFORE UPDATE ON fraud_alerts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_business_insights_updated_at
    BEFORE UPDATE ON business_insights
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ml_models_updated_at
    BEFORE UPDATE ON ml_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_alerts_updated_at
    BEFORE UPDATE ON model_alerts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Comments for documentation
-- =============================================================================
COMMENT ON TABLE ai_analysis_results IS 'Stores results from AI analysis operations';
COMMENT ON TABLE fraud_alerts IS 'Stores fraud detection alerts and their status';
COMMENT ON TABLE business_insights IS 'Stores AI-generated business insights and recommendations';
COMMENT ON TABLE ml_models IS 'Metadata about machine learning models used in the system';
COMMENT ON TABLE ai_processing_logs IS 'Logs of AI processing operations for monitoring and debugging';

COMMENT ON COLUMN ai_analysis_results.confidence_score IS 'Confidence score of the analysis (0.0 to 1.0)';
COMMENT ON COLUMN fraud_alerts.evidence IS 'JSON object containing supporting evidence for the alert';
COMMENT ON COLUMN business_insights.recommendations IS 'JSON array of recommended actions';
COMMENT ON COLUMN ml_models.accuracy_metrics IS 'JSON object containing model performance metrics';

-- =============================================================================
-- Compliance Issues Table (for issue tracking and resolution)
-- =============================================================================
CREATE TABLE IF NOT EXISTS compliance_issues (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID NOT NULL REFERENCES businesses(id) ON DELETE CASCADE,
    invoice_id UUID REFERENCES invoices(id) ON DELETE CASCADE,
    issue_type VARCHAR(50) NOT NULL CHECK (issue_type IN (
        'gst_validation',
        'tax_calculation',
        'missing_fields',
        'deadline_warning',
        'format_error',
        'place_of_supply'
    )),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    plain_language_explanation TEXT NOT NULL,
    suggested_fixes JSONB, -- Array of suggested fix actions
    field_name VARCHAR(100), -- Specific field that has the issue
    current_value TEXT, -- Current value of the problematic field
    expected_value TEXT, -- Expected/correct value
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'resolved', 'dismissed')),
    resolution_action TEXT, -- Action taken to resolve the issue
    resolution_notes TEXT, -- Additional notes about the resolution
    resolved_by UUID REFERENCES auth.users(id),
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for compliance issues
CREATE INDEX IF NOT EXISTS idx_compliance_issues_business_id ON compliance_issues(business_id);
CREATE INDEX IF NOT EXISTS idx_compliance_issues_invoice_id ON compliance_issues(invoice_id);
CREATE INDEX IF NOT EXISTS idx_compliance_issues_type ON compliance_issues(issue_type);
CREATE INDEX IF NOT EXISTS idx_compliance_issues_severity ON compliance_issues(severity);
CREATE INDEX IF NOT EXISTS idx_compliance_issues_status ON compliance_issues(status);
CREATE INDEX IF NOT EXISTS idx_compliance_issues_created_at ON compliance_issues(created_at);

-- =============================================================================
-- Compliance Reminders Table (for automated reminder system)
-- =============================================================================
CREATE TABLE IF NOT EXISTS compliance_reminders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID NOT NULL REFERENCES businesses(id) ON DELETE CASCADE,
    reminder_type VARCHAR(50) NOT NULL CHECK (reminder_type IN (
        'gst_filing',
        'tax_payment',
        'return_submission',
        'audit_preparation',
        'license_renewal',
        'custom'
    )),
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    due_date TIMESTAMP WITH TIME ZONE NOT NULL,
    priority VARCHAR(20) NOT NULL DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high')),
    recurring BOOLEAN DEFAULT FALSE,
    recurring_interval_days INTEGER, -- Days between recurring reminders
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'completed', 'cancelled')),
    completion_notes TEXT,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_by UUID REFERENCES auth.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for compliance reminders
CREATE INDEX IF NOT EXISTS idx_compliance_reminders_business_id ON compliance_reminders(business_id);
CREATE INDEX IF NOT EXISTS idx_compliance_reminders_type ON compliance_reminders(reminder_type);
CREATE INDEX IF NOT EXISTS idx_compliance_reminders_due_date ON compliance_reminders(due_date);
CREATE INDEX IF NOT EXISTS idx_compliance_reminders_status ON compliance_reminders(status);
CREATE INDEX IF NOT EXISTS idx_compliance_reminders_priority ON compliance_reminders(priority);

-- =============================================================================
-- Scheduled Notifications Table (for reminder notifications)
-- =============================================================================
CREATE TABLE IF NOT EXISTS scheduled_notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID NOT NULL REFERENCES businesses(id) ON DELETE CASCADE,
    reminder_id UUID REFERENCES compliance_reminders(id) ON DELETE CASCADE,
    notification_date TIMESTAMP WITH TIME ZONE NOT NULL,
    message TEXT NOT NULL,
    notification_type VARCHAR(30) DEFAULT 'reminder' CHECK (notification_type IN (
        'reminder',
        'warning',
        'urgent',
        'info'
    )),
    status VARCHAR(20) NOT NULL DEFAULT 'scheduled' CHECK (status IN (
        'scheduled',
        'sent',
        'failed',
        'cancelled'
    )),
    sent_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for scheduled notifications
CREATE INDEX IF NOT EXISTS idx_scheduled_notifications_business_id ON scheduled_notifications(business_id);
CREATE INDEX IF NOT EXISTS idx_scheduled_notifications_reminder_id ON scheduled_notifications(reminder_id);
CREATE INDEX IF NOT EXISTS idx_scheduled_notifications_date ON scheduled_notifications(notification_date);
CREATE INDEX IF NOT EXISTS idx_scheduled_notifications_status ON scheduled_notifications(status);

-- =============================================================================
-- Compliance Activity Log Table (for audit trail)
-- =============================================================================
CREATE TABLE IF NOT EXISTS compliance_activity_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID NOT NULL REFERENCES businesses(id) ON DELETE CASCADE,
    activity_type VARCHAR(50) NOT NULL CHECK (activity_type IN (
        'issue_created',
        'issue_resolved',
        'issue_dismissed',
        'reminder_created',
        'reminder_completed',
        'reminder_cancelled',
        'compliance_check_run',
        'bulk_check_started',
        'bulk_check_completed'
    )),
    entity_id UUID, -- ID of related entity (issue, reminder, etc.)
    entity_type VARCHAR(50), -- Type of related entity
    details JSONB, -- Additional activity details
    performed_by UUID REFERENCES auth.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for compliance activity log
CREATE INDEX IF NOT EXISTS idx_compliance_activity_log_business_id ON compliance_activity_log(business_id);
CREATE INDEX IF NOT EXISTS idx_compliance_activity_log_type ON compliance_activity_log(activity_type);
CREATE INDEX IF NOT EXISTS idx_compliance_activity_log_created_at ON compliance_activity_log(created_at);

-- =============================================================================
-- Bulk Compliance Jobs Table (for bulk processing)
-- =============================================================================
CREATE TABLE IF NOT EXISTS bulk_compliance_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID NOT NULL REFERENCES businesses(id) ON DELETE CASCADE,
    invoice_ids JSONB NOT NULL, -- Array of invoice IDs to process
    status VARCHAR(20) NOT NULL DEFAULT 'started' CHECK (status IN (
        'started',
        'processing',
        'completed',
        'failed',
        'cancelled'
    )),
    progress DECIMAL(5,2) DEFAULT 0.0 CHECK (progress >= 0 AND progress <= 100),
    results JSONB, -- Processing results
    error TEXT, -- Error message if failed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for bulk compliance jobs
CREATE INDEX IF NOT EXISTS idx_bulk_compliance_jobs_business_id ON bulk_compliance_jobs(business_id);
CREATE INDEX IF NOT EXISTS idx_bulk_compliance_jobs_status ON bulk_compliance_jobs(status);
CREATE INDEX IF NOT EXISTS idx_bulk_compliance_jobs_created_at ON bulk_compliance_jobs(created_at);

-- =============================================================================
-- Row Level Security (RLS) Policies for New Tables
-- =============================================================================

-- Enable RLS on compliance tables
ALTER TABLE compliance_issues ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance_reminders ENABLE ROW LEVEL SECURITY;
ALTER TABLE scheduled_notifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance_activity_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE bulk_compliance_jobs ENABLE ROW LEVEL SECURITY;

-- Compliance Issues Policies
CREATE POLICY "Users can view their business compliance issues" ON compliance_issues
    FOR SELECT USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Users can update their business compliance issues" ON compliance_issues
    FOR UPDATE USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Service role can manage compliance issues" ON compliance_issues
    FOR ALL USING (auth.role() = 'service_role');

-- Compliance Reminders Policies
CREATE POLICY "Users can view their business compliance reminders" ON compliance_reminders
    FOR SELECT USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Users can manage their business compliance reminders" ON compliance_reminders
    FOR ALL USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Service role can manage compliance reminders" ON compliance_reminders
    FOR ALL USING (auth.role() = 'service_role');

-- Scheduled Notifications Policies
CREATE POLICY "Users can view their business scheduled notifications" ON scheduled_notifications
    FOR SELECT USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Service role can manage scheduled notifications" ON scheduled_notifications
    FOR ALL USING (auth.role() = 'service_role');

-- Compliance Activity Log Policies
CREATE POLICY "Users can view their business compliance activity log" ON compliance_activity_log
    FOR SELECT USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Service role can manage compliance activity log" ON compliance_activity_log
    FOR ALL USING (auth.role() = 'service_role');

-- Bulk Compliance Jobs Policies
CREATE POLICY "Users can view their business bulk compliance jobs" ON bulk_compliance_jobs
    FOR SELECT USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Users can manage their business bulk compliance jobs" ON bulk_compliance_jobs
    FOR ALL USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Service role can manage bulk compliance jobs" ON bulk_compliance_jobs
    FOR ALL USING (auth.role() = 'service_role');

-- =============================================================================
-- Triggers for Updated At Columns
-- =============================================================================

CREATE TRIGGER update_compliance_issues_updated_at
    BEFORE UPDATE ON compliance_issues
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_compliance_reminders_updated_at
    BEFORE UPDATE ON compliance_reminders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Utility Functions for Compliance
-- =============================================================================

-- Function to automatically create compliance issues from analysis results
CREATE OR REPLACE FUNCTION create_compliance_issues_from_analysis(
    p_business_id UUID,
    p_invoice_id UUID,
    p_analysis_results JSONB
)
RETURNS INTEGER AS $$
DECLARE
    issue_count INTEGER := 0;
    issue JSONB;
BEGIN
    -- Loop through issues in analysis results
    FOR issue IN SELECT * FROM jsonb_array_elements(p_analysis_results->'issues')
    LOOP
        INSERT INTO compliance_issues (
            business_id,
            invoice_id,
            issue_type,
            severity,
            title,
            description,
            plain_language_explanation,
            suggested_fixes,
            field_name,
            current_value,
            expected_value
        ) VALUES (
            p_business_id,
            p_invoice_id,
            issue->>'type',
            issue->>'severity',
            COALESCE(issue->>'title', issue->>'description'),
            issue->>'description',
            issue->>'plain_language_explanation',
            issue->'suggested_fixes',
            issue->>'field_name',
            issue->>'current_value',
            issue->>'expected_value'
        );
        
        issue_count := issue_count + 1;
    END LOOP;
    
    RETURN issue_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get compliance summary for a business
CREATE OR REPLACE FUNCTION get_compliance_summary(p_business_id UUID)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_issues', COUNT(*),
        'active_issues', COUNT(*) FILTER (WHERE status = 'active'),
        'resolved_issues', COUNT(*) FILTER (WHERE status = 'resolved'),
        'critical_issues', COUNT(*) FILTER (WHERE severity = 'critical' AND status = 'active'),
        'high_issues', COUNT(*) FILTER (WHERE severity = 'high' AND status = 'active'),
        'medium_issues', COUNT(*) FILTER (WHERE severity = 'medium' AND status = 'active'),
        'low_issues', COUNT(*) FILTER (WHERE severity = 'low' AND status = 'active'),
        'issues_by_type', jsonb_object_agg(
            issue_type, 
            COUNT(*) FILTER (WHERE status = 'active')
        ),
        'last_updated', MAX(updated_at)
    ) INTO result
    FROM compliance_issues
    WHERE business_id = p_business_id;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Comments for New Tables
-- =============================================================================
COMMENT ON TABLE compliance_issues IS 'Tracks compliance issues found during invoice analysis';
COMMENT ON TABLE compliance_reminders IS 'Automated compliance reminders for businesses';
COMMENT ON TABLE scheduled_notifications IS 'Scheduled notifications for compliance reminders';
COMMENT ON TABLE compliance_activity_log IS 'Audit trail for compliance-related activities';
COMMENT ON TABLE bulk_compliance_jobs IS 'Background jobs for bulk compliance processing';

COMMENT ON COLUMN compliance_issues.plain_language_explanation IS 'User-friendly explanation of the compliance issue';
COMMENT ON COLUMN compliance_issues.suggested_fixes IS 'JSON array of suggested actions to fix the issue';
COMMENT ON COLUMN compliance_reminders.recurring_interval_days IS 'Days between recurring reminders (null for non-recurring)';
COMMENT ON COLUMN bulk_compliance_jobs.progress IS 'Processing progress as percentage (0-100)';

-- =============================================================================
-- Smart Notifications System Tables
-- =============================================================================

-- Smart notifications table
CREATE TABLE IF NOT EXISTS smart_notifications (
    id VARCHAR(100) PRIMARY KEY,
    user_id UUID NOT NULL,
    business_id UUID NOT NULL,
    type VARCHAR(50) NOT NULL CHECK (type IN (
        'fraud_alert',
        'compliance_warning',
        'cash_flow_prediction',
        'business_insight',
        'system_update'
    )),
    priority VARCHAR(20) NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    data JSONB DEFAULT '{}',
    channels TEXT[] NOT NULL,
    scheduled_for TIMESTAMP WITH TIME ZONE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE,
    consolidation_key VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    sent_at TIMESTAMP WITH TIME ZONE,
    acknowledged_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for smart notifications
CREATE INDEX IF NOT EXISTS idx_smart_notifications_user_id ON smart_notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_smart_notifications_business_id ON smart_notifications(business_id);
CREATE INDEX IF NOT EXISTS idx_smart_notifications_scheduled_for ON smart_notifications(scheduled_for);
CREATE INDEX IF NOT EXISTS idx_smart_notifications_consolidation_key ON smart_notifications(consolidation_key);
CREATE INDEX IF NOT EXISTS idx_smart_notifications_type ON smart_notifications(type);
CREATE INDEX IF NOT EXISTS idx_smart_notifications_priority ON smart_notifications(priority);

-- In-app notifications table
CREATE TABLE IF NOT EXISTS in_app_notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    notification_id VARCHAR(100) NOT NULL,
    user_id UUID NOT NULL,
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    data JSONB DEFAULT '{}',
    priority VARCHAR(20) NOT NULL,
    read BOOLEAN DEFAULT false,
    read_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (notification_id) REFERENCES smart_notifications(id) ON DELETE CASCADE
);

-- Indexes for in-app notifications
CREATE INDEX IF NOT EXISTS idx_in_app_notifications_user_id ON in_app_notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_in_app_notifications_read ON in_app_notifications(read);
CREATE INDEX IF NOT EXISTS idx_in_app_notifications_notification_id ON in_app_notifications(notification_id);
CREATE INDEX IF NOT EXISTS idx_in_app_notifications_created_at ON in_app_notifications(created_at);

-- Notification preferences table
CREATE TABLE IF NOT EXISTS notification_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    channel VARCHAR(20) NOT NULL CHECK (channel IN ('push', 'email', 'in_app', 'sms')),
    notification_type VARCHAR(50) NOT NULL CHECK (notification_type IN (
        'fraud_alert',
        'compliance_warning',
        'cash_flow_prediction',
        'business_insight',
        'system_update'
    )),
    enabled BOOLEAN DEFAULT true,
    quiet_hours_start TIME,
    quiet_hours_end TIME,
    frequency_limit INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, channel, notification_type)
);

-- Indexes for notification preferences
CREATE INDEX IF NOT EXISTS idx_notification_preferences_user_id ON notification_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_notification_preferences_channel ON notification_preferences(channel);
CREATE INDEX IF NOT EXISTS idx_notification_preferences_type ON notification_preferences(notification_type);

-- Notification metrics table for learning and optimization
CREATE TABLE IF NOT EXISTS notification_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    notification_id VARCHAR(100) NOT NULL,
    user_id UUID NOT NULL,
    type VARCHAR(50) NOT NULL,
    priority VARCHAR(20) NOT NULL,
    channels_used TEXT[] NOT NULL,
    scheduled_for TIMESTAMP WITH TIME ZONE NOT NULL,
    delivered_at TIMESTAMP WITH TIME ZONE NOT NULL,
    delivery_delay_seconds INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (notification_id) REFERENCES smart_notifications(id) ON DELETE CASCADE
);

-- Indexes for notification metrics
CREATE INDEX IF NOT EXISTS idx_notification_metrics_user_id ON notification_metrics(user_id);
CREATE INDEX IF NOT EXISTS idx_notification_metrics_type ON notification_metrics(type);
CREATE INDEX IF NOT EXISTS idx_notification_metrics_notification_id ON notification_metrics(notification_id);
CREATE INDEX IF NOT EXISTS idx_notification_metrics_delivered_at ON notification_metrics(delivered_at);

-- User interactions table for preference learning
CREATE TABLE IF NOT EXISTS user_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    notification_id VARCHAR(100),
    notification_type VARCHAR(50),
    priority VARCHAR(20),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    response_time_seconds INTEGER,
    interaction_type VARCHAR(50) DEFAULT 'acknowledgment' CHECK (interaction_type IN (
        'acknowledgment',
        'dismissal',
        'action_taken',
        'feedback_provided'
    )),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for user interactions
CREATE INDEX IF NOT EXISTS idx_user_interactions_user_id ON user_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_interactions_notification_type ON user_interactions(notification_type);
CREATE INDEX IF NOT EXISTS idx_user_interactions_acknowledged_at ON user_interactions(acknowledged_at);
CREATE INDEX IF NOT EXISTS idx_user_interactions_notification_id ON user_interactions(notification_id);

-- =============================================================================
-- Smart Notifications RLS Policies
-- =============================================================================

-- Enable RLS on smart notification tables
ALTER TABLE smart_notifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE in_app_notifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE notification_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE notification_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_interactions ENABLE ROW LEVEL SECURITY;

-- Smart Notifications Policies
CREATE POLICY "Users can view their own notifications" ON smart_notifications
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Service role can manage all notifications" ON smart_notifications
    FOR ALL USING (auth.role() = 'service_role');

-- In-app Notifications Policies
CREATE POLICY "Users can view their own in-app notifications" ON in_app_notifications
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can update their own in-app notifications" ON in_app_notifications
    FOR UPDATE USING (user_id = auth.uid());

CREATE POLICY "Service role can manage all in-app notifications" ON in_app_notifications
    FOR ALL USING (auth.role() = 'service_role');

-- Notification Preferences Policies
CREATE POLICY "Users can manage their own notification preferences" ON notification_preferences
    FOR ALL USING (user_id = auth.uid());

CREATE POLICY "Service role can manage all notification preferences" ON notification_preferences
    FOR ALL USING (auth.role() = 'service_role');

-- Notification Metrics Policies
CREATE POLICY "Users can view their own notification metrics" ON notification_metrics
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Service role can manage all notification metrics" ON notification_metrics
    FOR ALL USING (auth.role() = 'service_role');

-- User Interactions Policies
CREATE POLICY "Users can manage their own interactions" ON user_interactions
    FOR ALL USING (user_id = auth.uid());

CREATE POLICY "Service role can manage all user interactions" ON user_interactions
    FOR ALL USING (auth.role() = 'service_role');

-- =============================================================================
-- Smart Notifications Triggers
-- =============================================================================

-- Create trigger for notification preferences updated_at
CREATE TRIGGER update_notification_preferences_updated_at
    BEFORE UPDATE ON notification_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Smart Notifications Comments
-- =============================================================================
COMMENT ON TABLE smart_notifications IS 'Stores smart notifications with prioritization and scheduling';
COMMENT ON TABLE in_app_notifications IS 'Stores in-app notifications for user interface display';
COMMENT ON TABLE notification_preferences IS 'Stores user preferences for notification channels and types';
COMMENT ON TABLE notification_metrics IS 'Stores metrics for notification delivery and performance analysis';
COMMENT ON TABLE user_interactions IS 'Stores user interaction data for preference learning and optimization';

COMMENT ON COLUMN smart_notifications.consolidation_key IS 'Key used to group similar notifications for consolidation';
COMMENT ON COLUMN notification_preferences.quiet_hours_start IS 'Start time for quiet hours (no notifications)';
COMMENT ON COLUMN notification_preferences.quiet_hours_end IS 'End time for quiet hours (no notifications)';
COMMENT ON COLUMN notification_preferences.frequency_limit IS 'Maximum notifications per day for this type';
COMMENT ON COLUMN notification_metrics.delivery_delay_seconds IS 'Delay between scheduled time and actual delivery';
COMMENT ON COLUMN user_interactions.response_time_seconds IS 'Time taken by user to acknowledge notification';
-
- =============================================================================
-- SECURITY AND PRIVACY EXTENSIONS
-- =============================================================================

-- Anonymized patterns table for privacy-preserving analytics
CREATE TABLE IF NOT EXISTS anonymized_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID NOT NULL,
    pattern_type VARCHAR(50) NOT NULL CHECK (pattern_type IN (
        'transaction_frequency',
        'amount_distribution',
        'temporal_patterns',
        'category_distribution',
        'customer_behavior',
        'supplier_patterns'
    )),
    pattern_data JSONB NOT NULL,
    anonymization_method VARCHAR(50) NOT NULL DEFAULT 'hash_based',
    original_data_hash VARCHAR(64),
    retention_category VARCHAR(50) DEFAULT 'anonymized_patterns',
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '1 year'),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ML training data table with privacy controls
CREATE TABLE IF NOT EXISTS ml_training_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    training_data JSONB NOT NULL,
    anonymized BOOLEAN DEFAULT true,
    data_hash VARCHAR(64),
    retention_category VARCHAR(50) DEFAULT 'ml_training_data',
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '1 year'),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Temporary AI processing data with automatic cleanup
CREATE TABLE IF NOT EXISTS temp_ai_processing (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID NOT NULL,
    processing_type VARCHAR(50) NOT NULL,
    temp_data JSONB NOT NULL,
    session_id VARCHAR(100),
    retention_category VARCHAR(50) DEFAULT 'temporary_processing',
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '1 day'),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Data retention reviews table for manual review process
CREATE TABLE IF NOT EXISTS data_retention_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID NOT NULL,
    data_category VARCHAR(50) NOT NULL CHECK (data_category IN (
        'raw_financial_data',
        'processed_ai_results',
        'anonymized_patterns',
        'audit_logs',
        'ml_training_data',
        'fraud_alerts',
        'compliance_reports',
        'temporary_processing'
    )),
    data_id UUID NOT NULL,
    expiry_date TIMESTAMP WITH TIME ZONE NOT NULL,
    review_status VARCHAR(20) DEFAULT 'pending' CHECK (review_status IN (
        'pending',
        'approved_for_deletion',
        'extended_retention',
        'permanent_retention'
    )),
    reviewed_by UUID REFERENCES auth.users(id),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    action_taken VARCHAR(50),
    review_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- AI audit log table for comprehensive audit trail
CREATE TABLE IF NOT EXISTS ai_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id VARCHAR(100) UNIQUE NOT NULL,
    event_type VARCHAR(50) NOT NULL CHECK (event_type IN (
        'ai_processing_start',
        'ai_processing_end',
        'data_encryption',
        'data_decryption',
        'data_anonymization',
        'fraud_detection',
        'predictive_analysis',
        'compliance_check',
        'nlp_processing',
        'model_training',
        'model_inference',
        'data_access',
        'data_export',
        'security_violation',
        'privacy_breach',
        'error_occurred'
    )),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    business_id UUID,
    user_id UUID REFERENCES auth.users(id),
    session_id VARCHAR(100),
    ip_address INET,
    user_agent TEXT,
    operation_details JSONB,
    environment VARCHAR(20),
    retention_category VARCHAR(50) DEFAULT 'audit_logs',
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '7 years'),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Security events table for threat monitoring
CREATE TABLE IF NOT EXISTS security_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL CHECK (event_type IN (
        'rate_limit_exceeded',
        'suspicious_activity',
        'repeated_failed_attempts',
        'unauthorized_access_attempt',
        'data_breach_attempt',
        'malicious_request',
        'anomalous_behavior'
    )),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    business_id UUID,
    client_ip INET,
    user_agent TEXT,
    security_details JSONB,
    blocked BOOLEAN DEFAULT false,
    resolved BOOLEAN DEFAULT false,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by UUID REFERENCES auth.users(id),
    resolution_notes TEXT,
    retention_category VARCHAR(50) DEFAULT 'audit_logs',
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '7 years'),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Data encryption metadata table
CREATE TABLE IF NOT EXISTS encryption_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_id UUID NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    record_id UUID NOT NULL,
    encrypted_fields JSONB NOT NULL,
    encryption_method VARCHAR(50) NOT NULL DEFAULT 'fernet_symmetric',
    key_version VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(table_name, record_id)
);

-- =============================================================================
-- INDEXES FOR SECURITY AND PRIVACY TABLES
-- =============================================================================

-- Anonymized patterns indexes
CREATE INDEX IF NOT EXISTS idx_anonymized_patterns_business_id ON anonymized_patterns(business_id);
CREATE INDEX IF NOT EXISTS idx_anonymized_patterns_type ON anonymized_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_anonymized_patterns_expires_at ON anonymized_patterns(expires_at);

-- ML training data indexes
CREATE INDEX IF NOT EXISTS idx_ml_training_data_business_id ON ml_training_data(business_id);
CREATE INDEX IF NOT EXISTS idx_ml_training_data_model ON ml_training_data(model_name);
CREATE INDEX IF NOT EXISTS idx_ml_training_data_expires_at ON ml_training_data(expires_at);

-- Temporary processing data indexes
CREATE INDEX IF NOT EXISTS idx_temp_ai_processing_business_id ON temp_ai_processing(business_id);
CREATE INDEX IF NOT EXISTS idx_temp_ai_processing_session ON temp_ai_processing(session_id);
CREATE INDEX IF NOT EXISTS idx_temp_ai_processing_expires_at ON temp_ai_processing(expires_at);

-- Data retention reviews indexes
CREATE INDEX IF NOT EXISTS idx_data_retention_reviews_business_id ON data_retention_reviews(business_id);
CREATE INDEX IF NOT EXISTS idx_data_retention_reviews_category ON data_retention_reviews(data_category);
CREATE INDEX IF NOT EXISTS idx_data_retention_reviews_status ON data_retention_reviews(review_status);
CREATE INDEX IF NOT EXISTS idx_data_retention_reviews_expiry ON data_retention_reviews(expiry_date);

-- AI audit log indexes
CREATE INDEX IF NOT EXISTS idx_ai_audit_log_business_id ON ai_audit_log(business_id);
CREATE INDEX IF NOT EXISTS idx_ai_audit_log_event_type ON ai_audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_ai_audit_log_severity ON ai_audit_log(severity);
CREATE INDEX IF NOT EXISTS idx_ai_audit_log_created_at ON ai_audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_ai_audit_log_expires_at ON ai_audit_log(expires_at);

-- Security events indexes
CREATE INDEX IF NOT EXISTS idx_security_events_type ON security_events(event_type);
CREATE INDEX IF NOT EXISTS idx_security_events_severity ON security_events(severity);
CREATE INDEX IF NOT EXISTS idx_security_events_client_ip ON security_events(client_ip);
CREATE INDEX IF NOT EXISTS idx_security_events_created_at ON security_events(created_at);
CREATE INDEX IF NOT EXISTS idx_security_events_resolved ON security_events(resolved);

-- Encryption metadata indexes
CREATE INDEX IF NOT EXISTS idx_encryption_metadata_business_id ON encryption_metadata(business_id);
CREATE INDEX IF NOT EXISTS idx_encryption_metadata_table_record ON encryption_metadata(table_name, record_id);

-- =============================================================================
-- ROW LEVEL SECURITY FOR SECURITY TABLES
-- =============================================================================

-- Enable RLS on security and privacy tables
ALTER TABLE anonymized_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_training_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE temp_ai_processing ENABLE ROW LEVEL SECURITY;
ALTER TABLE data_retention_reviews ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE encryption_metadata ENABLE ROW LEVEL SECURITY;

-- Anonymized patterns policies
CREATE POLICY "Service role can manage anonymized patterns" ON anonymized_patterns
    FOR ALL USING (auth.role() = 'service_role');

-- ML training data policies
CREATE POLICY "Service role can manage ML training data" ON ml_training_data
    FOR ALL USING (auth.role() = 'service_role');

-- Temporary processing data policies
CREATE POLICY "Service role can manage temp processing data" ON temp_ai_processing
    FOR ALL USING (auth.role() = 'service_role');

-- Data retention reviews policies
CREATE POLICY "Users can view retention reviews for their business" ON data_retention_reviews
    FOR SELECT USING (
        business_id IN (
            SELECT business_id FROM user_profiles WHERE id = auth.uid()
        )
    );

CREATE POLICY "Service role can manage retention reviews" ON data_retention_reviews
    FOR ALL USING (auth.role() = 'service_role');

-- AI audit log policies (restricted access)
CREATE POLICY "Service role can manage audit logs" ON ai_audit_log
    FOR ALL USING (auth.role() = 'service_role');

-- Security events policies (admin access only)
CREATE POLICY "Service role can manage security events" ON security_events
    FOR ALL USING (auth.role() = 'service_role');

-- Encryption metadata policies
CREATE POLICY "Service role can manage encryption metadata" ON encryption_metadata
    FOR ALL USING (auth.role() = 'service_role');

-- =============================================================================
-- SECURITY AND PRIVACY UTILITY FUNCTIONS
-- =============================================================================

-- Function to clean up expired data based on retention policies
CREATE OR REPLACE FUNCTION cleanup_expired_security_data()
RETURNS TABLE(
    table_name TEXT,
    deleted_count INTEGER
) AS $$
DECLARE
    temp_count INTEGER;
BEGIN
    -- Clean up expired anonymized patterns
    DELETE FROM anonymized_patterns WHERE expires_at < NOW();
    GET DIAGNOSTICS temp_count = ROW_COUNT;
    table_name := 'anonymized_patterns';
    deleted_count := temp_count;
    RETURN NEXT;
    
    -- Clean up expired ML training data
    DELETE FROM ml_training_data WHERE expires_at < NOW();
    GET DIAGNOSTICS temp_count = ROW_COUNT;
    table_name := 'ml_training_data';
    deleted_count := temp_count;
    RETURN NEXT;
    
    -- Clean up expired temporary processing data
    DELETE FROM temp_ai_processing WHERE expires_at < NOW();
    GET DIAGNOSTICS temp_count = ROW_COUNT;
    table_name := 'temp_ai_processing';
    deleted_count := temp_count;
    RETURN NEXT;
    
    -- Clean up old audit logs (keep only what's required by retention policy)
    DELETE FROM ai_audit_log WHERE expires_at < NOW();
    GET DIAGNOSTICS temp_count = ROW_COUNT;
    table_name := 'ai_audit_log';
    deleted_count := temp_count;
    RETURN NEXT;
    
    -- Clean up old security events (keep only what's required by retention policy)
    DELETE FROM security_events WHERE expires_at < NOW();
    GET DIAGNOSTICS temp_count = ROW_COUNT;
    table_name := 'security_events';
    deleted_count := temp_count;
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- Function to anonymize data for a specific business
CREATE OR REPLACE FUNCTION anonymize_business_data(target_business_id UUID)
RETURNS INTEGER AS $$
DECLARE
    anonymized_count INTEGER := 0;
BEGIN
    -- This function would implement data anonymization logic
    -- For now, it's a placeholder that returns 0
    RETURN anonymized_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get retention policy information
CREATE OR REPLACE FUNCTION get_retention_policy(data_category TEXT)
RETURNS TABLE(
    category TEXT,
    retention_days INTEGER,
    auto_purge BOOLEAN,
    requires_user_consent BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        data_category as category,
        CASE data_category
            WHEN 'raw_financial_data' THEN 2555  -- 7 years
            WHEN 'processed_ai_results' THEN 90
            WHEN 'anonymized_patterns' THEN 365
            WHEN 'audit_logs' THEN 2555  -- 7 years
            WHEN 'ml_training_data' THEN 365
            WHEN 'fraud_alerts' THEN 2555  -- 7 years
            WHEN 'compliance_reports' THEN 2555  -- 7 years
            WHEN 'temporary_processing' THEN 1
            ELSE 90
        END as retention_days,
        CASE data_category
            WHEN 'raw_financial_data' THEN false
            WHEN 'audit_logs' THEN false
            WHEN 'fraud_alerts' THEN false
            WHEN 'compliance_reports' THEN false
            ELSE true
        END as auto_purge,
        CASE data_category
            WHEN 'raw_financial_data' THEN true
            WHEN 'ml_training_data' THEN true
            WHEN 'audit_logs' THEN false
            WHEN 'fraud_alerts' THEN false
            WHEN 'compliance_reports' THEN false
            ELSE false
        END as requires_user_consent;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- TRIGGERS FOR AUTOMATIC EXPIRY DATE SETTING
-- =============================================================================

-- Function to set expiry dates based on retention policies
CREATE OR REPLACE FUNCTION set_expiry_date()
RETURNS TRIGGER AS $$
DECLARE
    retention_days INTEGER;
BEGIN
    -- Get retention days for the category
    SELECT 
        CASE NEW.retention_category
            WHEN 'raw_financial_data' THEN 2555  -- 7 years
            WHEN 'processed_ai_results' THEN 90
            WHEN 'anonymized_patterns' THEN 365
            WHEN 'audit_logs' THEN 2555  -- 7 years
            WHEN 'ml_training_data' THEN 365
            WHEN 'fraud_alerts' THEN 2555  -- 7 years
            WHEN 'compliance_reports' THEN 2555  -- 7 years
            WHEN 'temporary_processing' THEN 1
            ELSE 90
        END INTO retention_days;
    
    -- Set expiry date if not already set
    IF NEW.expires_at IS NULL THEN
        NEW.expires_at := NOW() + (retention_days || ' days')::INTERVAL;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic expiry date setting
CREATE TRIGGER set_ai_analysis_results_expiry
    BEFORE INSERT ON ai_analysis_results
    FOR EACH ROW EXECUTE FUNCTION set_expiry_date();

CREATE TRIGGER set_fraud_alerts_expiry
    BEFORE INSERT ON fraud_alerts
    FOR EACH ROW EXECUTE FUNCTION set_expiry_date();

CREATE TRIGGER set_business_insights_expiry
    BEFORE INSERT ON business_insights
    FOR EACH ROW EXECUTE FUNCTION set_expiry_date();

-- =============================================================================
-- COMMENTS FOR SECURITY TABLES
-- =============================================================================
COMMENT ON TABLE anonymized_patterns IS 'Stores anonymized data patterns for privacy-preserving analytics';
COMMENT ON TABLE ml_training_data IS 'Stores anonymized training data for machine learning models';
COMMENT ON TABLE temp_ai_processing IS 'Temporary storage for AI processing data with automatic cleanup';
COMMENT ON TABLE data_retention_reviews IS 'Tracks data that requires manual review before deletion';
COMMENT ON TABLE ai_audit_log IS 'Comprehensive audit trail for all AI operations';
COMMENT ON TABLE security_events IS 'Security events and threat monitoring';
COMMENT ON TABLE encryption_metadata IS 'Metadata about encrypted fields in the database';

COMMENT ON COLUMN anonymized_patterns.anonymization_method IS 'Method used for data anonymization';
COMMENT ON COLUMN ai_audit_log.audit_id IS 'Unique identifier for audit correlation';
COMMENT ON COLUMN security_events.security_details IS 'JSON object containing security event details';
COMMENT ON COLUMN encryption_metadata.encrypted_fields IS 'JSON array of field names that are encrypted';


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