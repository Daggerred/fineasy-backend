"""
Configuration settings for AI Backend
"""
from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""
    
    # Application Configuration
    ENVIRONMENT: str = "development"
    API_VERSION: str = "v1"
    LOG_LEVEL: str = "INFO"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    DEBUG: bool = False
    
    # Supabase Configuration (Primary Database)
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_KEY: str = ""
    SUPABASE_ANON_KEY: str = ""
    
    # Redis Configuration (Optional - for caching)
    REDIS_URL: str = "redis://localhost:6379"
    REDISHOST: str = "localhost"
    REDISPORT: int = 6379
    REDIS_TTL_HOURS: int = 24
    ENABLE_REDIS: bool = False  # Optional caching
    
    # AI Service Configuration (Google Gemini)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-pro"
    AI_TIMEOUT: int = 30
    MAX_TOKENS: int = 2000
    
    # WhatsApp Configuration
    WHATSAPP_PROVIDER: str = "webhook"  # Options: twilio, whatsapp_business, webhook
    WHATSAPP_API_TOKEN: Optional[str] = None
    WHATSAPP_PHONE_NUMBER_ID: Optional[str] = None
    WHATSAPP_WEBHOOK_URL: Optional[str] = None
    
    # Twilio Configuration (if using Twilio)
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_WHATSAPP_NUMBER: Optional[str] = None
    
    # Security Configuration
    SECRET_KEY: str = "your-super-secret-key-here-min-32-chars"
    JWT_SECRET: str = "your-jwt-secret-key-here"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440
    
    # CORS Configuration
    ALLOWED_ORIGINS: str = '["http://localhost:3000", "https://yourdomain.com"]'
    ALLOWED_HOSTS: str = '["localhost", "127.0.0.1", "yourdomain.com"]'
    
    # GST API Configuration
    GST_API_URL: str = "https://api.gst.gov.in"
    GST_API_KEY: Optional[str] = None
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 10485760  # 10MB
    UPLOAD_DIR: str = "./uploads"
    ALLOWED_EXTENSIONS: str = '["pdf", "txt", "jpg", "png"]'
    
    # AI Feature Configuration
    FRAUD_DETECTION_ENABLED: bool = True
    PREDICTIVE_ANALYTICS_ENABLED: bool = True
    COMPLIANCE_CHECKING_ENABLED: bool = True
    NLP_INVOICE_ENABLED: bool = True
    ENABLE_AI_FEATURES: bool = True
    ENABLE_CV_PARSING: bool = True
    
    # ML Model Configuration
    FRAUD_MODEL_THRESHOLD: float = 0.85
    PREDICTION_CONFIDENCE_MIN: float = 0.7
    MODEL_CACHE_SIZE: int = 100
    
    # Processing Configuration
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT_SECONDS: int = 30
    BATCH_SIZE: int = 100
    
    # Monitoring & Logging
    SENTRY_DSN: Optional[str] = None
    ENABLE_METRICS: bool = True
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_BURST: int = 200
    
    # Development Settings
    DEBUG_MODE: bool = False
    ENABLE_CORS: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields in .env
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse ALLOWED_ORIGINS from JSON string to list"""
        try:
            import json
            if isinstance(self.ALLOWED_ORIGINS, str):
                return json.loads(self.ALLOWED_ORIGINS)
            return self.ALLOWED_ORIGINS if isinstance(self.ALLOWED_ORIGINS, list) else ["*"]
        except:
            return ["*"]
    
    @property
    def allowed_hosts_list(self) -> List[str]:
        """Parse ALLOWED_HOSTS from JSON string to list"""
        try:
            import json
            if isinstance(self.ALLOWED_HOSTS, str):
                return json.loads(self.ALLOWED_HOSTS)
            return self.ALLOWED_HOSTS if isinstance(self.ALLOWED_HOSTS, list) else ["localhost"]
        except:
            return ["localhost"]
    
    @property
    def allowed_extensions_list(self) -> List[str]:
        """Parse ALLOWED_EXTENSIONS from JSON string to list"""
        try:
            import json
            if isinstance(self.ALLOWED_EXTENSIONS, str):
                return json.loads(self.ALLOWED_EXTENSIONS)
            return self.ALLOWED_EXTENSIONS if isinstance(self.ALLOWED_EXTENSIONS, list) else ["pdf", "txt", "jpg", "png"]
        except:
            return ["pdf", "txt", "jpg", "png"]
    
    @property
    def database_url(self) -> str:
        """Get Supabase database URL"""
        if self.SUPABASE_URL:
            # Extract database URL from Supabase URL
            return self.SUPABASE_URL.replace('https://', 'postgresql://postgres:').replace('.supabase.co', '.supabase.co:5432/postgres')
        return ""


# Global settings instance
settings = Settings()


def validate_configuration():
    """Validate required configuration settings"""
    critical_settings = []
    
    # Check if we're in testing mode (mock URLs are allowed)
    is_testing = (
        settings.SUPABASE_URL in ['mock_url', 'test_url'] or
        settings.ENVIRONMENT in ['test', 'testing'] or
        'mock' in settings.SUPABASE_URL.lower() or
        settings.DEBUG is True
    )
    
    if not is_testing:
        # Supabase is required for database in production
        if not settings.SUPABASE_URL:
            critical_settings.append("SUPABASE_URL")
        if not settings.SUPABASE_SERVICE_KEY:
            critical_settings.append("SUPABASE_SERVICE_KEY")
        
        # Gemini API key is required for AI features in production
        if settings.ENABLE_AI_FEATURES and not settings.GEMINI_API_KEY:
            critical_settings.append("GEMINI_API_KEY")
        
        if critical_settings:
            raise ValueError(f"Missing required environment variables: {', '.join(critical_settings)}")
        
        # Validate URLs in production
        if settings.SUPABASE_URL and not settings.SUPABASE_URL.startswith(('http://', 'https://')):
            raise ValueError("SUPABASE_URL must be a valid URL")
    else:
        # Testing mode - just log the configuration
        print(f"Running in testing mode with SUPABASE_URL: {settings.SUPABASE_URL}")
        print("Relaxed validation enabled for testing/development")
    
    # Common validation for all environments
    if settings.ENABLE_REDIS and settings.REDIS_URL and not settings.REDIS_URL.startswith('redis://'):
        if not is_testing:
            raise ValueError("REDIS_URL must be a valid Redis URL")
        else:
            print(f"Warning: Invalid Redis URL in testing mode: {settings.REDIS_URL}")
    # Create upload directory if it doesn't exist
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    return True


# Global settings instance
settings = Settings()

# Validate configuration on import (but don't fail if optional settings are missing)
try:
    validate_configuration()
except ValueError as e:
    print(f"Configuration warning: {e}")
    print("Some features may not work properly. Please check your .env file.")