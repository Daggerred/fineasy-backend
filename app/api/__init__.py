# API endpoints package

# Import all API modules with error handling
try:
    from . import health
except ImportError:
    health = None

try:
    from . import fraud
except ImportError:
    fraud = None

try:
    from . import insights
except ImportError:
    insights = None

try:
    from . import compliance
except ImportError:
    compliance = None

try:
    from . import invoice
except ImportError:
    invoice = None

try:
    from . import ml_engine
except ImportError:
    ml_engine = None

try:
    from . import notifications
except ImportError:
    notifications = None

try:
    from . import cache_management
except ImportError:
    cache_management = None

try:
    from . import feature_flags
except ImportError:
    feature_flags = None

__all__ = [
    'health',
    'fraud', 
    'insights',
    'compliance',
    'invoice',
    'ml_engine',
    'notifications',
    'cache_management',
    'feature_flags'
]