# AI Services package

# Import all service modules with error handling
try:
    from . import fraud_detection
except ImportError:
    fraud_detection = None

try:
    from . import predictive_analytics
except ImportError:
    predictive_analytics = None

try:
    from . import compliance
except ImportError:
    compliance = None

try:
    from . import nlp_invoice
except ImportError:
    nlp_invoice = None

try:
    from . import ml_engine
except ImportError:
    ml_engine = None

try:
    from . import smart_notifications
except ImportError:
    smart_notifications = None

try:
    from . import gemini_service
except ImportError:
    gemini_service = None

__all__ = [
    'fraud_detection',
    'predictive_analytics',
    'compliance',
    'nlp_invoice',
    'ml_engine',
    'smart_notifications',
    'gemini_service'
]