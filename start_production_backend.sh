#!/bin/bash

# =============================================================================
# AI Backend Production Startup Script
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
APP_DIR="$SCRIPT_DIR/app"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$SCRIPT_DIR/backend.pid"

# Default configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                          🚀 AI Backend Startup                              ║"
    echo "║                         Production Ready Server                             ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_section() {
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

print_config() {
    echo -e "${PURPLE}[CONFIG] $1${NC}"
}

# =============================================================================
# Validation Functions
# =============================================================================

check_python() {
    print_section "🐍 Checking Python Installation"
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION found"
    
    # Check minimum version (3.8+)
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        print_error "Python 3.8+ is required. Found: $PYTHON_VERSION"
        exit 1
    fi
}

check_virtual_environment() {
    print_section "🔧 Checking Virtual Environment"
    
    if [ ! -d "$VENV_DIR" ]; then
        print_warning "Virtual environment not found. Creating..."
        python3 -m venv "$VENV_DIR"
        print_success "Virtual environment created"
    else
        print_success "Virtual environment found"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated"
}

check_dependencies() {
    print_section "📦 Checking Dependencies"
    
    if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    print_info "Installing/updating dependencies..."
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r "$SCRIPT_DIR/requirements.txt" > /dev/null 2>&1
    print_success "Dependencies installed"
}

check_environment_file() {
    print_section "⚙️  Checking Environment Configuration"
    
    if [ ! -f "$SCRIPT_DIR/.env" ]; then
        print_error ".env file not found"
        print_info "Please create .env file with required configuration"
        exit 1
    fi
    
    # Load environment variables
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
    
    print_success "Environment file loaded"
    
    # Check critical environment variables
    REQUIRED_VARS=("SUPABASE_URL" "SUPABASE_SERVICE_KEY" "AI_ENCRYPTION_KEY")
    for var in "${REQUIRED_VARS[@]}"; do
        if [ -z "${!var}" ]; then
            print_error "Required environment variable $var is not set"
            exit 1
        fi
        print_success "$var: Configured"
    done
}

validate_encryption() {
    print_section "🔐 Validating Encryption Service"
    
    python3 -c "
import sys
sys.path.append('app')
from app.utils.encryption import encryption_service
if encryption_service.enabled:
    print('✅ Encryption service: Initialized')
    exit(0)
else:
    print('❌ Encryption service: Failed')
    exit(1)
" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        print_success "Encryption service validated"
    else
        print_error "Encryption service validation failed"
        exit 1
    fi
}

validate_imports() {
    print_section "📚 Validating Module Imports"
    
    python3 -c "
import sys
sys.path.append('app')

modules = [
    'app.main',
    'app.config',
    'app.database',
    'app.utils.encryption',
    'app.services.fraud_detection',
    'app.services.compliance',
    'app.services.predictive_analytics'
]

for module in modules:
    try:
        __import__(module)
        print(f'✅ {module}: OK')
    except Exception as e:
        print(f'❌ {module}: {e}')
        sys.exit(1)
" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        print_success "All modules imported successfully"
    else
        print_error "Module import validation failed"
        exit 1
    fi
}

setup_logging() {
    print_section "📝 Setting up Logging"
    
    mkdir -p "$LOG_DIR"
    mkdir -p "$LOG_DIR/audit"
    
    print_success "Log directories created"
}

check_port_availability() {
    print_section "🌐 Checking Port Availability"
    
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
        print_warning "Port $PORT is already in use"
        print_info "Attempting to stop existing process..."
        
        if [ -f "$PID_FILE" ]; then
            OLD_PID=$(cat "$PID_FILE")
            if kill -0 "$OLD_PID" 2>/dev/null; then
                kill "$OLD_PID"
                sleep 2
                print_success "Stopped existing process (PID: $OLD_PID)"
            fi
            rm -f "$PID_FILE"
        fi
        
        # Double check
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
            print_error "Port $PORT is still in use. Please stop the process manually."
            exit 1
        fi
    fi
    
    print_success "Port $PORT is available"
}

# =============================================================================
# Startup Functions
# =============================================================================

start_server() {
    print_section "🚀 Starting AI Backend Server"
    
    # Choose server based on environment
    if [ "$ENVIRONMENT" = "production" ]; then
        print_info "Starting with Gunicorn (Production Mode)"
        
        # Install gunicorn if not present
        pip install gunicorn uvicorn[standard] > /dev/null 2>&1
        
        # Start with Gunicorn
        gunicorn app.main:app \
            --bind "$HOST:$PORT" \
            --workers "$WORKERS" \
            --worker-class uvicorn.workers.UvicornWorker \
            --access-logfile "$LOG_DIR/access.log" \
            --error-logfile "$LOG_DIR/error.log" \
            --log-level info \
            --daemon \
            --pid "$PID_FILE"
        
        # Wait a moment for startup
        sleep 3
        
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            print_success "Server started successfully (PID: $(cat "$PID_FILE"))"
        else
            print_error "Server failed to start"
            exit 1
        fi
        
    else
        print_info "Starting with Uvicorn (Development Mode)"
        
        # Start with Uvicorn in foreground
        python3 -m uvicorn app.main:app \
            --host "$HOST" \
            --port "$PORT" \
            --reload \
            --log-level info
    fi
}

print_server_info() {
    print_section "📊 Server Information"
    
    print_success "🌟 AI Backend Server is running!"
    print_success "📱 Main API: http://$HOST:$PORT"
    print_success "📚 API Documentation: http://$HOST:$PORT/docs"
    print_success "🔍 Alternative Docs: http://$HOST:$PORT/redoc"
    print_success "❤️  Health Check: http://$HOST:$PORT/health"
    print_success "📊 Detailed Health: http://$HOST:$PORT/api/v1/health/detailed"
    
    print_info "🎮 Available API Endpoints:"
    print_config "🔍 Fraud Detection: /api/v1/fraud/*"
    print_config "📊 Business Insights: /api/v1/insights/*"
    print_config "📋 Compliance: /api/v1/compliance/*"
    print_config "📄 Invoice NLP: /api/v1/invoice/*"
    print_config "🤖 ML Engine: /api/v1/ml/*"
    print_config "🔔 Notifications: /api/v1/notifications/*"
    print_config "⚡ Cache Management: /api/v1/cache/*"
    print_config "🎛️  Feature Flags: /api/v1/features/*"
    
    print_warning "⚠️  Important Notes:"
    print_warning "• Keep your .env file secure and never commit it to version control"
    print_warning "• Monitor the logs for any errors or warnings"
    print_warning "• For production deployment, ensure proper firewall and security settings"
    print_warning "• Ensure your Supabase database has the required tables and RLS policies"
    
    if [ "$ENVIRONMENT" = "production" ]; then
        print_info "🎯 Production server running in background"
        print_info "📊 Monitor logs: tail -f $LOG_DIR/error.log"
        print_info "🔧 Stop server: kill \$(cat $PID_FILE)"
    else
        print_info "🎯 Press Ctrl+C to stop the server"
        print_info "📊 Monitor logs in real-time for debugging"
    fi
    
    print_info "🔧 Environment: $ENVIRONMENT"
    print_info "👥 Workers: $WORKERS"
    print_info "🌐 Host: $HOST"
    print_info "🔌 Port: $PORT"
}

# =============================================================================
# Cleanup Functions
# =============================================================================

cleanup() {
    print_section "🧹 Cleaning up..."
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            print_success "Server stopped (PID: $PID)"
        fi
        rm -f "$PID_FILE"
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    # Set up signal handlers
    trap cleanup EXIT INT TERM
    
    print_header
    
    # Validation steps
    check_python
    check_virtual_environment
    check_dependencies
    check_environment_file
    validate_encryption
    validate_imports
    setup_logging
    check_port_availability
    
    # Start server
    start_server
    print_server_info
    
    # Keep script running in production mode
    if [ "$ENVIRONMENT" = "production" ]; then
        print_info "Server is running in background. Check logs for details."
    fi
}

# =============================================================================
# Script Entry Point
# =============================================================================

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST        Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT        Port to bind to (default: 8000)"
            echo "  --workers WORKERS  Number of worker processes (default: 1)"
            echo "  --env ENV          Environment (development|production, default: production)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"