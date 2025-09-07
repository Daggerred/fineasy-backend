# FinEasy AI Backend

AI-powered business intelligence backend for the FinEasy application.

## Features

- Fraud Detection and Error Prevention
- Predictive Business Analytics
- GST Compliance Checking (India-specific)
- Natural Language Invoice Generation
- Machine Learning Model Management
- Redis Caching for Performance

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Redis (for caching)
- Supabase account and credentials

### Environment Setup

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Update the `.env` file with your credentials:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key
SUPABASE_ANON_KEY=your_anon_key
```

### Development Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

4. Run the development server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Setup

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- API Documentation: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`
- Health check: `http://localhost:8000/health`

## API Endpoints

### Fraud Detection
- `POST /api/v1/fraud/analyze` - Analyze fraud for a business
- `GET /api/v1/fraud/alerts/{business_id}` - Get fraud alerts

### Business Insights
- `GET /api/v1/insights/{business_id}` - Get business insights
- `POST /api/v1/insights/generate` - Generate new insights

### Compliance
- `POST /api/v1/compliance/check` - Check invoice compliance
- `POST /api/v1/compliance/gst/validate` - Validate GST number

### Invoice Generation
- `POST /api/v1/invoice/generate` - Generate invoice from text
- `POST /api/v1/invoice/parse` - Parse invoice text

## Project Structure

```
ai-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── database.py          # Supabase connection
│   ├── models/              # Pydantic models
│   ├── services/            # AI service implementations
│   ├── api/                 # API endpoints
│   └── utils/               # Utility functions
├── ml_models/               # Trained ML models
├── logs/                    # Application logs
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
└── README.md               # This file
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black app/
flake8 app/
```

### Type Checking

```bash
mypy app/
```

## Configuration

Key environment variables:

- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_SERVICE_KEY` - Supabase service role key
- `REDIS_URL` - Redis connection URL
- `ENVIRONMENT` - development/production
- `LOG_LEVEL` - Logging level (DEBUG/INFO/WARNING/ERROR)

## Security

- All API endpoints require authentication
- Data is encrypted in transit and at rest
- Sensitive data is anonymized for ML processing
- Audit logging for all AI operations

## Monitoring

- Health check endpoint at `/health`
- Structured logging with correlation IDs
- Redis monitoring via Redis Commander (optional)
- Prometheus metrics (planned)

## License

Private - FinEasy Application