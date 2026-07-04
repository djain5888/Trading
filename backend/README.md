# Titan — Backend

Backend foundation for **Titan**, an AI quant trading platform for the Indian
stock market.

This repository currently contains **only the project skeleton**: a
production-ready FastAPI application with configuration, logging, a modular
package layout, and system endpoints. Trading logic, AI, indicators, scanners,
strategies, and market integrations are intentionally **not** included yet.

## Requirements

- Python **3.13+**

## Setup

```bash
cd backend

# Create and activate a virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install the app with development tooling
pip install -e ".[dev]"

# Configure the environment
cp .env.example .env
```

## Running

```bash
# Via the console script
titan

# Or directly with Uvicorn
uvicorn app.main:app --reload
```

The API is then available at http://localhost:8000.

## Endpoints

| Method | Path        | Description                         |
| ------ | ----------- | ----------------------------------- |
| GET    | `/health`   | Liveness probe (`{"status": "ok"}`) |
| GET    | `/version`  | App name, version and environment   |
| GET    | `/docs`     | Interactive OpenAPI (Swagger) UI    |

## Project layout

```
backend/
├── app/
│   ├── api/           # HTTP routers
│   ├── core/          # App factory, logging, cross-cutting concerns
│   ├── config/        # Settings / environment configuration
│   ├── database/      # DB session & engine (future)
│   ├── models/        # ORM models (future)
│   ├── schemas/       # Pydantic request/response schemas
│   ├── repositories/  # Data-access layer (future)
│   ├── services/      # Business services (future)
│   ├── providers/     # External data/broker providers (future)
│   ├── market/        # Market data (future)
│   ├── indicators/    # Technical indicators (future)
│   ├── scanner/       # Market scanners (future)
│   ├── strategies/    # Trading strategies (future)
│   ├── risk/          # Risk management (future)
│   ├── portfolio/     # Portfolio management (future)
│   ├── paper_trading/ # Paper trading engine (future)
│   ├── analytics/     # Analytics (future)
│   ├── backtesting/   # Backtesting engine (future)
│   ├── learning/      # Learning / research (future)
│   ├── ai/            # AI / ML (future)
│   ├── reports/       # Reporting (future)
│   ├── scheduler/     # Scheduled jobs (future)
│   ├── utils/         # Shared utilities
│   └── main.py        # ASGI entry point
└── tests/
```

## Development

```bash
ruff check .        # Lint
black .             # Format
mypy                # Static type checking
pytest              # Tests
```

## License

Proprietary. All rights reserved.
