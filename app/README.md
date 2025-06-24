# Application Structure

This document provides an overview of the application structure and how to use it.

## Directory Structure

```
app/
├── api/            # API endpoints using FastAPI
├── core/           # Core functionality (config, logging, security, etc.)
├── frontend/       # Frontend components (if separate from main.py)
├── models/         # Data models and schemas
├── services/       # Business logic and external service integrations
├── static/         # Static files (CSS, JS, images)
├── templates/      # Template files (if using Jinja2)
├── main.py         # Main application entry point for NiceGUI
└── README.md       # This file
```

## Core Modules

The `app/core/` directory contains essential functionality for the application:

- **config.py**: Application configuration using Pydantic settings management
- **logging.py**: Logging setup with console and file handlers
- **exceptions.py**: Custom exception classes
- **error_handlers.py**: Exception handlers for FastAPI
- **middleware.py**: FastAPI middleware (CORS, compression, etc.)
- **security.py**: Authentication and authorization utilities
- **utils.py**: General utility functions
- **health.py**: Health check functionality
- **database.py**: Database connection and utilities
- **deployment.py**: Deployment utilities for Docker and Fly.io

## API Structure

The `app/api/` directory contains FastAPI routers organized by feature:

- **auth.py**: Authentication endpoints (login, token, etc.)
- **example.py**: Example CRUD endpoints
- **router.py**: Main router that includes all feature routers

All API endpoints are available under the `/api` prefix (configurable in settings).

## Models

The `app/models/` directory contains Pydantic models for request/response validation:

- **user.py**: User-related models (authentication, profiles)
- **example.py**: Example models for the example API

## Frontend

The application uses NiceGUI for the frontend, with pages defined in `app/main.py`:

- **/** - Home page
- **/about** - About page
- **/api-demo** - Demo of API interaction
- **/protected** - Example of a protected page requiring authentication
- **/health** - Health check endpoint for Fly.io

## Authentication

The application uses JWT-based authentication:

1. Clients obtain a token via `/api/auth/token` (OAuth2 password flow)
2. The token is included in the `Authorization: Bearer {token}` header for protected endpoints
3. Protected endpoints use the `Depends(security.get_current_active_user)` dependency

## Configuration

Configuration is managed through environment variables, with defaults in `app/core/config.py`.
Key settings include:

- `APP_NAME`: Application name
- `API_PREFIX`: Prefix for all API endpoints
- `HOST`: Host to bind the server to
- `PORT`: Port to bind the server to
- `DEBUG`: Enable debug mode
- `SECRET_KEY`: Secret key for JWT tokens and encryption
- `ACCESS_TOKEN_EXPIRE_MINUTES`: JWT token expiration time

## Adding New Features

### Adding a New API Endpoint

1. Create a new file in `app/api/` (e.g., `items.py`)
2. Define a router and endpoints
3. Import and include the router in `app/api/router.py`

### Adding a New Model

1. Create a new file in `app/models/` (e.g., `item.py`)
2. Define Pydantic models for requests and responses

### Adding a New UI Page

1. Add a new page function with the `@ui.page` decorator in `app/main.py`
2. For complex UIs, consider creating a separate file in `app/frontend/`

## Development Workflow

1. Make changes to the codebase
2. Run the application with `python main.py`
3. Access the UI at `http://localhost:8000`
4. Access the API docs at `http://localhost:8000/api/docs`