# Minimal FastAPI Project Base

A streamlined foundation for building Python web applications using FastAPI.

## Features

- **FastAPI Core**: Leverages the high-performance FastAPI framework.
- **Docker Support**: Production-ready containerization with a multi-stage Dockerfile.
- **Fly.io Optimized**: Includes a `fly.toml` for easy deployment with auto-scaling and cost-saving measures.
- **Health Monitoring**: Basic health check endpoint (`/health`) included.
- **Environment Configuration**: Uses `.env` files for managing settings.

## Project Structure

```
project_base/
├── app/
│   ├── __init__.py
│   ├── api/            # API endpoints (e.g., FastAPI routers)
│   │   └── __init__.py
│   ├── core/           # Core configuration, settings, error handling, logging
│   │   └── __init__.py
│   ├── frontend/       # UI implementations (e.g., NiceGUI pages, ReactPy components, FastAPI routes)
│   │   ├── __init__.py
│   │   # ├── nicegui_app.py  # Example: NiceGUI implementation
│   │   # ├── reactpy_app.py  # Example: ReactPy implementation
│   │   # └── routes.py       # Example: FastAPI frontend routes
│   ├── generated/      # AI-generated application code
│   │   └── __init__.py
│   ├── models/         # Data models & schemas (e.g., Pydantic, SQLAlchemy)
│   │   └── __init__.py
│   ├── services/       # Business logic & external API integrations
│   │   └── __init__.py
│   ├── static/         # Static assets (CSS, JS, images). ALL image files MUST be placed here or in subdirectories within static/. Do NOT create separate top-level image directories like 'pictures/'.
│   ├── templates/      # HTML templates (Jinja2)
│   └── main.py         # Defines FastAPI routes and application logic for the 'app' module
├── .dockerignore         # Specifies intentionally untracked files for Docker
├── .env                  # Environment variables (create this file based on .env.example if provided)
├── Dockerfile            # Container configuration
├── fly.toml              # fly.io deployment configuration
├── main.py               # Application entry point (runs the Uvicorn server)
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)
- Fly.io account and `flyctl` CLI (optional, for Fly.io deployment)

### Installation

1.  **Clone the repository (if applicable)**
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Create a `.env` file** in the `project_base` directory (you can copy `.env.example` if one exists and modify it). At a minimum, it might look like this if you want to change the default port:
    ```env
    PORT=8000
    HOST=0.0.0.0
    ```
    If no `.env` file is present, the application will use default values (e.g., port 8000).

### Running the Application Locally

Execute the main application script:

```bash
python main.py
```

The application will typically be available at `http://0.0.0.0:8000` (or the port specified in your `.env` file).

## API Endpoints

-   `GET /`: Returns a welcome message.
-   `GET /health`: Returns a health status, useful for monitoring.

## Deployment

### Docker Deployment

1.  **Build the Docker image:**
    ```bash
    docker build -t my-fastapi-app .
    ```
2.  **Run the Docker container:**
    ```bash
    docker run -p 8000:8000 -d my-fastapi-app
    ```
    Replace `8000:8000` with `<host_port>:<container_port>` if you need to map to a different host port. The container port is determined by the `PORT` environment variable set in the `Dockerfile` or `fly.toml` (defaulting to 8000).

### Fly.io Deployment

1.  **Install `flyctl`**: Follow the instructions at [fly.io/docs/hands-on/install-flyctl/](https://fly.io/docs/hands-on/install-flyctl/).
2.  **Login to Fly.io**: `fly auth login`
3.  **Launch the app (first time only)**:
    ```bash
    fly launch --name your-unique-app-name --region sin
    ```
    (Replace `your-unique-app-name` and `sin` (Singapore) with your desired app name and region. This will also create a `fly.toml` if one doesn't exist, or update the existing one.)
4.  **Deploy changes**:
    ```bash
    fly deploy
    ```

The `fly.toml` file is pre-configured for auto-scaling and to stop machines when idle to save costs.

## Customization

-   **Add new API endpoints**: Modify `project_base/app/main.py` to include new routes and logic.
-   **Modify dependencies**: Update `project_base/requirements.txt` and reinstall.
-   **Adjust Docker configuration**: Edit `project_base/Dockerfile`.
-   **Change deployment settings**: Update `project_base/fly.toml` for Fly.io.

## Core Principles for Development

While this base is minimal, consider these principles as you expand your application:

-   **Modularity**: Keep code organized into logical modules.
-   **Clarity**: Write clear, understandable code with type hints where appropriate.
-   **Testing**: Implement unit and integration tests for new features.
-   **Security**: Follow security best practices (input validation, authentication if needed, etc.).
-   **Documentation**: Keep this README and code comments up-to-date.