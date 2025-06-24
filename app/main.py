from nicegui import ui
from fastapi import Depends
import asyncio
import json

from app.core import app_logger, settings, security

# Define the main UI page
@ui.page('/')
def main_page():
    with ui.card().classes('w-full max-w-3xl mx-auto my-6'):
        ui.label(settings.app_name).classes('text-2xl font-bold mb-4')
        ui.markdown(f'''
        ## Welcome to {settings.app_name}!
        
        This is a modern Python application built with:
        - **NiceGUI** for the frontend
        - **FastAPI** for the API
        - **Pydantic** for data validation
        
        ### Features
        - Modern, responsive UI
        - RESTful API endpoints
        - JWT authentication
        - Comprehensive error handling
        - Logging and monitoring
        - Docker and Fly.io deployment
        ''')
        
        ui.separator()
        
        with ui.row().classes('w-full justify-center gap-4 mt-4'):
            ui.button('API Documentation', on_click=lambda: ui.open(f'{settings.api_prefix}/docs' if settings.api_prefix else '/docs'))
            ui.button('GitHub', on_click=lambda: ui.open('https://github.com'))
            ui.button('About', on_click=lambda: ui.open('/about'))

# About page
@ui.page('/about')
def about_page():
    with ui.card().classes('w-full max-w-3xl mx-auto my-6'):
        ui.label('About').classes('text-2xl font-bold mb-4')
        ui.markdown(f'''
        ## About {settings.app_name}
        
        This application was built as a modern Python web application template.
        It demonstrates best practices for building web applications with NiceGUI and FastAPI.
        
        ### Core Modules
        - **API**: RESTful endpoints with FastAPI
        - **Frontend**: UI components with NiceGUI
        - **Models**: Data validation with Pydantic
        - **Services**: Business logic and external integrations
        - **Core**: Configuration, logging, security, and utilities
        
        ### Deployment
        The application is designed to be deployed to Fly.io or any Docker-compatible platform.
        ''')
        
        with ui.row().classes('w-full justify-center mt-4'):
            ui.button('Back to Home', on_click=lambda: ui.navigate('/'))

# API Demo page
@ui.page('/api-demo')
def api_demo_page():
    async def fetch_examples():
        try:
            # This is a placeholder - in a real app, you would use httpx to call your API
            # For demo purposes, we'll return hardcoded data
            await asyncio.sleep(0.5)  # Simulate API call
            examples = [
                {"id": 1, "title": "Example 1", "description": "This is example 1", "owner": "demo"},
                {"id": 2, "title": "Example 2", "description": "This is example 2", "owner": "demo"},
            ]
            examples_container.clear()
            for example in examples:
                with ui.card().classes('w-full'):
                    ui.label(example["title"]).classes('text-lg font-bold')
                    ui.label(example["description"])
                    ui.label(f"Owner: {example['owner']}").classes('text-sm text-gray-500')
            status.text = "Examples loaded successfully"
            status.classes('text-green-600')
        except Exception as e:
            app_logger.error(f"Error fetching examples: {e}")
            status.text = f"Error: {str(e)}"
            status.classes('text-red-600')
    
    with ui.card().classes('w-full max-w-3xl mx-auto my-6'):
        ui.label('API Demo').classes('text-2xl font-bold mb-4')
        ui.markdown('''
        This page demonstrates how to interact with the API endpoints from the UI.
        In a real application, you would use httpx to make API calls to your FastAPI endpoints.
        ''')
        
        with ui.row().classes('w-full justify-between items-center'):
            ui.button('Fetch Examples', on_click=fetch_examples).props('color=primary')
            status = ui.label('Ready').classes('text-gray-600')
        
        examples_container = ui.column().classes('w-full mt-4 gap-2')
        
        with ui.row().classes('w-full justify-center mt-4'):
            ui.button('Back to Home', on_click=lambda: ui.navigate('/'))

# Define a health check page for Fly.io
@ui.page('/health')
def health_check_page():
    # Fly.io health checks typically look for a 200 OK status.
    # This page will return 200 OK. Content can be simple.
    ui.label('{"status": "healthy"}')

# Protected page example
@ui.page('/protected')
def protected_page(user = Depends(security.get_current_active_user)):
    with ui.card().classes('w-full max-w-3xl mx-auto my-6'):
        ui.label('Protected Page').classes('text-2xl font-bold mb-4')
        ui.label(f'Hello, {user.username}!')
        ui.label('This page is only accessible to authenticated users.')
        
        with ui.row().classes('w-full justify-center mt-4'):
            ui.button('Back to Home', on_click=lambda: ui.navigate('/'))

# Note: No ui.run() here. 
# This file only defines the UI pages and elements.
# The actual server will be started by project_base/main.py