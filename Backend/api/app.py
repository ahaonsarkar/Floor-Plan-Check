'''FastAPI application entry point with CORS support.

This file creates the FastAPI `app` instance, adds the CORS middleware so that
the Vite dev server (http://localhost:3000) can call the API, and includes the
router defined in `Backend/api/main.py` which contains the `/upload-floor-plan`
endpoint and any other routes.
'''

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the existing router that holds all API endpoints.
# The router is defined in `Backend/api/main.py`.
from .main import router

app = FastAPI()

# ----------------------------------------------------------------------
# CORS – allow the Vite dev server to call the API
# ----------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Vite dev server origin
    allow_credentials=True,
    allow_methods=["*"],                     # Allow all HTTP methods
    allow_headers=["*"],                     # Allow any headers (including auth)
)

# Register the router containing the endpoint definitions.
app.include_router(router)
