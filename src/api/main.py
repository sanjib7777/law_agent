from fastapi import APIRouter

from .routes import upload_routes, retrieve_routes

api_router = APIRouter()
api_router.include_router(upload_routes.router)
api_router.include_router(retrieve_routes.router)