from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as cases_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="CogniVerdict legal reasoning copilot backend powered by Cognee Cloud AI memory."
)

# Enable CORS for Next.js and other UI environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],  # Adjust in production to match your front-end domain
    allow_credentials=True,
    allow_methods=["POST","GET","PUT","DELETE","OPTIONS"],
    allow_headers=["*"],
)

# Register exception handlers
register_exception_handlers(app)

# Include API routes
app.include_router(cases_router)

@app.get("/")
async def root():
    """
    Root check endpoint.
    """
    return {
        "application": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
        "cognee_connected": settings.has_cognee_credentials
    }
