from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from app.services.cognee_client import CogneeClientException
import logging

logger = logging.getLogger(__name__)

def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(CogneeClientException)
    async def cognee_exception_handler(request: Request, exc: CogneeClientException):
        logger.error(f"Cognee API error caught in global exception handler: {exc.details or str(exc)}")
        origin = request.headers.get("origin") or "*"
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true" if origin != "*" else "false",
            "Access-Control-Allow-Methods": "POST, GET, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
        return JSONResponse(
            status_code=exc.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "CogneeServiceError",
                "message": str(exc),
                "details": exc.details
            },
            headers=headers
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
        origin = request.headers.get("origin") or "*"
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true" if origin != "*" else "false",
            "Access-Control-Allow-Methods": "POST, GET, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred in the CogniVerdict backend.",
                "details": str(exc)
            },
            headers=headers
        )

