"""
Middleware for API security and rate limiting.
"""

import time
from collections import defaultdict
from typing import Callable, Optional

from fastapi import Request, HTTPException, status
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

from music_brain.auth import decode_access_token, verify_api_key


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm.
    
    Tracks requests per IP address or API key and enforces rate limits.
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        
        # Track requests: {identifier: [(timestamp, ...), ...]}
        self.minute_requests: dict[str, list[float]] = defaultdict(list)
        self.hour_requests: dict[str, list[float]] = defaultdict(list)
        self.day_requests: dict[str, list[float]] = defaultdict(list)
        
        # Cleanup old entries periodically
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get client identifier from request (IP or API key)."""
        # Check for API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key[:16]}"
        
        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        
        return f"ip:{ip}"
    
    def _cleanup_old_entries(self, current_time: float):
        """Remove old entries from tracking dictionaries."""
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        # Clean minute requests (older than 1 minute)
        for key in list(self.minute_requests.keys()):
            self.minute_requests[key] = [
                ts for ts in self.minute_requests[key]
                if current_time - ts < 60
            ]
            if not self.minute_requests[key]:
                del self.minute_requests[key]
        
        # Clean hour requests (older than 1 hour)
        for key in list(self.hour_requests.keys()):
            self.hour_requests[key] = [
                ts for ts in self.hour_requests[key]
                if current_time - ts < 3600
            ]
            if not self.hour_requests[key]:
                del self.hour_requests[key]
        
        # Clean day requests (older than 1 day)
        for key in list(self.day_requests.keys()):
            self.day_requests[key] = [
                ts for ts in self.day_requests[key]
                if current_time - ts < 86400
            ]
            if not self.day_requests[key]:
                del self.day_requests[key]
        
        self.last_cleanup = current_time
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and enforce rate limits."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)
        
        current_time = time.time()
        self._cleanup_old_entries(current_time)
        
        identifier = self._get_client_identifier(request)
        
        # Check minute limit
        minute_reqs = self.minute_requests[identifier]
        minute_reqs = [ts for ts in minute_reqs if current_time - ts < 60]
        if len(minute_reqs) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.requests_per_minute} requests per minute",
                headers={"Retry-After": "60"},
            )
        
        # Check hour limit
        hour_reqs = self.hour_requests[identifier]
        hour_reqs = [ts for ts in hour_reqs if current_time - ts < 3600]
        if len(hour_reqs) >= self.requests_per_hour:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.requests_per_hour} requests per hour",
                headers={"Retry-After": "3600"},
            )
        
        # Check day limit
        day_reqs = self.day_requests[identifier]
        day_reqs = [ts for ts in day_reqs if current_time - ts < 86400]
        if len(day_reqs) >= self.requests_per_day:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.requests_per_day} requests per day",
                headers={"Retry-After": "86400"},
            )
        
        # Record request
        self.minute_requests[identifier].append(current_time)
        self.hour_requests[identifier].append(current_time)
        self.day_requests[identifier].append(current_time)
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)
        response.headers["X-RateLimit-Limit-Day"] = str(self.requests_per_day)
        response.headers["X-RateLimit-Remaining-Minute"] = str(
            self.requests_per_minute - len(minute_reqs) - 1
        )
        response.headers["X-RateLimit-Remaining-Hour"] = str(
            self.requests_per_hour - len(hour_reqs) - 1
        )
        response.headers["X-RateLimit-Remaining-Day"] = str(
            self.requests_per_day - len(day_reqs) - 1
        )
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers."""
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'"
        )
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests for monitoring and debugging."""
    
    def __init__(self, app, logger=None):
        super().__init__(app)
        self.logger = logger
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request details."""
        import logging
        if not self.logger:
            self.logger = logging.getLogger("music_brain.api")
        
        start_time = time.time()
        
        # Log request
        self.logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        self.logger.info(
            f"Response: {request.method} {request.url.path} - {response.status_code}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration * 1000,
            }
        )
        
        # Add timing header
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response
