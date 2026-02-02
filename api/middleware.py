"""
Prometheus Metrics Middleware for FastAPI
"""

from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram
import time

# Metrics definitions
REQUEST_COUNT = Counter(
    "http_requests_total", 
    "Total HTTP requests", 
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"]
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_seconds",
    "Time taken for model inference",
    ["model_type"]
)

DISASTER_DETECTED = Counter(
    "disaster_events_detected",
    "Count of detected disaster types",
    ["disaster_type", "severity"]
)

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(process_time)
        
        return response
