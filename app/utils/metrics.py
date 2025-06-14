from functools import wraps
import time
from prometheus_client import Counter, Histogram
import logging

# Define metrics
REQUEST_COUNT = Counter("app_request_count", "Total number of requests", ["endpoint"])
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Request latency in seconds", ["endpoint"]
)
DOCUMENT_PROCESSING_TIME = Histogram(
    "document_processing_seconds", "Document processing time in seconds"
)
RAG_QUERY_TIME = Histogram(
    "rag_query_seconds", "RAG query processing time in seconds", ["model"]
)
AGENT_PROCESSING_TIME = Histogram(
    "agent_processing_seconds", "Agent processing time in seconds", ["operation"]
)


def track_request_metrics(func):
    """Decorator to track request metrics."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        endpoint = func.__name__
        REQUEST_COUNT.labels(endpoint=endpoint).inc()

        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)

    return wrapper


def track_document_processing(func):
    """Decorator to track document processing metrics."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            DOCUMENT_PROCESSING_TIME.observe(duration)

    return wrapper


def track_rag_query(model):
    """Decorator to track RAG query metrics."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                RAG_QUERY_TIME.labels(model=model).observe(duration)

        return wrapper

    return decorator


def track_agent_processing(operation):
    """Decorator to track agent processing metrics."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                AGENT_PROCESSING_TIME.labels(operation=operation).observe(duration)

        return wrapper

    return decorator
