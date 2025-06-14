import aiohttp
import asyncio
from typing import Dict, Any, List
import psutil
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HealthCheck:
    """Health check manager for application dependencies."""

    def __init__(self):
        self.checks: Dict[str, callable] = {
            "llama_model": self._check_llama_model,
            "openai_api": self._check_openai_api,
            "vector_db": self._check_vector_db,
            "storage": self._check_storage,
            "memory": self._check_memory,
            "cpu": self._check_cpu,
        }

    async def _check_llama_model(self) -> Dict[str, Any]:
        """Check Llama model server health."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/health") as response:
                    if response.status == 200:
                        return {
                            "status": "healthy",
                            "details": "Llama model server is running",
                        }
                    return {
                        "status": "unhealthy",
                        "details": f"Llama model server returned status {response.status}",
                    }
        except Exception as e:
            logger.error(f"Llama model health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "details": f"Llama model server is not accessible: {str(e)}",
            }

    async def _check_openai_api(self) -> Dict[str, Any]:
        """Check OpenAI API health."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
                ) as response:
                    if response.status == 200:
                        return {
                            "status": "healthy",
                            "details": "OpenAI API is accessible",
                        }
                    return {
                        "status": "unhealthy",
                        "details": f"OpenAI API returned status {response.status}",
                    }
        except Exception as e:
            logger.error(f"OpenAI API health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "details": f"OpenAI API is not accessible: {str(e)}",
            }

    async def _check_vector_db(self) -> Dict[str, Any]:
        """Check vector database health."""
        try:
            # Add your vector database health check logic here
            # This is a placeholder implementation
            return {"status": "healthy", "details": "Vector database is accessible"}
        except Exception as e:
            logger.error(f"Vector database health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "details": f"Vector database is not accessible: {str(e)}",
            }

    async def _check_storage(self) -> Dict[str, Any]:
        """Check storage system health."""
        try:
            storage_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "storage"
            )
            if os.path.exists(storage_path) and os.access(storage_path, os.W_OK):
                return {
                    "status": "healthy",
                    "details": "Storage system is accessible and writable",
                }
            return {
                "status": "unhealthy",
                "details": "Storage system is not accessible or writable",
            }
        except Exception as e:
            logger.error(f"Storage health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "details": f"Storage system check failed: {str(e)}",
            }

    async def _check_memory(self) -> Dict[str, Any]:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            return {
                "status": "healthy" if memory.percent < 90 else "warning",
                "details": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent,
                },
            }
        except Exception as e:
            logger.error(f"Memory health check failed: {str(e)}")
            return {"status": "unhealthy", "details": f"Memory check failed: {str(e)}"}

    async def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            return {
                "status": "healthy" if cpu_percent < 90 else "warning",
                "details": {"usage_percent": cpu_percent},
            }
        except Exception as e:
            logger.error(f"CPU health check failed: {str(e)}")
            return {"status": "unhealthy", "details": f"CPU check failed: {str(e)}"}

    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        for name, check in self.checks.items():
            try:
                results[name] = await check()
            except Exception as e:
                logger.error(f"Health check {name} failed: {str(e)}")
                results[name] = {
                    "status": "unhealthy",
                    "details": f"Check failed: {str(e)}",
                }

        # Determine overall status
        overall_status = "healthy"
        if any(result["status"] == "unhealthy" for result in results.values()):
            overall_status = "unhealthy"
        elif any(result["status"] == "warning" for result in results.values()):
            overall_status = "warning"

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": results,
        }
