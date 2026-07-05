"""
Base Client for API Services

Provides common functionality for all service clients.
"""

import os
import time
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """Standardized API response."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    status_code: int = 200
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "status_code": self.status_code,
            "timestamp": self.timestamp
        }


class RateLimiter:
    """Simple rate limiter to avoid API bans."""

    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.request_times: List[float] = []

    def wait_if_needed(self):
        """Wait if we've exceeded rate limit."""
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]

        if len(self.request_times) >= self.requests_per_minute:
            # Wait until oldest request is more than 1 minute old
            wait_time = 60 - (now - self.request_times[0]) + 0.1
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)

        self.request_times.append(time.time())


class BaseServiceClient(ABC):
    """
    Abstract base class for API service clients.

    All service clients should implement:
    - list_items()
    - get_item(id)
    - create_item(data)
    - export_data(destination) - For exfiltration simulation
    - share_external(resource, email) - For sharing simulation
    """

    SERVICE_NAME = "base"
    TRUST_LEVEL = "trusted"

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: int = 30,
        simulate_writes: bool = True
    ):
        """
        Initialize the service client.

        Args:
            api_key: API key/token for authentication
            rate_limit: Max requests per minute
            simulate_writes: If True, don't execute actual write operations
        """
        self.api_key = api_key
        self.rate_limiter = RateLimiter(rate_limit)
        self.simulate_writes = simulate_writes
        self._initialized = False

    @abstractmethod
    def _initialize_client(self) -> bool:
        """Initialize the underlying API client. Returns True if successful."""
        pass

    def ensure_initialized(self) -> bool:
        """Ensure the client is initialized."""
        if not self._initialized:
            self._initialized = self._initialize_client()
        return self._initialized

    @abstractmethod
    def list_items(self, limit: int = 10, **kwargs) -> APIResponse:
        """List items from the service."""
        pass

    @abstractmethod
    def get_item(self, item_id: str, **kwargs) -> APIResponse:
        """Get a specific item by ID."""
        pass

    @abstractmethod
    def create_item(self, data: Dict[str, Any], **kwargs) -> APIResponse:
        """Create a new item. May be simulated if simulate_writes=True."""
        pass

    def export_data(
        self,
        data: Any,
        destination: str,
        **kwargs
    ) -> APIResponse:
        """
        Simulate data export/exfiltration.

        This is ALWAYS simulated - never sends real data externally.
        """
        logger.warning(f"[SIMULATED] Would export data to {destination}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "destination": destination,
                "data_size": len(str(data)),
                "action": "export_data"
            }
        )

    def share_external(
        self,
        resource_id: str,
        email: str,
        permission: str = "read",
        **kwargs
    ) -> APIResponse:
        """
        Simulate sharing with external user.

        This is ALWAYS simulated for security.
        """
        logger.warning(f"[SIMULATED] Would share {resource_id} with {email}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "resource_id": resource_id,
                "shared_with": email,
                "permission": permission,
                "action": "share_external"
            }
        )

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an API request with rate limiting."""
        self.rate_limiter.wait_if_needed()
        # Subclasses implement actual API calls
        raise NotImplementedError

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about this service."""
        return {
            "service_name": self.SERVICE_NAME,
            "trust_level": self.TRUST_LEVEL,
            "initialized": self._initialized,
            "simulate_writes": self.simulate_writes
        }


class MockServiceClient(BaseServiceClient):
    """
    Mock service client for testing without real APIs.
    """

    SERVICE_NAME = "mock"
    TRUST_LEVEL = "trusted"

    def __init__(self, service_name: str = "mock", **kwargs):
        super().__init__(**kwargs)
        self.SERVICE_NAME = service_name
        self._mock_data = {}

    def _initialize_client(self) -> bool:
        return True

    def list_items(self, limit: int = 10, **kwargs) -> APIResponse:
        items = [
            {
                "id": f"item_{i}",
                "name": f"Mock Item {i}",
                "created_at": datetime.utcnow().isoformat()
            }
            for i in range(limit)
        ]
        return APIResponse(success=True, data={"items": items, "count": len(items)})

    def get_item(self, item_id: str, **kwargs) -> APIResponse:
        return APIResponse(
            success=True,
            data={
                "id": item_id,
                "name": f"Mock Item {item_id}",
                "content": f"Content for {item_id}",
                "created_at": datetime.utcnow().isoformat()
            }
        )

    def create_item(self, data: Dict[str, Any], **kwargs) -> APIResponse:
        if self.simulate_writes:
            return APIResponse(
                success=True,
                data={
                    "simulated": True,
                    "would_create": data
                }
            )
        item_id = f"item_{random.randint(1000, 9999)}"
        self._mock_data[item_id] = data
        return APIResponse(
            success=True,
            data={"id": item_id, **data}
        )


# Registry of available service clients
SERVICE_REGISTRY: Dict[str, type] = {
    "mock": MockServiceClient
}


def get_service_client(service_name: str, **kwargs) -> BaseServiceClient:
    """Get a service client by name."""
    if service_name not in SERVICE_REGISTRY:
        logger.warning(f"Unknown service {service_name}, using mock")
        return MockServiceClient(service_name=service_name, **kwargs)
    return SERVICE_REGISTRY[service_name](**kwargs)


def register_service(name: str, client_class: type):
    """Register a new service client."""
    SERVICE_REGISTRY[name] = client_class
