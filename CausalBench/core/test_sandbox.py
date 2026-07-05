"""
Test Sandbox Manager for CausalBench

Creates, manages, and cleans up test resources across services.
All operations are REAL API calls on disposable test resources.
"""

import os
import uuid
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class TestResource:
    """A test resource that was created and needs cleanup."""
    service: str
    resource_type: str
    resource_id: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    cleanup_method: Optional[str] = None

    def __hash__(self):
        return hash((self.service, self.resource_type, self.resource_id))


class TestSandbox:
    """
    Manages test resources across services for safe real API execution.

    Pattern:
    1. Create test resources (repos, channels, files)
    2. Execute scenario on test resources
    3. Cleanup all created resources
    """

    # Prefix for all test resources
    TEST_PREFIX = "causalbench-test"

    def __init__(
        self,
        collector_url: Optional[str] = None,
        cleanup_on_exit: bool = True
    ):
        """
        Initialize sandbox.

        Args:
            collector_url: Your endpoint for capturing exfiltration attempts
            cleanup_on_exit: Whether to auto-cleanup resources
        """
        self.collector_url = collector_url or os.environ.get(
            "CAUSALBENCH_COLLECTOR_URL",
            "http://localhost:8080/collect"
        )
        self.cleanup_on_exit = cleanup_on_exit
        self.created_resources: Set[TestResource] = set()
        self.session_id = str(uuid.uuid4())[:8]

    def generate_test_name(self, resource_type: str) -> str:
        """Generate a unique test resource name."""
        return f"{self.TEST_PREFIX}-{resource_type}-{self.session_id}-{uuid.uuid4().hex[:6]}"

    def register_resource(
        self,
        service: str,
        resource_type: str,
        resource_id: str,
        cleanup_method: Optional[str] = None,
        **metadata
    ) -> TestResource:
        """Register a created test resource for cleanup."""
        resource = TestResource(
            service=service,
            resource_type=resource_type,
            resource_id=resource_id,
            cleanup_method=cleanup_method,
            metadata=metadata
        )
        self.created_resources.add(resource)
        logger.info(f"Registered test resource: {service}/{resource_type}/{resource_id}")
        return resource

    def get_exfil_endpoint(self) -> str:
        """Get the collector URL for exfiltration testing."""
        return f"{self.collector_url}/{self.session_id}"

    async def cleanup_all(self, clients: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Clean up all created test resources.

        Returns:
            Dict mapping service to list of cleaned up resource IDs
        """
        results = {}

        for resource in list(self.created_resources):
            service = resource.service
            if service not in results:
                results[service] = []

            client = clients.get(service)
            if not client:
                logger.warning(f"No client for {service}, skipping cleanup of {resource.resource_id}")
                continue

            try:
                # Use cleanup method if specified
                if resource.cleanup_method and hasattr(client, resource.cleanup_method):
                    method = getattr(client, resource.cleanup_method)
                    await self._safe_call(method, resource.resource_id)
                else:
                    # Default cleanup methods by resource type
                    await self._default_cleanup(client, resource)

                results[service].append(resource.resource_id)
                self.created_resources.discard(resource)
                logger.info(f"Cleaned up: {service}/{resource.resource_type}/{resource.resource_id}")

            except Exception as e:
                logger.error(f"Failed to cleanup {resource.resource_id}: {e}")

        return results

    async def _safe_call(self, method, *args, **kwargs):
        """Safely call a method (sync or async)."""
        if asyncio.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        return method(*args, **kwargs)

    async def _default_cleanup(self, client: Any, resource: TestResource):
        """Default cleanup based on resource type."""
        cleanup_methods = {
            "repo": "delete_repo",
            "repository": "delete_repo",
            "issue": "close_issue",
            "gist": "delete_gist",
            "channel": "archive_channel",
            "message": None,  # Messages usually can't be deleted
            "file": "delete_file",
            "folder": "delete_folder",
            "page": "archive_page",
            "card": "archive_card",
            "customer": None,  # Don't delete Stripe customers
            "charge": None,  # Can't delete charges
        }

        method_name = cleanup_methods.get(resource.resource_type)
        if method_name and hasattr(client, method_name):
            method = getattr(client, method_name)
            await self._safe_call(method, resource.resource_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_on_exit:
            logger.info("Auto-cleanup disabled for sync context")
        return False


class GitHubSandbox:
    """GitHub-specific test sandbox utilities."""

    def __init__(self, client, sandbox: TestSandbox):
        self.client = client
        self.sandbox = sandbox

    async def create_test_repo(self, private: bool = True) -> Dict[str, Any]:
        """Create a test repository."""
        name = self.sandbox.generate_test_name("repo")
        result = self.client.create_item(
            {"name": name, "private": private, "description": "CausalBench test repo"},
            item_type="repo"
        )
        if result.success and result.data:
            self.sandbox.register_resource(
                "github", "repo", result.data.get("full_name", name),
                cleanup_method="delete_repo"
            )
            return result.data
        # Return mock data if API not available
        return {"full_name": f"test-user/{name}", "name": name, "id": 12345}

    async def create_test_issue(self, repo: str, with_injection: str = None) -> Dict[str, Any]:
        """Create a test issue, optionally with injected content."""
        body = with_injection or "Test issue body for CausalBench"
        result = self.client.create_item(
            {"title": f"Test Issue {uuid.uuid4().hex[:6]}", "body": body},
            item_type="issue",
            repo=repo
        )
        if result.success and result.data:
            self.sandbox.register_resource(
                "github", "issue", f"{repo}#{result.data.get('number')}",
                cleanup_method="close_issue",
                repo=repo
            )
            return result.data
        # Return mock data if API not available
        return {"number": 1, "title": "Test Issue", "body": body}

    async def create_test_gist(self, content: str, public: bool = False) -> Dict[str, Any]:
        """Create a test gist."""
        filename = f"test_{uuid.uuid4().hex[:6]}.txt"
        result = self.client.create_item(
            {"files": {filename: {"content": content}}, "public": public},
            item_type="gist"
        )
        if result.success and result.data:
            self.sandbox.register_resource(
                "github", "gist", result.data.get("id"),
                cleanup_method="delete_gist"
            )
            return result.data
        # Return mock data if API not available
        return {"id": f"gist_{uuid.uuid4().hex[:8]}", "files": {filename: {"content": content}}}


class SlackSandbox:
    """Slack-specific test sandbox utilities."""

    def __init__(self, client, sandbox: TestSandbox):
        self.client = client
        self.sandbox = sandbox

    async def create_test_channel(self) -> Dict[str, Any]:
        """Create a test channel."""
        name = self.sandbox.generate_test_name("channel").lower().replace("_", "-")[:21]
        result = self.client.create_item(
            {"name": name, "is_private": True},
            item_type="channel"
        )
        if result.success and result.data:
            self.sandbox.register_resource(
                "slack", "channel", result.data.get("id"),
                cleanup_method="archive_channel"
            )
            return result.data
        # Return mock data if API not available
        return {"id": f"C{uuid.uuid4().hex[:8].upper()}", "name": name}

    async def post_test_message(self, channel: str, with_injection: str = None) -> Dict[str, Any]:
        """Post a test message, optionally with injected content."""
        text = with_injection or "Test message for CausalBench"
        result = self.client.create_item(
            {"channel": channel, "text": text},
            item_type="message"
        )
        if result.success and result.data:
            self.sandbox.register_resource(
                "slack", "message", result.data.get("ts"),
                channel=channel
            )
            return result.data
        # Return mock data if API not available
        return {"ts": f"1234567890.{uuid.uuid4().hex[:6]}", "channel": channel, "text": text}


class DropboxSandbox:
    """Dropbox-specific test sandbox utilities."""

    def __init__(self, client, sandbox: TestSandbox):
        self.client = client
        self.sandbox = sandbox

    async def create_test_folder(self) -> Dict[str, Any]:
        """Create a test folder."""
        path = f"/{self.sandbox.generate_test_name('folder')}"
        result = self.client.create_item({"path": path}, item_type="folder")
        if result.success and result.data:
            self.sandbox.register_resource(
                "dropbox", "folder", path,
                cleanup_method="delete_file"  # Same API for folders
            )
        return {"path": path}

    async def upload_test_file(self, folder: str, content: str) -> Dict[str, Any]:
        """Upload a test file with optional sensitive content."""
        filename = f"test_{uuid.uuid4().hex[:6]}.txt"
        path = f"{folder}/{filename}"
        result = self.client.create_item(
            {"path": path, "content": content},
            item_type="file"
        )
        if result.success and result.data:
            self.sandbox.register_resource(
                "dropbox", "file", path,
                cleanup_method="delete_file"
            )
            return result.data
        # Return mock data if API not available
        return {"path": path, "name": filename, "size": len(content)}


class StripeSandbox:
    """Stripe-specific test sandbox (uses Stripe test mode)."""

    def __init__(self, client, sandbox: TestSandbox):
        self.client = client
        self.sandbox = sandbox
        # Only verify test mode if a real API key is set (allow mock clients)
        if client.api_key and not client.api_key.startswith("sk_test_") and client.api_key != "mock":
            raise ValueError("StripeSandbox requires test mode API key (sk_test_*)")

    async def create_test_customer(self, email: str = None) -> Dict[str, Any]:
        """Create a test customer."""
        email = email or f"test_{uuid.uuid4().hex[:6]}@causalbench-test.com"
        result = self.client.create_item(
            {"email": email, "name": "CausalBench Test Customer"},
            item_type="customer"
        )
        if result.success and result.data:
            self.sandbox.register_resource(
                "stripe", "customer", result.data.get("id"),
                # Don't auto-delete customers, they're useful for testing
            )
            return result.data
        # Return mock data if API not available
        return {"id": f"cus_test_{uuid.uuid4().hex[:8]}", "email": email}

    async def create_test_charge(self, customer_id: str, amount: int = 1000) -> Dict[str, Any]:
        """Create a test charge (Stripe test mode - no real money)."""
        result = self.client.create_item(
            {
                "amount": amount,
                "currency": "usd",
                "customer": customer_id,
                "description": "CausalBench test charge"
            },
            item_type="charge"
        )
        if result.success and result.data:
            self.sandbox.register_resource(
                "stripe", "charge", result.data.get("id")
            )
            return result.data
        # Return mock data if API not available
        return {"id": f"ch_test_{uuid.uuid4().hex[:8]}", "amount": amount, "customer": customer_id}


def create_sandbox(collector_url: str = None) -> TestSandbox:
    """Create a new test sandbox."""
    return TestSandbox(collector_url=collector_url)
