"""
Service Clients for CausalBench

Provides API clients for various services used in trajectory generation.
"""

from .base_client import (
    BaseServiceClient,
    MockServiceClient,
    APIResponse,
    RateLimiter,
    get_service_client,
    register_service,
    SERVICE_REGISTRY
)

# Import clients to register them
from .github_client import GitHubClient
from .slack_client import SlackClient
from .stripe_client import StripeClient
from .gmail_client import GmailClient
from .dropbox_client import DropboxClient
from .notion_client import NotionClient
from .trello_client import TrelloClient

__all__ = [
    # Base
    "BaseServiceClient",
    "MockServiceClient",
    "APIResponse",
    "RateLimiter",
    "get_service_client",
    "register_service",
    "SERVICE_REGISTRY",
    # Clients
    "GitHubClient",
    "SlackClient",
    "StripeClient",
    "GmailClient",
    "DropboxClient",
    "NotionClient",
    "TrelloClient",
]
