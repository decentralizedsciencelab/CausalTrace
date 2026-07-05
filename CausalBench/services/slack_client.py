"""
Slack API Client for CausalBench

Uses slack-sdk for real Slack API interactions.
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .base_client import BaseServiceClient, APIResponse, register_service

logger = logging.getLogger(__name__)

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    logger.warning("slack-sdk not installed. Run: pip install slack-sdk")


class SlackClient(BaseServiceClient):
    """
    Slack API client for channel and message operations.
    """

    SERVICE_NAME = "slack"
    TRUST_LEVEL = "trusted"

    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs
    ):
        api_key = api_key or os.environ.get("SLACK_TOKEN")
        super().__init__(api_key=api_key, **kwargs)
        self.client = None

    def _initialize_client(self) -> bool:
        if not SLACK_AVAILABLE:
            logger.error("slack-sdk not available")
            return False
        if not self.api_key:
            logger.error("SLACK_TOKEN not set")
            return False
        try:
            self.client = WebClient(token=self.api_key)
            # Test connection
            result = self.client.auth_test()
            if result["ok"]:
                self.team_id = result.get("team_id")
                self.user_id = result.get("user_id")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Slack client: {e}")
            return False

    def list_items(self, limit: int = 10, item_type: str = "channels", **kwargs) -> APIResponse:
        """List channels, users, or messages."""
        if not self.ensure_initialized():
            return APIResponse(success=False, error="Client not initialized")

        self.rate_limiter.wait_if_needed()

        try:
            if item_type == "channels":
                result = self.client.conversations_list(limit=limit)
                items = [
                    {
                        "id": ch["id"],
                        "name": ch["name"],
                        "is_private": ch.get("is_private", False),
                        "is_member": ch.get("is_member", False),
                        "num_members": ch.get("num_members", 0),
                        "purpose": ch.get("purpose", {}).get("value", ""),
                        "created": ch.get("created")
                    }
                    for ch in result.get("channels", [])[:limit]
                ]

            elif item_type == "users":
                result = self.client.users_list(limit=limit)
                items = [
                    {
                        "id": u["id"],
                        "name": u.get("name"),
                        "real_name": u.get("real_name"),
                        "email": u.get("profile", {}).get("email"),
                        "is_admin": u.get("is_admin", False),
                        "is_bot": u.get("is_bot", False)
                    }
                    for u in result.get("members", [])[:limit]
                ]

            elif item_type == "messages":
                channel = kwargs.get("channel")
                if not channel:
                    return APIResponse(success=False, error="channel parameter required")
                result = self.client.conversations_history(channel=channel, limit=limit)
                items = [
                    {
                        "ts": m["ts"],
                        "text": m.get("text", "")[:500],
                        "user": m.get("user"),
                        "type": m.get("type"),
                        "subtype": m.get("subtype")
                    }
                    for m in result.get("messages", [])[:limit]
                ]

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

            return APIResponse(success=True, data={"items": items, "count": len(items)})

        except SlackApiError as e:
            return APIResponse(success=False, error=str(e), status_code=e.response.status_code)
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def get_item(self, item_id: str, item_type: str = "channel", **kwargs) -> APIResponse:
        """Get a specific channel, user, or message."""
        if not self.ensure_initialized():
            return APIResponse(success=False, error="Client not initialized")

        self.rate_limiter.wait_if_needed()

        try:
            if item_type == "channel":
                result = self.client.conversations_info(channel=item_id)
                ch = result["channel"]
                data = {
                    "id": ch["id"],
                    "name": ch["name"],
                    "is_private": ch.get("is_private", False),
                    "topic": ch.get("topic", {}).get("value", ""),
                    "purpose": ch.get("purpose", {}).get("value", ""),
                    "num_members": ch.get("num_members", 0)
                }

            elif item_type == "user":
                result = self.client.users_info(user=item_id)
                u = result["user"]
                data = {
                    "id": u["id"],
                    "name": u.get("name"),
                    "real_name": u.get("real_name"),
                    "email": u.get("profile", {}).get("email"),
                    "title": u.get("profile", {}).get("title"),
                    "is_admin": u.get("is_admin", False)
                }

            elif item_type == "message":
                channel = kwargs.get("channel")
                if not channel:
                    return APIResponse(success=False, error="channel parameter required")
                result = self.client.conversations_history(
                    channel=channel,
                    latest=item_id,
                    inclusive=True,
                    limit=1
                )
                messages = result.get("messages", [])
                if not messages:
                    return APIResponse(success=False, error="Message not found")
                m = messages[0]
                data = {
                    "ts": m["ts"],
                    "text": m.get("text", ""),
                    "user": m.get("user"),
                    "reactions": m.get("reactions", []),
                    "thread_ts": m.get("thread_ts")
                }

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

            return APIResponse(success=True, data=data)

        except SlackApiError as e:
            return APIResponse(success=False, error=str(e), status_code=e.response.status_code)
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def create_item(self, data: Dict[str, Any], item_type: str = "message", **kwargs) -> APIResponse:
        """Send a message or create a channel."""
        if not self.ensure_initialized():
            return APIResponse(success=False, error="Client not initialized")

        if self.simulate_writes:
            logger.info(f"[SIMULATED] Would create {item_type}: {data}")
            return APIResponse(
                success=True,
                data={"simulated": True, "item_type": item_type, "would_create": data}
            )

        self.rate_limiter.wait_if_needed()

        try:
            if item_type == "message":
                channel = data.get("channel") or kwargs.get("channel")
                if not channel:
                    return APIResponse(success=False, error="channel required")
                result = self.client.chat_postMessage(
                    channel=channel,
                    text=data.get("text", ""),
                    blocks=data.get("blocks")
                )
                return APIResponse(success=True, data={
                    "ts": result["ts"],
                    "channel": result["channel"]
                })

            elif item_type == "channel":
                result = self.client.conversations_create(
                    name=data.get("name"),
                    is_private=data.get("is_private", False)
                )
                return APIResponse(success=True, data={
                    "id": result["channel"]["id"],
                    "name": result["channel"]["name"]
                })

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

        except SlackApiError as e:
            return APIResponse(success=False, error=str(e), status_code=e.response.status_code)
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def invite_user(self, channel: str, user: str) -> APIResponse:
        """Invite a user to a channel (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would invite {user} to {channel}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "invite_user",
                "channel": channel,
                "user": user
            }
        )

    def send_dm(self, user: str, text: str) -> APIResponse:
        """Send a direct message (SENSITIVE for bulk/external)."""
        if self.simulate_writes:
            logger.warning(f"[SIMULATED] Would DM {user}: {text[:50]}...")
            return APIResponse(
                success=True,
                data={"simulated": True, "action": "send_dm", "user": user}
            )

        try:
            # Open DM channel
            result = self.client.conversations_open(users=[user])
            channel_id = result["channel"]["id"]

            # Send message
            result = self.client.chat_postMessage(channel=channel_id, text=text)
            return APIResponse(success=True, data={"ts": result["ts"]})

        except SlackApiError as e:
            return APIResponse(success=False, error=str(e))

    def upload_file(self, channels: List[str], content: str, filename: str) -> APIResponse:
        """Upload a file (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would upload {filename} to {channels}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "upload_file",
                "filename": filename,
                "channels": channels
            }
        )


# Register the client
register_service("slack", SlackClient)
