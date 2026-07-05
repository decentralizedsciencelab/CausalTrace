"""
Trello API Client for CausalBench

Uses requests for Trello REST API interactions.
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import requests

from .base_client import BaseServiceClient, APIResponse, register_service

logger = logging.getLogger(__name__)


class TrelloClient(BaseServiceClient):
    """
    Trello API client for board, list, and card operations.
    """

    SERVICE_NAME = "trello"
    TRUST_LEVEL = "trusted"
    BASE_URL = "https://api.trello.com/1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_token: Optional[str] = None,
        **kwargs
    ):
        api_key = api_key or os.environ.get("TRELLO_API_KEY")
        self.api_token = api_token or os.environ.get("TRELLO_TOKEN")
        super().__init__(api_key=api_key, **kwargs)

    def _initialize_client(self) -> bool:
        if not self.api_key or not self.api_token:
            logger.error("TRELLO_API_KEY and TRELLO_TOKEN must be set")
            return False
        try:
            # Test connection
            response = self._make_api_request("GET", "/members/me")
            return response.get("id") is not None
        except Exception as e:
            logger.error(f"Failed to initialize Trello client: {e}")
            return False

    def _make_api_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make an authenticated API request."""
        url = f"{self.BASE_URL}{endpoint}"
        params = kwargs.get("params", {})
        params["key"] = self.api_key
        params["token"] = self.api_token

        response = requests.request(method, url, params=params, json=kwargs.get("json"))
        response.raise_for_status()
        return response.json()

    def list_items(self, limit: int = 10, item_type: str = "boards", **kwargs) -> APIResponse:
        """List boards, lists, or cards."""
        if not self.ensure_initialized():
            return APIResponse(success=False, error="Client not initialized")

        self.rate_limiter.wait_if_needed()

        try:
            if item_type == "boards":
                result = self._make_api_request("GET", "/members/me/boards")
                items = [
                    {
                        "id": board["id"],
                        "name": board["name"],
                        "url": board.get("url"),
                        "closed": board.get("closed", False),
                        "organization_id": board.get("idOrganization")
                    }
                    for board in result[:limit]
                ]

            elif item_type == "lists":
                board_id = kwargs.get("board_id")
                if not board_id:
                    return APIResponse(success=False, error="board_id parameter required")
                result = self._make_api_request("GET", f"/boards/{board_id}/lists")
                items = [
                    {
                        "id": lst["id"],
                        "name": lst["name"],
                        "closed": lst.get("closed", False),
                        "pos": lst.get("pos")
                    }
                    for lst in result[:limit]
                ]

            elif item_type == "cards":
                board_id = kwargs.get("board_id")
                list_id = kwargs.get("list_id")
                if list_id:
                    result = self._make_api_request("GET", f"/lists/{list_id}/cards")
                elif board_id:
                    result = self._make_api_request("GET", f"/boards/{board_id}/cards")
                else:
                    return APIResponse(success=False, error="board_id or list_id required")
                items = [
                    {
                        "id": card["id"],
                        "name": card["name"],
                        "desc": card.get("desc", "")[:500],
                        "url": card.get("url"),
                        "list_id": card.get("idList"),
                        "due": card.get("due"),
                        "labels": [l["name"] for l in card.get("labels", [])]
                    }
                    for card in result[:limit]
                ]

            elif item_type == "members":
                board_id = kwargs.get("board_id")
                if not board_id:
                    return APIResponse(success=False, error="board_id parameter required")
                result = self._make_api_request("GET", f"/boards/{board_id}/members")
                items = [
                    {
                        "id": member["id"],
                        "username": member.get("username"),
                        "full_name": member.get("fullName")
                    }
                    for member in result[:limit]
                ]

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

            return APIResponse(success=True, data={"items": items, "count": len(items)})

        except requests.HTTPError as e:
            return APIResponse(success=False, error=str(e), status_code=e.response.status_code)
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def get_item(self, item_id: str, item_type: str = "card", **kwargs) -> APIResponse:
        """Get a specific board, list, or card."""
        if not self.ensure_initialized():
            return APIResponse(success=False, error="Client not initialized")

        self.rate_limiter.wait_if_needed()

        try:
            if item_type == "board":
                board = self._make_api_request("GET", f"/boards/{item_id}")
                data = {
                    "id": board["id"],
                    "name": board["name"],
                    "desc": board.get("desc"),
                    "url": board.get("url"),
                    "closed": board.get("closed", False),
                    "prefs": board.get("prefs", {})
                }

            elif item_type == "list":
                lst = self._make_api_request("GET", f"/lists/{item_id}")
                data = {
                    "id": lst["id"],
                    "name": lst["name"],
                    "closed": lst.get("closed", False),
                    "board_id": lst.get("idBoard")
                }

            elif item_type == "card":
                card = self._make_api_request("GET", f"/cards/{item_id}")
                data = {
                    "id": card["id"],
                    "name": card["name"],
                    "desc": card.get("desc"),
                    "url": card.get("url"),
                    "list_id": card.get("idList"),
                    "board_id": card.get("idBoard"),
                    "due": card.get("due"),
                    "labels": [l["name"] for l in card.get("labels", [])],
                    "members": card.get("idMembers", []),
                    "attachments": card.get("badges", {}).get("attachments", 0)
                }

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

            return APIResponse(success=True, data=data)

        except requests.HTTPError as e:
            return APIResponse(success=False, error=str(e), status_code=e.response.status_code)
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def create_item(self, data: Dict[str, Any], item_type: str = "card", **kwargs) -> APIResponse:
        """Create a card, list, or board."""
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
            if item_type == "card":
                list_id = data.get("list_id") or kwargs.get("list_id")
                if not list_id:
                    return APIResponse(success=False, error="list_id required")
                result = self._make_api_request(
                    "POST", "/cards",
                    params={
                        "idList": list_id,
                        "name": data.get("name", "New Card"),
                        "desc": data.get("desc", "")
                    }
                )
                return APIResponse(success=True, data={
                    "id": result["id"],
                    "url": result.get("url")
                })

            elif item_type == "list":
                board_id = data.get("board_id") or kwargs.get("board_id")
                if not board_id:
                    return APIResponse(success=False, error="board_id required")
                result = self._make_api_request(
                    "POST", "/lists",
                    params={
                        "idBoard": board_id,
                        "name": data.get("name", "New List")
                    }
                )
                return APIResponse(success=True, data={"id": result["id"]})

            elif item_type == "comment":
                card_id = data.get("card_id") or kwargs.get("card_id")
                if not card_id:
                    return APIResponse(success=False, error="card_id required")
                result = self._make_api_request(
                    "POST", f"/cards/{card_id}/actions/comments",
                    params={"text": data.get("text", "")}
                )
                return APIResponse(success=True, data={"id": result["id"]})

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

        except requests.HTTPError as e:
            return APIResponse(success=False, error=str(e), status_code=e.response.status_code)
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def add_member(self, board_id: str, email: str) -> APIResponse:
        """Add a member to a board (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would add {email} to board {board_id}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "add_member",
                "board_id": board_id,
                "email": email
            }
        )

    def delete_card(self, card_id: str) -> APIResponse:
        """Delete a card (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would delete card {card_id}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "delete_card",
                "card_id": card_id
            }
        )

    def make_board_public(self, board_id: str) -> APIResponse:
        """Make a board public (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would make board {board_id} public")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "make_board_public",
                "board_id": board_id
            }
        )


# Register the client
register_service("trello", TrelloClient)
