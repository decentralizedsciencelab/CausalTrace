"""
Notion API Client for CausalBench

Uses notion-client for real Notion API interactions.
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .base_client import BaseServiceClient, APIResponse, register_service

logger = logging.getLogger(__name__)

try:
    from notion_client import Client as NotionAPIClient
    from notion_client.errors import APIResponseError
    NOTION_AVAILABLE = True
except ImportError:
    NOTION_AVAILABLE = False
    logger.warning("notion-client not installed. Run: pip install notion-client")


class NotionClient(BaseServiceClient):
    """
    Notion API client for page and database operations.
    """

    SERVICE_NAME = "notion"
    TRUST_LEVEL = "trusted"

    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs
    ):
        api_key = api_key or os.environ.get("NOTION_TOKEN")
        super().__init__(api_key=api_key, **kwargs)
        self.client = None

    def _initialize_client(self) -> bool:
        if not NOTION_AVAILABLE:
            logger.error("notion-client not available")
            return False
        if not self.api_key:
            logger.error("NOTION_TOKEN not set")
            return False
        try:
            self.client = NotionAPIClient(auth=self.api_key)
            # Test connection
            self.client.users.me()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Notion client: {e}")
            return False

    def list_items(self, limit: int = 10, item_type: str = "pages", **kwargs) -> APIResponse:
        """List pages, databases, or users."""
        if not self.ensure_initialized():
            return APIResponse(success=False, error="Client not initialized")

        self.rate_limiter.wait_if_needed()

        try:
            if item_type == "pages":
                # Search for pages
                result = self.client.search(
                    filter={"property": "object", "value": "page"},
                    page_size=limit
                )
                items = [
                    {
                        "id": page["id"],
                        "title": self._extract_title(page),
                        "url": page.get("url"),
                        "created_time": page.get("created_time"),
                        "last_edited_time": page.get("last_edited_time"),
                        "parent_type": page.get("parent", {}).get("type")
                    }
                    for page in result.get("results", [])[:limit]
                ]

            elif item_type == "databases":
                result = self.client.search(
                    filter={"property": "object", "value": "database"},
                    page_size=limit
                )
                items = [
                    {
                        "id": db["id"],
                        "title": self._extract_db_title(db),
                        "url": db.get("url"),
                        "created_time": db.get("created_time")
                    }
                    for db in result.get("results", [])[:limit]
                ]

            elif item_type == "users":
                result = self.client.users.list(page_size=limit)
                items = [
                    {
                        "id": user["id"],
                        "name": user.get("name"),
                        "type": user.get("type"),
                        "email": user.get("person", {}).get("email") if user.get("type") == "person" else None
                    }
                    for user in result.get("results", [])[:limit]
                ]

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

            return APIResponse(success=True, data={"items": items, "count": len(items)})

        except APIResponseError as e:
            return APIResponse(success=False, error=str(e), status_code=e.status)
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def get_item(self, item_id: str, item_type: str = "page", **kwargs) -> APIResponse:
        """Get a specific page or database."""
        if not self.ensure_initialized():
            return APIResponse(success=False, error="Client not initialized")

        self.rate_limiter.wait_if_needed()

        try:
            if item_type == "page":
                page = self.client.pages.retrieve(page_id=item_id)
                # Get page content
                blocks = self.client.blocks.children.list(block_id=item_id, page_size=50)
                content = self._extract_blocks_text(blocks.get("results", []))

                data = {
                    "id": page["id"],
                    "title": self._extract_title(page),
                    "url": page.get("url"),
                    "created_time": page.get("created_time"),
                    "last_edited_time": page.get("last_edited_time"),
                    "content": content[:5000]  # Truncate
                }

            elif item_type == "database":
                db = self.client.databases.retrieve(database_id=item_id)
                # Get database items
                items_result = self.client.databases.query(database_id=item_id, page_size=20)
                items = [
                    {
                        "id": item["id"],
                        "properties": self._simplify_properties(item.get("properties", {}))
                    }
                    for item in items_result.get("results", [])
                ]

                data = {
                    "id": db["id"],
                    "title": self._extract_db_title(db),
                    "url": db.get("url"),
                    "properties": list(db.get("properties", {}).keys()),
                    "items": items
                }

            elif item_type == "block":
                block = self.client.blocks.retrieve(block_id=item_id)
                data = {
                    "id": block["id"],
                    "type": block.get("type"),
                    "content": self._extract_block_text(block)
                }

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

            return APIResponse(success=True, data=data)

        except APIResponseError as e:
            return APIResponse(success=False, error=str(e), status_code=e.status)
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def create_item(self, data: Dict[str, Any], item_type: str = "page", **kwargs) -> APIResponse:
        """Create a page or database entry."""
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
            if item_type == "page":
                parent_id = data.get("parent_id")
                title = data.get("title", "Untitled")
                content = data.get("content", "")

                page = self.client.pages.create(
                    parent={"page_id": parent_id} if parent_id else {"workspace": True},
                    properties={
                        "title": {"title": [{"text": {"content": title}}]}
                    },
                    children=[
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{"type": "text", "text": {"content": content}}]
                            }
                        }
                    ] if content else []
                )
                return APIResponse(success=True, data={"id": page["id"], "url": page.get("url")})

            elif item_type == "database_entry":
                database_id = data.get("database_id")
                properties = data.get("properties", {})

                page = self.client.pages.create(
                    parent={"database_id": database_id},
                    properties=properties
                )
                return APIResponse(success=True, data={"id": page["id"]})

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

        except APIResponseError as e:
            return APIResponse(success=False, error=str(e), status_code=e.status)
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def share_page(self, page_id: str, email: str, permission: str = "read") -> APIResponse:
        """Share a page (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would share page {page_id} with {email}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "share_page",
                "page_id": page_id,
                "email": email,
                "permission": permission
            }
        )

    def delete_page(self, page_id: str) -> APIResponse:
        """Delete/archive a page (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would delete page {page_id}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "delete_page",
                "page_id": page_id
            }
        )

    # Helper methods
    def _extract_title(self, page: Dict) -> str:
        """Extract title from page properties."""
        props = page.get("properties", {})
        for key, value in props.items():
            if value.get("type") == "title":
                title_list = value.get("title", [])
                if title_list:
                    return title_list[0].get("plain_text", "Untitled")
        return "Untitled"

    def _extract_db_title(self, db: Dict) -> str:
        """Extract title from database."""
        title_list = db.get("title", [])
        if title_list:
            return title_list[0].get("plain_text", "Untitled")
        return "Untitled"

    def _extract_block_text(self, block: Dict) -> str:
        """Extract text from a block."""
        block_type = block.get("type")
        if block_type and block_type in block:
            rich_text = block[block_type].get("rich_text", [])
            return "".join([t.get("plain_text", "") for t in rich_text])
        return ""

    def _extract_blocks_text(self, blocks: List[Dict]) -> str:
        """Extract text from multiple blocks."""
        texts = []
        for block in blocks:
            text = self._extract_block_text(block)
            if text:
                texts.append(text)
        return "\n".join(texts)

    def _simplify_properties(self, properties: Dict) -> Dict:
        """Simplify Notion properties for output."""
        result = {}
        for key, value in properties.items():
            prop_type = value.get("type")
            if prop_type == "title":
                title_list = value.get("title", [])
                result[key] = title_list[0].get("plain_text", "") if title_list else ""
            elif prop_type == "rich_text":
                text_list = value.get("rich_text", [])
                result[key] = text_list[0].get("plain_text", "") if text_list else ""
            elif prop_type == "number":
                result[key] = value.get("number")
            elif prop_type == "select":
                select = value.get("select")
                result[key] = select.get("name") if select else None
            elif prop_type == "checkbox":
                result[key] = value.get("checkbox")
            else:
                result[key] = f"<{prop_type}>"
        return result


# Register the client
register_service("notion", NotionClient)
