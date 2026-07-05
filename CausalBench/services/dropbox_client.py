"""
Dropbox API Client for CausalBench

Uses dropbox-sdk-python for real Dropbox API interactions.
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .base_client import BaseServiceClient, APIResponse, register_service

logger = logging.getLogger(__name__)

try:
    import dropbox
    from dropbox.exceptions import ApiError, AuthError
    DROPBOX_AVAILABLE = True
except ImportError:
    DROPBOX_AVAILABLE = False
    logger.warning("dropbox not installed. Run: pip install dropbox")


class DropboxClient(BaseServiceClient):
    """
    Dropbox API client for file and folder operations.
    """

    SERVICE_NAME = "dropbox"
    TRUST_LEVEL = "sensitive"  # File storage is sensitive

    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs
    ):
        api_key = api_key or os.environ.get("DROPBOX_ACCESS_TOKEN")
        super().__init__(api_key=api_key, **kwargs)
        self.client = None

    def _initialize_client(self) -> bool:
        if not DROPBOX_AVAILABLE:
            logger.error("dropbox not available")
            return False
        if not self.api_key:
            logger.error("DROPBOX_ACCESS_TOKEN not set")
            return False
        try:
            self.client = dropbox.Dropbox(self.api_key)
            # Test connection
            self.client.users_get_current_account()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Dropbox client: {e}")
            return False

    def list_items(self, limit: int = 10, item_type: str = "files", **kwargs) -> APIResponse:
        """List files, folders, or shared links."""
        if not self.ensure_initialized():
            return APIResponse(success=False, error="Client not initialized")

        self.rate_limiter.wait_if_needed()

        try:
            path = kwargs.get("path", "")

            if item_type in ["files", "folders"]:
                result = self.client.files_list_folder(path, limit=limit)
                items = []
                for entry in result.entries[:limit]:
                    item = {
                        "id": entry.id if hasattr(entry, 'id') else None,
                        "name": entry.name,
                        "path": entry.path_display,
                    }
                    if isinstance(entry, dropbox.files.FileMetadata):
                        item["type"] = "file"
                        item["size"] = entry.size
                        item["modified"] = entry.server_modified.isoformat() if entry.server_modified else None
                    else:
                        item["type"] = "folder"
                    items.append(item)

            elif item_type == "shared_links":
                result = self.client.sharing_list_shared_links()
                items = [
                    {
                        "id": link.id,
                        "url": link.url,
                        "name": link.name,
                        "path": link.path_lower,
                        "expires": link.expires.isoformat() if hasattr(link, 'expires') and link.expires else None
                    }
                    for link in result.links[:limit]
                ]

            elif item_type == "shared_folders":
                result = self.client.sharing_list_folders()
                items = [
                    {
                        "shared_folder_id": folder.shared_folder_id,
                        "name": folder.name,
                        "path": folder.path_lower,
                        "access_type": str(folder.access_type)
                    }
                    for folder in result.entries[:limit]
                ]

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

            return APIResponse(success=True, data={"items": items, "count": len(items)})

        except ApiError as e:
            return APIResponse(success=False, error=str(e), status_code=e.error.get_tag() if hasattr(e.error, 'get_tag') else 500)
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def get_item(self, item_id: str, item_type: str = "file", **kwargs) -> APIResponse:
        """Get file metadata or content."""
        if not self.ensure_initialized():
            return APIResponse(success=False, error="Client not initialized")

        self.rate_limiter.wait_if_needed()

        try:
            if item_type == "file":
                # item_id is the path for files
                metadata = self.client.files_get_metadata(item_id)
                data = {
                    "id": metadata.id if hasattr(metadata, 'id') else None,
                    "name": metadata.name,
                    "path": metadata.path_display,
                }
                if isinstance(metadata, dropbox.files.FileMetadata):
                    data["type"] = "file"
                    data["size"] = metadata.size
                    data["modified"] = metadata.server_modified.isoformat() if metadata.server_modified else None
                    data["content_hash"] = metadata.content_hash
                else:
                    data["type"] = "folder"

            elif item_type == "file_content":
                # Download file content (for small files)
                metadata, response = self.client.files_download(item_id)
                content = response.content
                # Truncate large files
                if len(content) > 10000:
                    content = content[:10000]
                    truncated = True
                else:
                    truncated = False
                data = {
                    "name": metadata.name,
                    "path": metadata.path_display,
                    "size": metadata.size,
                    "content": content.decode('utf-8', errors='ignore'),
                    "truncated": truncated
                }

            elif item_type == "shared_link":
                result = self.client.sharing_get_shared_link_metadata(item_id)
                data = {
                    "id": result.id,
                    "url": result.url,
                    "name": result.name,
                    "path": result.path_lower
                }

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

            return APIResponse(success=True, data=data)

        except ApiError as e:
            return APIResponse(success=False, error=str(e))
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def create_item(self, data: Dict[str, Any], item_type: str = "file", **kwargs) -> APIResponse:
        """Upload a file or create a folder."""
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
            if item_type == "file":
                path = data.get("path")
                content = data.get("content", "").encode('utf-8')
                result = self.client.files_upload(
                    content,
                    path,
                    mode=dropbox.files.WriteMode.overwrite
                )
                return APIResponse(success=True, data={
                    "id": result.id,
                    "path": result.path_display,
                    "size": result.size
                })

            elif item_type == "folder":
                path = data.get("path")
                result = self.client.files_create_folder_v2(path)
                return APIResponse(success=True, data={
                    "path": result.metadata.path_display
                })

            elif item_type == "shared_link":
                path = data.get("path")
                result = self.client.sharing_create_shared_link_with_settings(path)
                return APIResponse(success=True, data={
                    "url": result.url,
                    "path": result.path_lower
                })

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

        except ApiError as e:
            return APIResponse(success=False, error=str(e))
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def share_file(self, path: str, email: str, access_level: str = "viewer") -> APIResponse:
        """Share a file with someone (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would share {path} with {email} as {access_level}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "share_file",
                "path": path,
                "email": email,
                "access_level": access_level
            }
        )

    def create_public_link(self, path: str) -> APIResponse:
        """Create a public shared link (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would create public link for {path}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "create_public_link",
                "path": path
            }
        )

    def delete_file(self, path: str) -> APIResponse:
        """Delete a file (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would delete {path}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "delete_file",
                "path": path
            }
        )

    def move_file(self, from_path: str, to_path: str) -> APIResponse:
        """Move a file (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would move {from_path} to {to_path}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "move_file",
                "from_path": from_path,
                "to_path": to_path
            }
        )


# Register the client
register_service("dropbox", DropboxClient)
