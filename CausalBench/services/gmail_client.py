"""
Gmail API Client for CausalBench

Uses google-api-python-client for real Gmail API interactions.
"""

import os
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

from .base_client import BaseServiceClient, APIResponse, register_service

logger = logging.getLogger(__name__)

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    GMAIL_AVAILABLE = True
except ImportError:
    GMAIL_AVAILABLE = False
    logger.warning("google-api-python-client not installed. Run: pip install google-api-python-client google-auth-oauthlib")


class GmailClient(BaseServiceClient):
    """
    Gmail API client for email operations.
    """

    SERVICE_NAME = "gmail"
    TRUST_LEVEL = "sensitive"  # Email is sensitive
    SCOPES = [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.send',
        'https://www.googleapis.com/auth/gmail.modify'
    ]

    def __init__(
        self,
        credentials_file: Optional[str] = None,
        token_file: Optional[str] = None,
        **kwargs
    ):
        # Gmail uses OAuth, not API key
        super().__init__(api_key=None, **kwargs)
        self.credentials_file = credentials_file or os.environ.get("GMAIL_CREDENTIALS_FILE", "credentials.json")
        self.token_file = token_file or os.environ.get("GMAIL_TOKEN_FILE", "token.json")
        self.service = None

    def _initialize_client(self) -> bool:
        if not GMAIL_AVAILABLE:
            logger.error("google-api-python-client not available")
            return False

        try:
            creds = None
            if os.path.exists(self.token_file):
                creds = Credentials.from_authorized_user_file(self.token_file, self.SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                elif os.path.exists(self.credentials_file):
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, self.SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                    # Save for future runs
                    with open(self.token_file, 'w') as token:
                        token.write(creds.to_json())
                else:
                    logger.error(f"Credentials file not found: {self.credentials_file}")
                    return False

            self.service = build('gmail', 'v1', credentials=creds)
            # Test connection
            self.service.users().getProfile(userId='me').execute()
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Gmail client: {e}")
            return False

    def list_items(self, limit: int = 10, item_type: str = "messages", **kwargs) -> APIResponse:
        """List messages, labels, or threads."""
        if not self.ensure_initialized():
            return APIResponse(success=False, error="Client not initialized")

        self.rate_limiter.wait_if_needed()

        try:
            if item_type == "messages":
                query = kwargs.get("query", "")
                result = self.service.users().messages().list(
                    userId='me',
                    maxResults=limit,
                    q=query
                ).execute()
                messages = result.get('messages', [])

                items = []
                for msg in messages[:limit]:
                    detail = self.service.users().messages().get(
                        userId='me',
                        id=msg['id'],
                        format='metadata',
                        metadataHeaders=['From', 'Subject', 'Date']
                    ).execute()
                    headers = {h['name']: h['value'] for h in detail.get('payload', {}).get('headers', [])}
                    items.append({
                        "id": msg['id'],
                        "thread_id": msg.get('threadId'),
                        "from": headers.get('From', ''),
                        "subject": headers.get('Subject', ''),
                        "date": headers.get('Date', ''),
                        "snippet": detail.get('snippet', '')[:200]
                    })

            elif item_type == "labels":
                result = self.service.users().labels().list(userId='me').execute()
                items = [
                    {
                        "id": label['id'],
                        "name": label['name'],
                        "type": label.get('type', 'user')
                    }
                    for label in result.get('labels', [])[:limit]
                ]

            elif item_type == "threads":
                result = self.service.users().threads().list(
                    userId='me',
                    maxResults=limit
                ).execute()
                items = [
                    {
                        "id": thread['id'],
                        "snippet": thread.get('snippet', '')[:200],
                        "history_id": thread.get('historyId')
                    }
                    for thread in result.get('threads', [])[:limit]
                ]

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

            return APIResponse(success=True, data={"items": items, "count": len(items)})

        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def get_item(self, item_id: str, item_type: str = "message", **kwargs) -> APIResponse:
        """Get a specific message, label, or thread."""
        if not self.ensure_initialized():
            return APIResponse(success=False, error="Client not initialized")

        self.rate_limiter.wait_if_needed()

        try:
            if item_type == "message":
                msg = self.service.users().messages().get(
                    userId='me',
                    id=item_id,
                    format='full'
                ).execute()
                headers = {h['name']: h['value'] for h in msg.get('payload', {}).get('headers', [])}

                # Extract body
                body = ""
                payload = msg.get('payload', {})
                if 'body' in payload and payload['body'].get('data'):
                    body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
                elif 'parts' in payload:
                    for part in payload['parts']:
                        if part.get('mimeType') == 'text/plain' and part.get('body', {}).get('data'):
                            body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                            break

                data = {
                    "id": msg['id'],
                    "thread_id": msg.get('threadId'),
                    "from": headers.get('From', ''),
                    "to": headers.get('To', ''),
                    "subject": headers.get('Subject', ''),
                    "date": headers.get('Date', ''),
                    "body": body[:5000],  # Truncate long bodies
                    "labels": msg.get('labelIds', [])
                }

            elif item_type == "thread":
                thread = self.service.users().threads().get(
                    userId='me',
                    id=item_id
                ).execute()
                messages = []
                for msg in thread.get('messages', []):
                    headers = {h['name']: h['value'] for h in msg.get('payload', {}).get('headers', [])}
                    messages.append({
                        "id": msg['id'],
                        "from": headers.get('From', ''),
                        "subject": headers.get('Subject', ''),
                        "snippet": msg.get('snippet', '')[:200]
                    })
                data = {
                    "id": thread['id'],
                    "messages": messages
                }

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

            return APIResponse(success=True, data=data)

        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def create_item(self, data: Dict[str, Any], item_type: str = "message", **kwargs) -> APIResponse:
        """Send an email or create a draft."""
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
                message = MIMEMultipart()
                message['to'] = data.get('to', '')
                message['subject'] = data.get('subject', '')
                message.attach(MIMEText(data.get('body', ''), 'plain'))

                raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
                result = self.service.users().messages().send(
                    userId='me',
                    body={'raw': raw}
                ).execute()
                return APIResponse(success=True, data={"id": result['id']})

            elif item_type == "draft":
                message = MIMEMultipart()
                message['to'] = data.get('to', '')
                message['subject'] = data.get('subject', '')
                message.attach(MIMEText(data.get('body', ''), 'plain'))

                raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
                result = self.service.users().drafts().create(
                    userId='me',
                    body={'message': {'raw': raw}}
                ).execute()
                return APIResponse(success=True, data={"id": result['id']})

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def forward_email(self, message_id: str, to: str) -> APIResponse:
        """Forward an email (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would forward {message_id} to {to}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "forward_email",
                "message_id": message_id,
                "to": to
            }
        )

    def create_filter(self, criteria: Dict, actions: Dict) -> APIResponse:
        """Create an email filter (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would create filter: {criteria} -> {actions}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "create_filter",
                "criteria": criteria,
                "actions": actions
            }
        )

    def delete_message(self, message_id: str) -> APIResponse:
        """Delete a message (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would delete message {message_id}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "delete_message",
                "message_id": message_id
            }
        )


# Register the client
register_service("gmail", GmailClient)
