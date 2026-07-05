"""
GitHub API Client for CausalBench

Uses PyGithub for real GitHub API interactions.
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .base_client import BaseServiceClient, APIResponse, register_service

logger = logging.getLogger(__name__)

try:
    from github import Github, GithubException
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    logger.warning("PyGithub not installed. Run: pip install PyGithub")


class GitHubClient(BaseServiceClient):
    """
    GitHub API client for repository and issue operations.
    """

    SERVICE_NAME = "github"
    TRUST_LEVEL = "trusted"

    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs
    ):
        api_key = api_key or os.environ.get("GITHUB_TOKEN")
        super().__init__(api_key=api_key, **kwargs)
        self.client = None

    def _initialize_client(self) -> bool:
        if not GITHUB_AVAILABLE:
            logger.error("PyGithub not available")
            return False
        if not self.api_key:
            logger.error("GITHUB_TOKEN not set")
            return False
        try:
            self.client = Github(self.api_key)
            # Test connection
            _ = self.client.get_user().login
            return True
        except Exception as e:
            logger.error(f"Failed to initialize GitHub client: {e}")
            return False

    def list_items(self, limit: int = 10, item_type: str = "repos", **kwargs) -> APIResponse:
        """List repositories, issues, or other items."""
        if not self.ensure_initialized():
            return APIResponse(success=False, error="Client not initialized")

        self.rate_limiter.wait_if_needed()

        try:
            if item_type == "repos":
                user = self.client.get_user()
                repos = list(user.get_repos()[:limit])
                items = [
                    {
                        "id": repo.id,
                        "name": repo.name,
                        "full_name": repo.full_name,
                        "description": repo.description,
                        "private": repo.private,
                        "url": repo.html_url,
                        "created_at": repo.created_at.isoformat() if repo.created_at else None
                    }
                    for repo in repos
                ]
            elif item_type == "issues":
                repo_name = kwargs.get("repo")
                if not repo_name:
                    return APIResponse(success=False, error="repo parameter required")
                repo = self.client.get_repo(repo_name)
                issues = list(repo.get_issues(state="open")[:limit])
                items = [
                    {
                        "id": issue.id,
                        "number": issue.number,
                        "title": issue.title,
                        "body": issue.body[:500] if issue.body else None,
                        "state": issue.state,
                        "user": issue.user.login if issue.user else None,
                        "created_at": issue.created_at.isoformat() if issue.created_at else None
                    }
                    for issue in issues
                ]
            elif item_type == "collaborators":
                repo_name = kwargs.get("repo")
                if not repo_name:
                    return APIResponse(success=False, error="repo parameter required")
                repo = self.client.get_repo(repo_name)
                collabs = list(repo.get_collaborators()[:limit])
                items = [
                    {
                        "id": c.id,
                        "login": c.login,
                        "type": c.type,
                        "permissions": c.permissions
                    }
                    for c in collabs
                ]
            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

            return APIResponse(success=True, data={"items": items, "count": len(items)})

        except GithubException as e:
            return APIResponse(success=False, error=str(e), status_code=e.status)
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def get_item(self, item_id: str, item_type: str = "repo", **kwargs) -> APIResponse:
        """Get a specific repository, issue, or other item."""
        if not self.ensure_initialized():
            return APIResponse(success=False, error="Client not initialized")

        self.rate_limiter.wait_if_needed()

        try:
            if item_type == "repo":
                repo = self.client.get_repo(item_id)
                data = {
                    "id": repo.id,
                    "name": repo.name,
                    "full_name": repo.full_name,
                    "description": repo.description,
                    "private": repo.private,
                    "url": repo.html_url,
                    "default_branch": repo.default_branch,
                    "stargazers_count": repo.stargazers_count,
                    "forks_count": repo.forks_count,
                    "open_issues_count": repo.open_issues_count
                }
            elif item_type == "issue":
                repo_name = kwargs.get("repo")
                if not repo_name:
                    return APIResponse(success=False, error="repo parameter required")
                repo = self.client.get_repo(repo_name)
                issue = repo.get_issue(int(item_id))
                data = {
                    "id": issue.id,
                    "number": issue.number,
                    "title": issue.title,
                    "body": issue.body,
                    "state": issue.state,
                    "user": issue.user.login if issue.user else None,
                    "labels": [l.name for l in issue.labels],
                    "comments": issue.comments,
                    "created_at": issue.created_at.isoformat() if issue.created_at else None
                }
            elif item_type == "user":
                user = self.client.get_user(item_id)
                data = {
                    "id": user.id,
                    "login": user.login,
                    "name": user.name,
                    "email": user.email,
                    "bio": user.bio,
                    "public_repos": user.public_repos,
                    "followers": user.followers
                }
            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

            return APIResponse(success=True, data=data)

        except GithubException as e:
            return APIResponse(success=False, error=str(e), status_code=e.status)
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def create_item(self, data: Dict[str, Any], item_type: str = "issue", **kwargs) -> APIResponse:
        """Create an issue, gist, or comment."""
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
            if item_type == "issue":
                repo_name = kwargs.get("repo")
                if not repo_name:
                    return APIResponse(success=False, error="repo parameter required")
                repo = self.client.get_repo(repo_name)
                issue = repo.create_issue(
                    title=data.get("title", "New Issue"),
                    body=data.get("body", "")
                )
                return APIResponse(success=True, data={
                    "id": issue.id,
                    "number": issue.number,
                    "url": issue.html_url
                })

            elif item_type == "comment":
                repo_name = kwargs.get("repo")
                issue_number = kwargs.get("issue_number")
                if not repo_name or not issue_number:
                    return APIResponse(success=False, error="repo and issue_number required")
                repo = self.client.get_repo(repo_name)
                issue = repo.get_issue(int(issue_number))
                comment = issue.create_comment(data.get("body", ""))
                return APIResponse(success=True, data={
                    "id": comment.id,
                    "url": comment.html_url
                })

            elif item_type == "gist":
                user = self.client.get_user()
                gist = user.create_gist(
                    public=data.get("public", False),
                    files=data.get("files", {}),
                    description=data.get("description", "")
                )
                return APIResponse(success=True, data={
                    "id": gist.id,
                    "url": gist.html_url
                })

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

        except GithubException as e:
            return APIResponse(success=False, error=str(e), status_code=e.status)
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def add_collaborator(self, repo: str, username: str, permission: str = "pull") -> APIResponse:
        """Add a collaborator to a repository (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would add {username} to {repo} with {permission}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "add_collaborator",
                "repo": repo,
                "username": username,
                "permission": permission
            }
        )

    def create_deploy_key(self, repo: str, title: str, key: str) -> APIResponse:
        """Create a deploy key (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would add deploy key to {repo}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "create_deploy_key",
                "repo": repo,
                "title": title
            }
        )

    def create_webhook(self, repo: str, url: str, events: List[str]) -> APIResponse:
        """Create a webhook (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would create webhook for {repo} pointing to {url}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "create_webhook",
                "repo": repo,
                "url": url,
                "events": events
            }
        )


# Register the client
register_service("github", GitHubClient)
