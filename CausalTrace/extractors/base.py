"""
Base extractor interface for parsing agent trajectories.

This module defines the abstract base class for trajectory extractors.
All benchmark-specific extractors should inherit from BaseExtractor.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path

from causaltrace.models import Trajectory


class BaseExtractor(ABC):
    """
    Abstract base class for trajectory extractors.

    Extractors are responsible for parsing benchmark-specific log formats
    and converting them into the unified Trajectory representation.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the extractor.

        Args:
            verbose: Whether to print verbose output during extraction
        """
        self.verbose = verbose

    @abstractmethod
    def extract_from_log(self, log_path: str) -> Optional[Trajectory]:
        """
        Parse a single log file into a Trajectory.

        Args:
            log_path: Path to the log file

        Returns:
            Trajectory object or None if parsing fails

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement extract_from_log")

    @abstractmethod
    def extract_from_directory(self, dir_path: str, pattern: str = "*.json") -> List[Trajectory]:
        """
        Parse all log files in a directory.

        Args:
            dir_path: Path to the directory containing logs
            pattern: Glob pattern for matching log files (default: "*.json")

        Returns:
            List of Trajectory objects

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement extract_from_directory")

    def _log(self, message: str) -> None:
        """
        Print a log message if verbose mode is enabled.

        Args:
            message: Message to print
        """
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")

    def _validate_path(self, path: str) -> bool:
        """
        Validate that a path exists.

        Args:
            path: Path to validate

        Returns:
            True if path exists, False otherwise
        """
        return Path(path).exists()

    def save_trajectories(self, trajectories: List[Trajectory], output_dir: str) -> None:
        """
        Save a list of trajectories to JSON files.

        Args:
            trajectories: List of Trajectory objects
            output_dir: Directory to save trajectories to
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for trajectory in trajectories:
            filename = f"{trajectory.trajectory_id}.json"
            filepath = output_path / filename
            trajectory.to_json(str(filepath))
            self._log(f"Saved trajectory to {filepath}")

        self._log(f"Saved {len(trajectories)} trajectories to {output_dir}")
