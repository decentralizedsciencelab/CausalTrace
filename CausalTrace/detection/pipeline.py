"""
End-to-end attack detection pipeline.

Provides a simple interface to detect attacks from trajectory files,
handling all steps from extraction to graph building to detection.
"""

from typing import Optional, Union, Any
from pathlib import Path
import json

from .detector import BaseDetector, DetectionResult
from ..graph import GraphBuilder, CausalGraph


class AttackDetectionPipeline:
    """
    Complete pipeline: trajectory to detection.

    This class provides a simple interface for end-to-end attack detection:
    1. Extract trajectory from log file
    2. Build causal graph
    3. Detect attacks using trained detector

    Example:
        ```python
        from causaltrace.detection import MLDetector, AttackDetectionPipeline

        # Create and train detector
        detector = MLDetector()
        detector.fit(training_graphs, training_labels)

        # Create pipeline
        pipeline = AttackDetectionPipeline(detector)

        # Detect from file
        result = pipeline.detect_from_trajectory_file(
            "path/to/trajectory.json",
            source="wasp"
        )
        print(result)
        ```
    """

    def __init__(self, detector: BaseDetector, graph_builder: Optional[GraphBuilder] = None):
        """
        Initialize pipeline.

        Args:
            detector: Trained detector
            graph_builder: Optional graph builder (creates default if not provided)
        """
        self.detector = detector
        self.graph_builder = graph_builder or GraphBuilder()
        self.extractor = None  # Set dynamically based on source

    def detect_from_trajectory_file(
        self,
        file_path: str,
        source: str = "auto"
    ) -> DetectionResult:
        """
        Detect attack from trajectory log file.

        Args:
            file_path: Path to trajectory log file
            source: "wasp", "safearena", or "auto" (auto-detect from file structure)

        Returns:
            DetectionResult

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If source is invalid or cannot be auto-detected
        """
        # Validate file exists
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {file_path}")

        # Auto-detect source if needed
        if source == "auto":
            source = self._detect_source(file_path)
            print(f"Auto-detected source: {source}")

        # Load extractor for source
        self._load_extractor(source)

        # Extract trajectory
        print(f"Extracting trajectory from {file_path}...")
        trajectory = self.extractor.extract_from_log(file_path)

        # Build causal graph
        print("Building causal graph...")
        graph = self.graph_builder.build(trajectory)

        # Detect
        print("Running detection...")
        result = self.detector.predict(graph)

        return result

    def detect_from_trajectory(self, trajectory: Any) -> DetectionResult:
        """
        Detect attack from a trajectory object.

        Args:
            trajectory: Trajectory object (format depends on extraction agent)

        Returns:
            DetectionResult
        """
        # Build causal graph
        graph = self.graph_builder.build(trajectory)

        # Detect
        result = self.detector.predict(graph)

        return result

    def detect_from_graph(self, graph: CausalGraph) -> DetectionResult:
        """
        Detect attack from a causal graph.

        Args:
            graph: CausalGraph object

        Returns:
            DetectionResult
        """
        return self.detector.predict(graph)

    def detect_batch_from_directory(
        self,
        directory: str,
        source: str = "auto",
        file_pattern: str = "*.json",
        max_files: Optional[int] = None
    ) -> list[tuple[str, DetectionResult]]:
        """
        Detect attacks from all trajectory files in a directory.

        Args:
            directory: Directory containing trajectory files
            source: Source type
            file_pattern: Glob pattern for files (e.g., "*.json")
            max_files: Optional limit on number of files to process

        Returns:
            List of (filename, DetectionResult) tuples
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find trajectory files
        files = sorted(dir_path.glob(file_pattern))
        if max_files:
            files = files[:max_files]

        print(f"Processing {len(files)} trajectory files from {directory}...")

        results = []
        for i, file_path in enumerate(files):
            print(f"  [{i+1}/{len(files)}] {file_path.name}")

            try:
                result = self.detect_from_trajectory_file(str(file_path), source)
                results.append((file_path.name, result))
            except Exception as e:
                print(f"    Error: {e}")
                continue

        # Summary
        num_attacks = sum(1 for _, r in results if r.is_attack)
        print(f"\nSummary: {num_attacks}/{len(results)} trajectories flagged as attacks")

        return results

    def _detect_source(self, file_path: str) -> str:
        """
        Auto-detect the source type from file path or content.

        Args:
            file_path: Path to trajectory file

        Returns:
            Source type ("wasp" or "safearena")

        Raises:
            ValueError: If source cannot be determined
        """
        path = Path(file_path)

        # Check path for hints
        path_str = str(path).lower()
        if 'wasp' in path_str or 'webarena' in path_str:
            return 'wasp'
        elif 'safearena' in path_str or 'safe_arena' in path_str:
            return 'safearena'

        # Try to detect from file content
        try:
            with open(file_path, 'r') as f:
                content = f.read(1000)  # Read first 1000 chars

                if 'webarena' in content.lower() or 'wasp' in content.lower():
                    return 'wasp'
                elif 'safearena' in content.lower():
                    return 'safearena'
        except Exception:
            pass

        raise ValueError(
            f"Cannot auto-detect source for {file_path}. "
            "Please specify source explicitly ('wasp' or 'safearena')"
        )

    def _load_extractor(self, source: str) -> None:
        """
        Load the appropriate extractor for the source.


        Args:
            source: Source type

        Raises:
            ValueError: If source is not supported
        """
        if source == 'wasp':
            # Stub - Extraction Agent will implement
            from ..extractors.wasp import WASPExtractor
            self.extractor = WASPExtractor()
        elif source == 'safearena':
            # Stub - Extraction Agent will implement
            from ..extractors.safearena import SafeArenaExtractor
            self.extractor = SafeArenaExtractor()
        else:
            raise ValueError(f"Unsupported source: {source}")

    def save_results(
        self,
        results: list[tuple[str, DetectionResult]],
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        Save detection results to file.

        Args:
            results: List of (filename, DetectionResult) tuples
            output_path: Path to output file
            format: Output format ("json" or "csv")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            self._save_json(results, output_path)
        elif format == "csv":
            self._save_csv(results, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Results saved to {output_path}")

    def _save_json(self, results: list[tuple[str, DetectionResult]], path: Path) -> None:
        """Save results as JSON."""
        data = {
            'num_results': len(results),
            'num_attacks': sum(1 for _, r in results if r.is_attack),
            'results': [
                {
                    'file': filename,
                    **result.to_dict()
                }
                for filename, result in results
            ]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_csv(self, results: list[tuple[str, DetectionResult]], path: Path) -> None:
        """Save results as CSV."""
        import csv

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['file', 'is_attack', 'confidence', 'explanation'])

            # Rows
            for filename, result in results:
                writer.writerow([
                    filename,
                    result.is_attack,
                    f"{result.confidence:.4f}",
                    result.explanation
                ])

    def __repr__(self) -> str:
        """String representation."""
        return f"AttackDetectionPipeline(detector={self.detector.__class__.__name__})"


__all__ = ['AttackDetectionPipeline']
