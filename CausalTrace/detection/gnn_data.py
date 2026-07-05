"""
GNN Data Module for CausalTrace.

Provides utilities for converting CausalGraph objects to PyTorch Geometric
Data objects and creating datasets/dataloaders for GNN training.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from pathlib import Path
import json
import warnings

try:
    import torch
    from torch_geometric.data import Data, Dataset, InMemoryDataset
    from torch_geometric.loader import DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch Geometric not available. Install with: "
        "pip install torch torch_geometric"
    )

from ..graph import CausalGraph
from .gnn_detector import causal_graph_to_pyg, get_node_feature_dim, get_edge_feature_dim


class CausalGraphDataset:
    """
    Dataset wrapper for CausalGraph objects.

    Provides utilities for:
    - Converting graphs to PyG format
    - Creating train/val/test splits
    - Creating DataLoaders
    """

    def __init__(
        self,
        graphs: List[Union[CausalGraph, Dict[str, Any]]],
        labels: List[bool],
        transform: Optional[Callable] = None,
    ):
        """
        Initialize dataset from graphs and labels.

        Args:
            graphs: List of CausalGraph objects or graph dictionaries
            labels: List of labels (True=attack, False=benign)
            transform: Optional transform function to apply to each Data object
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch Geometric required for CausalGraphDataset")

        if len(graphs) != len(labels):
            raise ValueError(f"Mismatch: {len(graphs)} graphs vs {len(labels)} labels")

        self.graphs = graphs
        self.labels = labels
        self.transform = transform
        self._pyg_data = None
        self._conversion_errors = []

    def __len__(self) -> int:
        """Number of samples in dataset."""
        return len(self.graphs)

    def __getitem__(self, idx: int) -> "Data":
        """Get a single sample."""
        if self._pyg_data is None:
            self._convert_all()
        return self._pyg_data[idx]

    def _convert_all(self) -> None:
        """Convert all graphs to PyG format."""
        self._pyg_data = []
        self._conversion_errors = []

        for i, (graph, label) in enumerate(zip(self.graphs, self.labels)):
            try:
                data = causal_graph_to_pyg(graph, label)
                if self.transform is not None:
                    data = self.transform(data)
                self._pyg_data.append(data)
                self._conversion_errors.append(None)
            except Exception as e:
                # Create empty graph for failed conversions
                data = Data(
                    x=torch.zeros((1, get_node_feature_dim()), dtype=torch.float),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, get_edge_feature_dim()), dtype=torch.float),
                    y=torch.tensor([int(label)], dtype=torch.long),
                )
                self._pyg_data.append(data)
                self._conversion_errors.append(str(e))

    def get_pyg_list(self) -> List["Data"]:
        """Get all graphs as PyG Data objects."""
        if self._pyg_data is None:
            self._convert_all()
        return self._pyg_data

    def get_conversion_errors(self) -> List[Optional[str]]:
        """Get list of conversion errors (None for successful conversions)."""
        if self._pyg_data is None:
            self._convert_all()
        return self._conversion_errors

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if self._pyg_data is None:
            self._convert_all()

        num_attacks = sum(self.labels)
        num_benign = len(self.labels) - num_attacks
        num_errors = sum(1 for e in self._conversion_errors if e is not None)

        node_counts = [d.x.shape[0] for d in self._pyg_data]
        edge_counts = [d.edge_index.shape[1] for d in self._pyg_data]

        return {
            "num_samples": len(self.graphs),
            "num_attacks": num_attacks,
            "num_benign": num_benign,
            "num_conversion_errors": num_errors,
            "node_feature_dim": get_node_feature_dim(),
            "edge_feature_dim": get_edge_feature_dim(),
            "avg_nodes": sum(node_counts) / len(node_counts) if node_counts else 0,
            "min_nodes": min(node_counts) if node_counts else 0,
            "max_nodes": max(node_counts) if node_counts else 0,
            "avg_edges": sum(edge_counts) / len(edge_counts) if edge_counts else 0,
            "min_edges": min(edge_counts) if edge_counts else 0,
            "max_edges": max(edge_counts) if edge_counts else 0,
        }

    def create_loader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        **kwargs,
    ) -> "DataLoader":
        """
        Create a DataLoader for this dataset.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            **kwargs: Additional arguments passed to DataLoader

        Returns:
            PyG DataLoader
        """
        pyg_data = self.get_pyg_list()
        return DataLoader(pyg_data, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = True,
        random_state: int = 42,
    ) -> Tuple["CausalGraphDataset", "CausalGraphDataset", "CausalGraphDataset"]:
        """
        Split dataset into train/val/test sets.

        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            stratify: Whether to stratify by label
            random_state: Random seed

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        from sklearn.model_selection import train_test_split

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Ratios must sum to 1.0")

        # First split: train vs (val + test)
        indices = list(range(len(self.graphs)))
        stratify_labels = self.labels if stratify else None

        # Handle edge case where test_ratio is 0
        if test_ratio < 0.001:  # Essentially 0
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_ratio,
                random_state=random_state,
                stratify=stratify_labels,
            )
            test_idx = []
        elif val_ratio < 0.001:  # Essentially 0
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_ratio,
                random_state=random_state,
                stratify=stratify_labels,
            )
            val_idx = []
        else:
            train_idx, temp_idx = train_test_split(
                indices,
                test_size=(val_ratio + test_ratio),
                random_state=random_state,
                stratify=stratify_labels,
            )

            # Second split: val vs test
            temp_labels = [self.labels[i] for i in temp_idx] if stratify else None
            relative_test_ratio = test_ratio / (val_ratio + test_ratio)

            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=relative_test_ratio,
                random_state=random_state,
                stratify=temp_labels,
            )

        # Create datasets
        train_graphs = [self.graphs[i] for i in train_idx]
        train_labels = [self.labels[i] for i in train_idx]

        val_graphs = [self.graphs[i] for i in val_idx]
        val_labels = [self.labels[i] for i in val_idx]

        test_graphs = [self.graphs[i] for i in test_idx]
        test_labels = [self.labels[i] for i in test_idx]

        return (
            CausalGraphDataset(train_graphs, train_labels, self.transform),
            CausalGraphDataset(val_graphs, val_labels, self.transform),
            CausalGraphDataset(test_graphs, test_labels, self.transform),
        )

    @classmethod
    def from_json_files(
        cls,
        graph_files: List[str],
        label_file: Optional[str] = None,
        labels: Optional[List[bool]] = None,
    ) -> "CausalGraphDataset":
        """
        Load dataset from JSON files.

        Args:
            graph_files: List of paths to graph JSON files
            label_file: Optional path to JSON file with labels
            labels: Optional list of labels (alternative to label_file)

        Returns:
            CausalGraphDataset instance
        """
        graphs = []
        for f in graph_files:
            with open(f, "r") as fp:
                graphs.append(json.load(fp))

        if label_file is not None:
            with open(label_file, "r") as fp:
                labels_data = json.load(fp)
                labels = labels_data if isinstance(labels_data, list) else list(labels_data.values())
        elif labels is None:
            # Default: assume all benign if no labels provided
            labels = [False] * len(graphs)

        return cls(graphs, labels)

    @classmethod
    def from_directory(
        cls,
        graph_dir: str,
        label_file: Optional[str] = None,
        pattern: str = "*.json",
    ) -> "CausalGraphDataset":
        """
        Load dataset from a directory of graph JSON files.

        Args:
            graph_dir: Directory containing graph JSON files
            label_file: Optional path to labels JSON file
            pattern: Glob pattern for graph files

        Returns:
            CausalGraphDataset instance
        """
        graph_dir = Path(graph_dir)
        graph_files = sorted(graph_dir.glob(pattern))

        if not graph_files:
            raise ValueError(f"No files matching '{pattern}' in {graph_dir}")

        return cls.from_json_files([str(f) for f in graph_files], label_file)


# Only define InMemoryCausalGraphDataset if PyTorch is available
if TORCH_AVAILABLE:
    class InMemoryCausalGraphDataset(InMemoryDataset):
        """
        PyTorch Geometric InMemoryDataset for CausalGraphs.

        This class follows PyG conventions and can be used with standard
        PyG utilities for transforms, splits, etc.
        """

        def __init__(
            self,
            root: str,
            graphs: Optional[List[Union[CausalGraph, Dict[str, Any]]]] = None,
            labels: Optional[List[bool]] = None,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
        ):
            """
            Initialize InMemory dataset.

            Args:
                root: Root directory for processed data
                graphs: Optional list of graphs (if None, loads from processed)
                labels: Optional list of labels
                transform: Transform applied at access time
                pre_transform: Transform applied during processing
                pre_filter: Filter applied during processing
            """
            self._graphs = graphs
            self._labels = labels

            super().__init__(root, transform, pre_transform, pre_filter)
            self.load(self.processed_paths[0])

        @property
        def raw_file_names(self) -> List[str]:
            """Raw file names (not used, data provided directly)."""
            return []

        @property
        def processed_file_names(self) -> List[str]:
            """Processed file names."""
            return ["data.pt"]

        def download(self):
            """Download method (not needed, data provided directly)."""
            pass

        def process(self):
            """Process graphs into PyG Data objects."""
            if self._graphs is None or self._labels is None:
                # If no new data, just load existing
                return

            data_list = []
            for graph, label in zip(self._graphs, self._labels):
                try:
                    data = causal_graph_to_pyg(graph, label)
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)
                except Exception:
                    continue

            self.save(data_list, self.processed_paths[0])
else:
    # Placeholder when PyTorch is not available
    InMemoryCausalGraphDataset = None


def create_data_loaders(
    graphs: List[Union[CausalGraph, Dict[str, Any]]],
    labels: List[bool],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    stratify: bool = True,
    random_state: int = 42,
) -> Tuple["DataLoader", "DataLoader", "DataLoader"]:
    """
    Convenience function to create train/val/test DataLoaders.

    Args:
        graphs: List of graphs
        labels: List of labels
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        batch_size: Batch size for all loaders
        stratify: Whether to stratify splits
        random_state: Random seed

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch Geometric required")

    dataset = CausalGraphDataset(graphs, labels)
    train_ds, val_ds, test_ds = dataset.split(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify=stratify,
        random_state=random_state,
    )

    train_loader = train_ds.create_loader(batch_size=batch_size, shuffle=True)
    val_loader = val_ds.create_loader(batch_size=batch_size, shuffle=False)
    test_loader = test_ds.create_loader(batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def collate_graphs(
    graphs: List[Union[CausalGraph, Dict[str, Any]]],
    labels: Optional[List[bool]] = None,
) -> "Data":
    """
    Collate multiple graphs into a single batched graph.

    Args:
        graphs: List of graphs to batch
        labels: Optional labels for each graph

    Returns:
        Batched PyG Data object
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch Geometric required")

    from torch_geometric.data import Batch

    data_list = []
    for i, graph in enumerate(graphs):
        label = labels[i] if labels is not None else None
        try:
            data = causal_graph_to_pyg(graph, label)
            data_list.append(data)
        except Exception:
            continue

    if not data_list:
        raise ValueError("No valid graphs to collate")

    return Batch.from_data_list(data_list)


# =============================================================================
# Node Feature Augmentation
# =============================================================================


class NodeFeatureAugmenter:
    """
    Augment node features with additional computed attributes.

    Can be used as a transform for CausalGraphDataset.
    """

    def __init__(
        self,
        add_degree: bool = True,
        add_pagerank: bool = False,
        add_clustering: bool = False,
    ):
        """
        Initialize augmenter.

        Args:
            add_degree: Add in/out degree features
            add_pagerank: Add PageRank scores
            add_clustering: Add clustering coefficients
        """
        self.add_degree = add_degree
        self.add_pagerank = add_pagerank
        self.add_clustering = add_clustering

    def __call__(self, data: "Data") -> "Data":
        """Augment a PyG Data object."""
        import torch

        x = data.x
        edge_index = data.edge_index
        num_nodes = x.shape[0]

        augmented_features = []

        if self.add_degree:
            # Compute in-degree and out-degree
            in_degree = torch.zeros(num_nodes, dtype=torch.float)
            out_degree = torch.zeros(num_nodes, dtype=torch.float)

            if edge_index.shape[1] > 0:
                for i in range(edge_index.shape[1]):
                    src = edge_index[0, i].item()
                    dst = edge_index[1, i].item()
                    out_degree[src] += 1
                    in_degree[dst] += 1

            # Normalize
            max_deg = max(in_degree.max().item(), out_degree.max().item(), 1)
            in_degree = in_degree / max_deg
            out_degree = out_degree / max_deg

            augmented_features.extend([in_degree.unsqueeze(1), out_degree.unsqueeze(1)])

        if self.add_pagerank:
            # Simple power iteration PageRank
            pagerank = self._compute_pagerank(edge_index, num_nodes)
            augmented_features.append(pagerank.unsqueeze(1))

        if self.add_clustering:
            # Clustering coefficient (simplified)
            clustering = self._compute_clustering(edge_index, num_nodes)
            augmented_features.append(clustering.unsqueeze(1))

        if augmented_features:
            augmented = torch.cat(augmented_features, dim=1)
            data.x = torch.cat([x, augmented], dim=1)

        return data

    def _compute_pagerank(
        self, edge_index: "torch.Tensor", num_nodes: int, damping: float = 0.85, iterations: int = 10
    ) -> "torch.Tensor":
        """Compute PageRank scores."""
        import torch

        if edge_index.shape[1] == 0 or num_nodes == 0:
            return torch.ones(num_nodes) / max(num_nodes, 1)

        # Build adjacency
        pr = torch.ones(num_nodes) / num_nodes
        out_degree = torch.zeros(num_nodes)

        for i in range(edge_index.shape[1]):
            out_degree[edge_index[0, i]] += 1

        out_degree = torch.clamp(out_degree, min=1)

        for _ in range(iterations):
            new_pr = torch.zeros(num_nodes)
            for i in range(edge_index.shape[1]):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                new_pr[dst] += pr[src] / out_degree[src]
            pr = (1 - damping) / num_nodes + damping * new_pr

        # Normalize to [0, 1]
        pr = pr / pr.max() if pr.max() > 0 else pr
        return pr

    def _compute_clustering(self, edge_index: "torch.Tensor", num_nodes: int) -> "torch.Tensor":
        """Compute local clustering coefficients."""
        import torch

        clustering = torch.zeros(num_nodes)

        if edge_index.shape[1] == 0:
            return clustering

        # Build neighbor sets
        neighbors = [set() for _ in range(num_nodes)]
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            neighbors[src].add(dst)
            neighbors[dst].add(src)  # Treat as undirected for clustering

        for node in range(num_nodes):
            neighbor_list = list(neighbors[node])
            k = len(neighbor_list)
            if k < 2:
                continue

            # Count edges between neighbors
            edges = 0
            for i in range(len(neighbor_list)):
                for j in range(i + 1, len(neighbor_list)):
                    if neighbor_list[j] in neighbors[neighbor_list[i]]:
                        edges += 1

            max_edges = k * (k - 1) / 2
            clustering[node] = edges / max_edges if max_edges > 0 else 0

        return clustering


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "CausalGraphDataset",
    "InMemoryCausalGraphDataset",
    "create_data_loaders",
    "collate_graphs",
    "NodeFeatureAugmenter",
]
