"""
Graph Neural Network based attack detector.

Uses GCN, GAT, GraphSAGE, or GIN architectures for graph classification
to detect attacks from causal graph structure.

Supports:
- Multiple GNN architectures (GCN, GAT, GraphSAGE, GIN)
- Multiple pooling strategies (mean, max, add, attention)
- Rich node and edge feature encoding
- Integration with CausalTrace BaseDetector interface
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field, asdict
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import (
        GCNConv,
        GATConv,
        GINConv,
        SAGEConv,
        global_mean_pool,
        global_max_pool,
        global_add_pool,
        GlobalAttention,
    )

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch Geometric not available. Install with: "
        "pip install torch torch_geometric torch_scatter torch_sparse"
    )

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)

from .detector import BaseDetector, DetectionResult
from ..graph import CausalGraph


# =============================================================================
# Node and Edge Feature Encoders
# =============================================================================

ACTION_TYPES = [
    "navigate",
    "click",
    "type",
    "submit",
    "read",
    "scroll",
    "hover",
    "select",
    "upload",
    "download",
    "login",
    "logout",
    "tool_call",
    "api_call",
    "execute",
    "unknown",
]

EDGE_TYPES = [
    "data_dependency",
    "trust_transfer",
    "state_enablement",
    "unknown",
]

DOMAIN_CATEGORIES = [
    "gitlab",
    "reddit",
    "github",
    "google",
    "attacker",
    "bank",
    "mail",
    "search",
    "shop",
    "forum",
    "unknown",
    "other",
]


def encode_action_type(action_type: str) -> List[float]:
    """One-hot encode action type."""
    action_type = str(action_type).lower() if action_type else "unknown"
    vec = [0.0] * len(ACTION_TYPES)
    if action_type in ACTION_TYPES:
        vec[ACTION_TYPES.index(action_type)] = 1.0
    else:
        vec[ACTION_TYPES.index("unknown")] = 1.0
    return vec


def encode_domain(domain: str) -> List[float]:
    """Encode domain into category."""
    domain = str(domain).lower() if domain else ""
    vec = [0.0] * len(DOMAIN_CATEGORIES)

    if "gitlab" in domain:
        vec[DOMAIN_CATEGORIES.index("gitlab")] = 1.0
    elif "reddit" in domain:
        vec[DOMAIN_CATEGORIES.index("reddit")] = 1.0
    elif "github" in domain:
        vec[DOMAIN_CATEGORIES.index("github")] = 1.0
    elif "google" in domain:
        vec[DOMAIN_CATEGORIES.index("google")] = 1.0
    elif "bank" in domain:
        vec[DOMAIN_CATEGORIES.index("bank")] = 1.0
    elif "mail" in domain:
        vec[DOMAIN_CATEGORIES.index("mail")] = 1.0
    elif "search" in domain:
        vec[DOMAIN_CATEGORIES.index("search")] = 1.0
    elif "shop" in domain:
        vec[DOMAIN_CATEGORIES.index("shop")] = 1.0
    elif "forum" in domain:
        vec[DOMAIN_CATEGORIES.index("forum")] = 1.0
    elif any(
        x in domain for x in ["attacker", "evil", "malicious", "collector", "harvest"]
    ):
        vec[DOMAIN_CATEGORIES.index("attacker")] = 1.0
    elif domain == "" or domain == "unknown":
        vec[DOMAIN_CATEGORIES.index("unknown")] = 1.0
    else:
        vec[DOMAIN_CATEGORIES.index("other")] = 1.0

    return vec


def encode_edge_type(edge_type: str) -> List[float]:
    """One-hot encode edge type."""
    edge_type = str(edge_type).lower() if edge_type else "unknown"
    # Handle EdgeType enum values
    if hasattr(edge_type, "value"):
        edge_type = edge_type.value
    vec = [0.0] * len(EDGE_TYPES)
    if edge_type in EDGE_TYPES:
        vec[EDGE_TYPES.index(edge_type)] = 1.0
    else:
        vec[EDGE_TYPES.index("unknown")] = 1.0
    return vec


def get_node_feature_dim() -> int:
    """Get dimension of node features."""
    # action_type (one-hot 16) + domain (one-hot 12) + numeric features (7)
    # Numeric: has_injection, is_cross_domain, data_produced, data_consumed,
    #          position, is_untrusted, injection_detected
    return len(ACTION_TYPES) + len(DOMAIN_CATEGORIES) + 7


def get_edge_feature_dim() -> int:
    """Get dimension of edge features."""
    # edge_type (one-hot) + cross_domain + injection_related + temporal_distance
    return len(EDGE_TYPES) + 3


# =============================================================================
# Graph Conversion
# =============================================================================


def causal_graph_to_pyg(
    graph: Union[CausalGraph, Dict[str, Any]], label: Optional[bool] = None
) -> "Data":
    """
    Convert a CausalGraph or graph dict to PyTorch Geometric Data object.

    Args:
        graph: CausalGraph instance or dictionary with nodes/edges
        label: Optional label for the graph (True=attack, False=benign)

    Returns:
        PyG Data object with node features, edge index, and edge attributes
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch Geometric required but not installed")

    # Handle both CausalGraph objects and dictionaries
    if isinstance(graph, CausalGraph):
        graph_dict = graph.export_to_json()
    else:
        graph_dict = graph

    nodes = graph_dict.get("nodes", [])
    edges = graph_dict.get("edges", [])

    # Handle empty graphs
    if len(nodes) == 0:
        x = torch.zeros((1, get_node_feature_dim()), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, get_edge_feature_dim()), dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if label is not None:
            data.y = torch.tensor([int(label)], dtype=torch.long)
        return data

    # Create node ID mapping
    node_id_to_idx = {}
    for idx, node in enumerate(nodes):
        node_id = node.get("action_id", node.get("id", idx))
        node_id_to_idx[node_id] = idx

    # Encode node features
    node_features = []
    num_nodes = len(nodes)

    for idx, node in enumerate(nodes):
        features = []

        # Action type (one-hot)
        action_type = node.get("action_type", "unknown")
        features.extend(encode_action_type(action_type))

        # Domain (one-hot category)
        domain = node.get("domain", "unknown")
        features.extend(encode_domain(domain))

        # Numeric features
        features.append(float(node.get("has_injection", False)))
        features.append(float(node.get("is_cross_domain", False)))
        features.append(float(len(node.get("data_produced", []))))
        features.append(float(len(node.get("data_consumed", []))))

        # Position in trajectory (normalized)
        features.append(float(idx) / max(num_nodes - 1, 1))

        # Trust level (if available)
        provenance = node.get("provenance", {})
        is_untrusted = provenance.get("is_untrusted", False)
        injection_detected = provenance.get("injection_detected", False)
        features.append(float(is_untrusted))
        features.append(float(injection_detected))

        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    # Encode edges
    edge_sources = []
    edge_targets = []
    edge_features = []

    for edge in edges:
        source = edge.get("source", edge.get("from", edge.get("source_action_id")))
        target = edge.get("target", edge.get("to", edge.get("target_action_id")))

        # Map to indices
        if source in node_id_to_idx and target in node_id_to_idx:
            source_idx = node_id_to_idx[source]
            target_idx = node_id_to_idx[target]

            edge_sources.append(source_idx)
            edge_targets.append(target_idx)

            # Edge features
            e_features = []
            edge_type = edge.get("edge_type", edge.get("type", "unknown"))
            e_features.extend(encode_edge_type(edge_type))

            metadata = edge.get("metadata", {})
            e_features.append(float(metadata.get("cross_domain", False)))
            e_features.append(float(metadata.get("injection_related", False)))

            # Temporal distance (normalized by trajectory length)
            temporal_dist = abs(target_idx - source_idx) / max(num_nodes - 1, 1)
            e_features.append(temporal_dist)

            edge_features.append(e_features)

    if len(edge_sources) > 0:
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, get_edge_feature_dim()), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if label is not None:
        data.y = torch.tensor([int(label)], dtype=torch.long)

    return data


# =============================================================================
# GNN Models
# =============================================================================

if not TORCH_AVAILABLE:
    # Define stub classes when PyTorch is not available
    class GCNClassifier:
        """Stub class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for GNN models")

    class GATClassifier:
        """Stub class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for GNN models")

    class GraphSAGEClassifier:
        """Stub class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for GNN models")

    class GINClassifier:
        """Stub class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for GNN models")
else:
    class GCNClassifier(nn.Module):
        """Graph Convolutional Network for graph classification."""

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 64,
            num_layers: int = 3,
            dropout: float = 0.5,
            pooling: str = "mean",
        ):
            super().__init__()

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()

            # First layer
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

            # Hidden layers
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))

            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 2),  # Binary classification
            )

            self.dropout = dropout
            self.pooling = pooling

        def forward(self, x, edge_index, batch):
            # Graph convolutions
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            # Global pooling
            if self.pooling == "mean":
                x = global_mean_pool(x, batch)
            elif self.pooling == "max":
                x = global_max_pool(x, batch)
            else:
                x = global_add_pool(x, batch)

            # Classification
            return self.classifier(x)


    class GATClassifier(nn.Module):
        """Graph Attention Network for graph classification."""

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 64,
            num_layers: int = 3,
            heads: int = 4,
            dropout: float = 0.5,
            pooling: str = "mean",
        ):
            super().__init__()

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()

            # First layer
            self.convs.append(
                GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

            # Hidden layers
            for _ in range(num_layers - 1):
                self.convs.append(
                    GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
                )
                self.bns.append(nn.BatchNorm1d(hidden_channels))

            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 2),
            )

            self.dropout = dropout
            self.pooling = pooling

        def forward(self, x, edge_index, batch):
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            if self.pooling == "mean":
                x = global_mean_pool(x, batch)
            elif self.pooling == "max":
                x = global_max_pool(x, batch)
            else:
                x = global_add_pool(x, batch)

            return self.classifier(x)


    class GraphSAGEClassifier(nn.Module):
        """GraphSAGE for graph classification.

        GraphSAGE uses sampling and aggregation for inductive learning,
        making it suitable for graphs with varying structures.
        """

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 64,
            num_layers: int = 3,
            dropout: float = 0.5,
            pooling: str = "mean",
            aggr: str = "mean",  # Aggregation: mean, max, lstm
        ):
            super().__init__()

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()

            # First layer
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

            # Hidden layers
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
                self.bns.append(nn.BatchNorm1d(hidden_channels))

            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 2),
            )

            self.dropout = dropout
            self.pooling = pooling

        def forward(self, x, edge_index, batch):
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            if self.pooling == "mean":
                x = global_mean_pool(x, batch)
            elif self.pooling == "max":
                x = global_max_pool(x, batch)
            else:
                x = global_add_pool(x, batch)

            return self.classifier(x)


    class GINClassifier(nn.Module):
        """Graph Isomorphism Network for graph classification.

        GIN is theoretically the most expressive GNN for graph classification,
        as powerful as the Weisfeiler-Lehman graph isomorphism test.
        """

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 64,
            num_layers: int = 3,
            dropout: float = 0.5,
            pooling: str = "mean",
        ):
            super().__init__()

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()

            # First layer
            mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

            # Hidden layers
            for _ in range(num_layers - 1):
                mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                self.convs.append(GINConv(mlp, train_eps=True))
                self.bns.append(nn.BatchNorm1d(hidden_channels))

            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 2),
            )

            self.dropout = dropout
            self.pooling = pooling

        def forward(self, x, edge_index, batch):
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            if self.pooling == "mean":
                x = global_mean_pool(x, batch)
            elif self.pooling == "max":
                x = global_max_pool(x, batch)
            else:
                x = global_add_pool(x, batch)

            return self.classifier(x)


    class AttentionPoolClassifier(nn.Module):
        """GNN with attention-based graph pooling.

        Uses learned attention weights for graph-level readout instead of
        simple mean/max pooling.
        """

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 64,
            num_layers: int = 3,
            dropout: float = 0.5,
            base_model: str = "gcn",  # gcn, gat, sage, gin
        ):
            super().__init__()

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()

            # First layer
            if base_model == "gat":
                self.convs.append(
                    GATConv(in_channels, hidden_channels, heads=4, concat=False)
                )
            elif base_model == "sage":
                self.convs.append(SAGEConv(in_channels, hidden_channels))
            elif base_model == "gin":
                mlp = nn.Sequential(
                    nn.Linear(in_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                self.convs.append(GINConv(mlp))
            else:  # gcn
                self.convs.append(GCNConv(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

            # Hidden layers
            for _ in range(num_layers - 1):
                if base_model == "gat":
                    self.convs.append(
                        GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
                    )
                elif base_model == "sage":
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                elif base_model == "gin":
                    mlp = nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, hidden_channels),
                    )
                    self.convs.append(GINConv(mlp))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))

            # Attention pooling
            gate_nn = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, 1),
            )
            self.pool = GlobalAttention(gate_nn)

            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 2),
            )

            self.dropout = dropout

        def forward(self, x, edge_index, batch):
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            # Attention pooling
            x = self.pool(x, batch)

            return self.classifier(x)


# =============================================================================
# GNN Detector Configuration
# =============================================================================


@dataclass
class GNNConfig:
    """Configuration for GNN detector."""

    model_type: str = "gin"  # gcn, gat, sage, gin, attention
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.5
    pooling: str = "mean"  # mean, max, add (or attention for AttentionPoolClassifier)
    heads: int = 4  # For GAT only
    sage_aggr: str = "mean"  # For GraphSAGE: mean, max
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 32
    patience: int = 15  # Early stopping patience
    min_delta: float = 0.001  # Minimum improvement for early stopping
    use_class_weights: bool = True  # Weight loss by class frequency
    lr_scheduler: str = "plateau"  # none, plateau, step, cosine

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GNNConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# GNN Detector Class
# =============================================================================


class GNNDetector(BaseDetector):
    """
    GNN-based attack detector implementing BaseDetector interface.

    Supports multiple GNN architectures for graph-level classification
    to detect attacks from causal graph structure.
    """

    def __init__(self, config: Optional[GNNConfig] = None):
        """
        Initialize GNN detector.

        Args:
            config: GNN configuration. If None, uses default config.

        Raises:
            ImportError: If PyTorch Geometric is not installed.
        """
        super().__init__()

        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric required. Install with: "
                "pip install torch torch_geometric torch_scatter torch_sparse"
            )

        self.config = config or GNNConfig()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self._in_channels = get_node_feature_dim()

    def _create_model(self, in_channels: int) -> nn.Module:
        """Create GNN model based on config."""
        if self.config.model_type == "gcn":
            return GCNClassifier(
                in_channels=in_channels,
                hidden_channels=self.config.hidden_channels,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                pooling=self.config.pooling,
            )
        elif self.config.model_type == "gat":
            return GATClassifier(
                in_channels=in_channels,
                hidden_channels=self.config.hidden_channels,
                num_layers=self.config.num_layers,
                heads=self.config.heads,
                dropout=self.config.dropout,
                pooling=self.config.pooling,
            )
        elif self.config.model_type == "sage":
            return GraphSAGEClassifier(
                in_channels=in_channels,
                hidden_channels=self.config.hidden_channels,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                pooling=self.config.pooling,
                aggr=self.config.sage_aggr,
            )
        elif self.config.model_type == "attention":
            return AttentionPoolClassifier(
                in_channels=in_channels,
                hidden_channels=self.config.hidden_channels,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                base_model="gcn",
            )
        else:  # gin (default)
            return GINClassifier(
                in_channels=in_channels,
                hidden_channels=self.config.hidden_channels,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                pooling=self.config.pooling,
            )

    def fit(
        self,
        graphs: List[Union[CausalGraph, Dict[str, Any]]],
        labels: List[bool],
        val_split: float = 0.15,
        verbose: bool = True,
    ) -> "GNNDetector":
        """
        Train the GNN detector.

        Args:
            graphs: List of CausalGraph objects or graph dictionaries
            labels: List of boolean labels (True=attack, False=benign)
            val_split: Fraction of data for validation
            verbose: Print training progress

        Returns:
            self (for method chaining)
        """
        # Validate input
        if len(graphs) != len(labels):
            raise ValueError(f"Mismatch: {len(graphs)} graphs vs {len(labels)} labels")

        # Convert graphs to PyG format
        if verbose:
            print(f"Converting {len(graphs)} graphs to PyG format...")

        pyg_graphs = []
        valid_labels = []
        for g, label in zip(graphs, labels):
            try:
                data = causal_graph_to_pyg(g, label)
                pyg_graphs.append(data)
                valid_labels.append(label)
            except Exception as e:
                if verbose:
                    print(f"  Skipping graph: {e}")

        if verbose:
            print(f"  Converted {len(pyg_graphs)} graphs successfully")

        if len(pyg_graphs) < 2:
            raise ValueError("Need at least 2 valid graphs for training")

        # Check class balance
        num_attacks = sum(valid_labels)
        num_benign = len(valid_labels) - num_attacks
        if num_attacks == 0 or num_benign == 0:
            raise ValueError("Need at least one sample from each class")

        # Stratified train/val split
        min_val_size = max(2, int(len(pyg_graphs) * val_split))
        if min_val_size >= len(pyg_graphs):
            min_val_size = max(1, len(pyg_graphs) // 5)

        train_data, val_data = train_test_split(
            pyg_graphs,
            test_size=val_split,
            random_state=42,
            stratify=valid_labels if min(num_attacks, num_benign) >= 2 else None,
        )

        train_loader = DataLoader(
            train_data, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_data, batch_size=self.config.batch_size)

        # Create model
        self._in_channels = pyg_graphs[0].x.shape[1]
        self.model = self._create_model(self._in_channels).to(self.device)

        # Class weights for imbalanced data
        if self.config.use_class_weights:
            weight = torch.tensor(
                [num_attacks / len(valid_labels), num_benign / len(valid_labels)]
            ).to(self.device)
        else:
            weight = None

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        scheduler = None
        if self.config.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
        elif self.config.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.5
            )
        elif self.config.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.epochs
            )

        criterion = nn.CrossEntropyLoss(weight=weight)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        if verbose:
            print(f"\nTraining {self.config.model_type.upper()} model...")
            print(f"  Device: {self.device}")
            print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
            print(f"  Epochs: {self.config.epochs}, Batch size: {self.config.batch_size}")
            print(f"  Class distribution: {num_attacks} attacks, {num_benign} benign")

        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

        for epoch in range(self.config.epochs):
            # Train
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels_epoch = []

            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch.num_graphs

                preds = out.argmax(dim=1)
                train_preds.extend(preds.cpu().tolist())
                train_labels_epoch.extend(batch.y.cpu().tolist())

            train_loss /= len(train_data)
            self.train_losses.append(train_loss)

            # Train metrics
            train_f1 = f1_score(train_labels_epoch, train_preds, zero_division=0)
            self.train_metrics.append({"f1": train_f1, "loss": train_loss})

            # Validate
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_labels_epoch = []

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = self.model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out, batch.y)
                    val_loss += loss.item() * batch.num_graphs

                    preds = out.argmax(dim=1)
                    val_preds.extend(preds.cpu().tolist())
                    val_labels_epoch.extend(batch.y.cpu().tolist())

            val_loss /= len(val_data)
            self.val_losses.append(val_loss)

            # Val metrics
            val_f1 = f1_score(val_labels_epoch, val_preds, zero_division=0)
            self.val_metrics.append({"f1": val_f1, "loss": val_loss})

            # Learning rate scheduling
            if scheduler is not None:
                if self.config.lr_scheduler == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Early stopping
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, val_f1={val_f1:.3f}, lr={current_lr:.6f}"
                )

            if patience_counter >= self.config.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model = self.model.to(self.device)

        # Final validation metrics
        val_preds_final, val_probs_final = self._predict_loader(val_loader)
        val_labels_final = [d.y.item() for d in val_data]

        self.is_fitted = True
        self.metadata["num_training_samples"] = len(graphs)
        self.metadata["num_attacks"] = sum(labels)
        self.metadata["model_type"] = self.config.model_type
        self.metadata["config"] = self.config.to_dict()
        self.metadata["best_val_loss"] = best_val_loss
        self.metadata["epochs_trained"] = len(self.train_losses)

        if verbose:
            final_precision = precision_score(val_labels_final, val_preds_final, zero_division=0)
            final_recall = recall_score(val_labels_final, val_preds_final, zero_division=0)
            final_f1 = f1_score(val_labels_final, val_preds_final, zero_division=0)
            final_acc = accuracy_score(val_labels_final, val_preds_final)

            print(f"\nValidation Results:")
            print(f"  Precision: {final_precision:.3f}")
            print(f"  Recall:    {final_recall:.3f}")
            print(f"  F1 Score:  {final_f1:.3f}")
            print(f"  Accuracy:  {final_acc:.3f}")

        return self

    def _predict_loader(self, loader: "DataLoader") -> Tuple[List[int], List[float]]:
        """Get predictions for a data loader."""
        self.model.eval()
        predictions = []
        probabilities = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                probs = F.softmax(out, dim=1)
                preds = out.argmax(dim=1)

                predictions.extend(preds.cpu().tolist())
                probabilities.extend(probs[:, 1].cpu().tolist())

        return predictions, probabilities

    def predict(self, graph: Union[CausalGraph, Dict[str, Any]]) -> DetectionResult:
        """
        Predict whether a causal graph represents an attack.

        Args:
            graph: CausalGraph object or graph dictionary

        Returns:
            DetectionResult with classification and explanation
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Detector must be fitted before prediction")

        # Convert to PyG
        try:
            data = causal_graph_to_pyg(graph)
        except Exception as e:
            # Return uncertain result for failed conversion
            return DetectionResult(
                is_attack=False,
                confidence=0.5,
                triggered_features={"error": str(e)},
                explanation=f"Failed to convert graph: {e}",
            )

        # Create batch of one
        data = data.to(self.device)
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index, batch)
            probs = F.softmax(out, dim=1)
            pred = out.argmax(dim=1).item()
            attack_prob = probs[0, 1].item()

        is_attack = bool(pred == 1)
        confidence = attack_prob if is_attack else (1.0 - attack_prob)

        # Generate explanation
        explanation = self._generate_explanation(graph, is_attack, confidence, attack_prob)

        result = DetectionResult(
            is_attack=is_attack,
            confidence=confidence,
            triggered_features={
                "attack_probability": attack_prob,
                "model_type": self.config.model_type,
            },
            explanation=explanation,
            raw_scores={"attack_prob": attack_prob, "benign_prob": 1.0 - attack_prob},
        )

        # Attach watermark metadata if available
        if isinstance(graph, CausalGraph):
            result = self._attach_watermark_metadata(graph, result)

        return result

    def predict_batch(
        self, graphs: List[Union[CausalGraph, Dict[str, Any]]]
    ) -> List[DetectionResult]:
        """
        Predict for multiple graphs (optimized batch version).

        Args:
            graphs: List of CausalGraph objects or graph dictionaries

        Returns:
            List of DetectionResult objects
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Detector must be fitted before prediction")

        # Convert all graphs
        pyg_graphs = []
        conversion_errors = []
        for i, g in enumerate(graphs):
            try:
                data = causal_graph_to_pyg(g)
                pyg_graphs.append((i, data, g))
                conversion_errors.append(None)
            except Exception as e:
                conversion_errors.append(str(e))

        # Batch predict
        loader = DataLoader(
            [d for _, d, _ in pyg_graphs], batch_size=self.config.batch_size
        )
        all_preds, all_probs = self._predict_loader(loader)

        # Create results
        results = [None] * len(graphs)
        pred_idx = 0

        for i, error in enumerate(conversion_errors):
            if error is not None:
                results[i] = DetectionResult(
                    is_attack=False,
                    confidence=0.5,
                    triggered_features={"error": error},
                    explanation=f"Failed to convert graph: {error}",
                )
            else:
                orig_idx, _, orig_graph = pyg_graphs[pred_idx]
                pred = all_preds[pred_idx]
                attack_prob = all_probs[pred_idx]

                is_attack = bool(pred == 1)
                confidence = attack_prob if is_attack else (1.0 - attack_prob)
                explanation = self._generate_explanation(
                    orig_graph, is_attack, confidence, attack_prob
                )

                result = DetectionResult(
                    is_attack=is_attack,
                    confidence=confidence,
                    triggered_features={
                        "attack_probability": attack_prob,
                        "model_type": self.config.model_type,
                    },
                    explanation=explanation,
                    raw_scores={"attack_prob": attack_prob, "benign_prob": 1.0 - attack_prob},
                )

                if isinstance(orig_graph, CausalGraph):
                    result = self._attach_watermark_metadata(orig_graph, result)

                results[orig_idx] = result
                pred_idx += 1

        return results

    def _generate_explanation(
        self,
        graph: Union[CausalGraph, Dict[str, Any]],
        is_attack: bool,
        confidence: float,
        attack_prob: float,
    ) -> str:
        """Generate human-readable explanation."""
        result_str = "attack" if is_attack else "benign"

        # Get graph statistics
        if isinstance(graph, CausalGraph):
            num_nodes = graph.num_nodes()
            num_edges = graph.num_edges()
        else:
            num_nodes = len(graph.get("nodes", []))
            num_edges = len(graph.get("edges", []))

        return (
            f"Classified as {result_str} with {confidence:.1%} confidence "
            f"(attack_prob={attack_prob:.3f}). "
            f"Graph has {num_nodes} nodes and {num_edges} edges. "
            f"Model: {self.config.model_type.upper()}"
        )

    def save(self, path: str) -> None:
        """
        Save trained detector to disk.

        Args:
            path: Path to save directory
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save unfitted detector")

        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = path_obj / "model.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config.to_dict(),
                "in_channels": self._in_channels,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "train_metrics": self.train_metrics,
                "val_metrics": self.val_metrics,
                "metadata": self.metadata,
            },
            model_path,
        )

        print(f"GNN Detector saved to {path}")

    @classmethod
    def load(cls, path: str) -> "GNNDetector":
        """
        Load trained detector from disk.

        Args:
            path: Path to saved detector directory

        Returns:
            Loaded GNNDetector instance
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch Geometric required to load GNN detector")

        path_obj = Path(path)
        model_path = path_obj / "model.pt"

        checkpoint = torch.load(model_path, map_location="cpu")

        # Create detector with saved config
        config = GNNConfig.from_dict(checkpoint["config"])
        detector = cls(config=config)

        # Restore state
        detector._in_channels = checkpoint["in_channels"]
        detector.model = detector._create_model(detector._in_channels)
        detector.model.load_state_dict(checkpoint["model_state_dict"])
        detector.model = detector.model.to(detector.device)

        detector.train_losses = checkpoint["train_losses"]
        detector.val_losses = checkpoint["val_losses"]
        detector.train_metrics = checkpoint.get("train_metrics", [])
        detector.val_metrics = checkpoint.get("val_metrics", [])
        detector.metadata = checkpoint["metadata"]
        detector.is_fitted = True

        print(f"GNN Detector loaded from {path}")
        return detector

    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history.

        Returns:
            Dictionary with train_losses, val_losses, and metrics
        """
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
        }

    def __repr__(self) -> str:
        """String representation."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        return f"GNNDetector(model={self.config.model_type}, {fitted_status})"


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_gnn_models(
    graphs: List[Union[CausalGraph, Dict[str, Any]]],
    labels: List[bool],
    test_split: float = 0.3,
    model_types: Optional[List[str]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple GNN architectures on the same data.

    Args:
        graphs: List of CausalGraph objects or dictionaries
        labels: List of labels (True=attack, False=benign)
        test_split: Fraction for test set
        model_types: List of model types to evaluate (default: all)
        config_overrides: Override default config parameters
        verbose: Print progress

    Returns:
        Dictionary mapping model type to test metrics
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch Geometric required")

    if model_types is None:
        model_types = ["gcn", "gat", "sage", "gin"]

    # Train/test split
    train_graphs, test_graphs, train_labels, test_labels = train_test_split(
        graphs, labels, test_size=test_split, random_state=42, stratify=labels
    )

    results = {}

    for model_type in model_types:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating {model_type.upper()}")
            print(f"{'='*60}")

        config_params = {
            "model_type": model_type,
            "hidden_channels": 64,
            "num_layers": 3,
            "epochs": 100,
            "patience": 15,
        }
        if config_overrides:
            config_params.update(config_overrides)

        config = GNNConfig(**config_params)
        detector = GNNDetector(config)

        try:
            detector.fit(train_graphs, train_labels, verbose=verbose)

            # Test predictions
            test_results = detector.predict_batch(test_graphs)
            test_preds = [r.is_attack for r in test_results]
            test_probs = [r.raw_scores["attack_prob"] for r in test_results]

            test_metrics = {
                "precision": precision_score(test_labels, test_preds, zero_division=0),
                "recall": recall_score(test_labels, test_preds, zero_division=0),
                "f1": f1_score(test_labels, test_preds, zero_division=0),
                "accuracy": accuracy_score(test_labels, test_preds),
            }

            # ROC AUC if we have both classes in test set
            if len(set(test_labels)) > 1:
                test_metrics["roc_auc"] = roc_auc_score(test_labels, test_probs)

            if verbose:
                print(f"\nTest Results:")
                print(f"  Precision: {test_metrics['precision']:.3f}")
                print(f"  Recall:    {test_metrics['recall']:.3f}")
                print(f"  F1 Score:  {test_metrics['f1']:.3f}")
                print(f"  Accuracy:  {test_metrics['accuracy']:.3f}")
                if "roc_auc" in test_metrics:
                    print(f"  ROC AUC:   {test_metrics['roc_auc']:.3f}")

            results[model_type] = test_metrics

        except Exception as e:
            if verbose:
                print(f"  Error training {model_type}: {e}")
            results[model_type] = {"error": str(e)}

    return results


# =============================================================================
# Module Entry Point
# =============================================================================

if __name__ == "__main__":
    print("GNN Detector Module")
    print(f"PyTorch Geometric available: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"Node feature dimension: {get_node_feature_dim()}")
        print(f"Edge feature dimension: {get_edge_feature_dim()}")
        print(f"CUDA available: {torch.cuda.is_available()}")
