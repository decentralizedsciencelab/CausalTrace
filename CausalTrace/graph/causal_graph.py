"""
Causal graph representation and operations.

This module provides the core CausalGraph class for representing causal dependencies
between actions in an agent trajectory as a directed acyclic graph (DAG).

The implementation encapsulates the graph backend (currently NetworkX) to allow
future swapping to faster backends (rustworkx, igraph) or GNN frameworks (PyG, DGL).
"""

import networkx as nx
import json
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Iterator, Protocol
from enum import Enum
from causaltrace.models.trajectory import Trajectory, Action


class EdgeType(Enum):
    """Types of causal edges between actions."""
    DATA_DEPENDENCY = "data_dependency"
    TRUST_TRANSFER = "trust_transfer"
    STATE_ENABLEMENT = "state_enablement"


@dataclass
class CausalEdge:
    """
    Represents a causal edge between two actions.

    Attributes:
        source_action_id: ID of the source action
        target_action_id: ID of the target action
        edge_type: Type of causal dependency
        metadata: Evidence and additional information about the edge
    """
    source_action_id: int
    target_action_id: int
    edge_type: EdgeType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary."""
        return {
            "source_action_id": self.source_action_id,
            "target_action_id": self.target_action_id,
            "edge_type": self.edge_type.value,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalEdge":
        """Create edge from dictionary."""
        return cls(
            source_action_id=data["source_action_id"],
            target_action_id=data["target_action_id"],
            edge_type=EdgeType(data["edge_type"]),
            metadata=data.get("metadata", {})
        )


class CausalGraph:
    """
    Directed acyclic graph of causal dependencies between actions.

    The graph represents causal relationships in an agent trajectory using three
    types of edges: data dependencies, trust transfer, and state enablement.

    The internal graph backend is encapsulated - do not access _graph directly.
    Use the provided methods or adapters (e.g., to_pyg()) for external access.
    """

    def __init__(self, trajectory: Trajectory):
        """
        Initialize causal graph from a trajectory.

        Args:
            trajectory: The trajectory to build the graph from
        """
        self.trajectory = trajectory
        self._graph = nx.DiGraph()  # Private - do not access directly
        self.metadata: Dict[str, Any] = {}
        self._build_nodes()

    def _build_nodes(self):
        """Add action nodes to the graph."""
        # Build observation lookup for embedding in nodes
        obs_by_id = {}
        for chunk in self.trajectory.observation_chunks:
            obs_by_id[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content[:500] if chunk.content else "",  # Truncate for efficiency
                "domain": chunk.domain,
                "metadata": chunk.metadata if chunk.metadata else {},  # Include metadata for injection detection
            }

        for action in self.trajectory.actions:
            # Get observation chunks for this action
            obs_chunks = []
            if action.provenance and action.provenance.observation_chunks:
                for obs_id in action.provenance.observation_chunks:
                    if obs_id in obs_by_id:
                        obs_chunks.append(obs_by_id[obs_id])

            # Get provenance info
            provenance_info = {}
            if action.provenance:
                provenance_info = {
                    "is_untrusted": action.provenance.is_untrusted,
                    "injection_detected": action.provenance.injection_detected,
                    "untrusted_domains": action.provenance.untrusted_domains,
                }

            self._graph.add_node(
                action.action_id,
                action_type=action.action_type,
                domain=action.domain,
                timestamp=action.timestamp,
                target=action.target,
                data_produced=action.data_produced,
                data_consumed=action.data_consumed,
                observation_chunks=obs_chunks,
                provenance=provenance_info,
            )

    # =========================================================================
    # Core Graph Operations (encapsulated)
    # =========================================================================

    def add_edge(self, edge: CausalEdge):
        """
        Add a causal edge to the graph.

        Args:
            edge: The causal edge to add
        """
        self._graph.add_edge(
            edge.source_action_id,
            edge.target_action_id,
            edge_type=edge.edge_type,
            metadata=edge.metadata
        )

    def has_node(self, action_id: int) -> bool:
        """Check if a node exists in the graph."""
        return action_id in self._graph.nodes()

    def has_edge(self, source_id: int, target_id: int) -> bool:
        """Check if an edge exists between two nodes."""
        return self._graph.has_edge(source_id, target_id)

    def num_nodes(self) -> int:
        """Get number of nodes in the graph."""
        return len(self._graph.nodes())

    def num_edges(self) -> int:
        """Get number of edges in the graph."""
        return len(self._graph.edges())

    def node_ids(self) -> List[int]:
        """Get list of all node IDs."""
        return list(self._graph.nodes())

    def get_node_data(self, action_id: int) -> Optional[Dict[str, Any]]:
        """
        Get node attributes for an action.

        Args:
            action_id: The action ID

        Returns:
            Dictionary of node attributes, or None if not found
        """
        if action_id not in self._graph.nodes():
            return None
        return dict(self._graph.nodes[action_id])

    def set_node_attribute(self, action_id: int, key: str, value: Any) -> None:
        """Set a node attribute."""
        if action_id in self._graph.nodes():
            self._graph.nodes[action_id][key] = value

    def get_node_attribute(self, action_id: int, key: str, default: Any = None) -> Any:
        """Get a specific node attribute."""
        if action_id not in self._graph.nodes():
            return default
        return self._graph.nodes[action_id].get(key, default)

    def get_edge_data(self, source_id: int, target_id: int) -> Optional[Dict[str, Any]]:
        """
        Get edge attributes between two nodes.

        Args:
            source_id: Source action ID
            target_id: Target action ID

        Returns:
            Dictionary of edge attributes, or None if edge doesn't exist
        """
        if not self._graph.has_edge(source_id, target_id):
            return None
        return dict(self._graph.edges[source_id, target_id])

    def set_metadata(self, key: str, value: Any) -> None:
        """Attach metadata to the graph."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve graph-level metadata."""
        return self.metadata.get(key, default)

    def in_degree(self, action_id: int) -> int:
        """Get number of incoming edges for a node."""
        if action_id not in self._graph.nodes():
            return 0
        return self._graph.in_degree(action_id)

    def out_degree(self, action_id: int) -> int:
        """Get number of outgoing edges for a node."""
        if action_id not in self._graph.nodes():
            return 0
        return self._graph.out_degree(action_id)

    # =========================================================================
    # Iteration Methods
    # =========================================================================

    def iter_nodes(self, data: bool = False) -> Iterator:
        """
        Iterate over nodes.

        Args:
            data: If True, yield (node_id, node_data) tuples

        Yields:
            node_id or (node_id, node_data) tuples
        """
        if data:
            for node_id, node_data in self._graph.nodes(data=True):
                yield node_id, dict(node_data)
        else:
            yield from self._graph.nodes()

    def iter_edges(self, data: bool = False) -> Iterator:
        """
        Iterate over edges.

        Args:
            data: If True, yield (source, target, edge_data) tuples

        Yields:
            (source, target) or (source, target, edge_data) tuples
        """
        if data:
            for source, target, edge_data in self._graph.edges(data=True):
                yield source, target, dict(edge_data)
        else:
            for source, target in self._graph.edges():
                yield source, target

    # =========================================================================
    # Graph Traversal
    # =========================================================================

    def get_predecessors(self, action_id: int) -> List[int]:
        """
        Get all immediate predecessors of an action.

        Predecessors are actions that have edges pointing TO the target action.

        Args:
            action_id: ID of the target action

        Returns:
            List of predecessor action IDs
        """
        if action_id not in self._graph.nodes():
            return []
        return list(self._graph.predecessors(action_id))

    def get_successors(self, action_id: int) -> List[int]:
        """
        Get all immediate successors of an action.

        Successors are actions that the target action has edges pointing TO.

        Args:
            action_id: ID of the source action

        Returns:
            List of successor action IDs
        """
        if action_id not in self._graph.nodes():
            return []
        return list(self._graph.successors(action_id))

    def get_ancestors(self, action_id: int) -> Set[int]:
        """
        Get all ancestors of an action (transitive predecessors).

        Args:
            action_id: ID of the action

        Returns:
            Set of all ancestor action IDs
        """
        if action_id not in self._graph.nodes():
            return set()
        return nx.ancestors(self._graph, action_id)

    def get_descendants(self, action_id: int) -> Set[int]:
        """
        Get all descendants of an action (transitive successors).

        Args:
            action_id: ID of the action

        Returns:
            Set of all descendant action IDs
        """
        if action_id not in self._graph.nodes():
            return set()
        return nx.descendants(self._graph, action_id)

    def get_root_nodes(self) -> List[int]:
        """Get nodes with no incoming edges."""
        return [n for n in self._graph.nodes() if self._graph.in_degree(n) == 0]

    def get_leaf_nodes(self) -> List[int]:
        """Get nodes with no outgoing edges."""
        return [n for n in self._graph.nodes() if self._graph.out_degree(n) == 0]

    def has_path(self, source_id: int, target_id: int) -> bool:
        """Check if there's a path from source to target."""
        if source_id not in self._graph.nodes() or target_id not in self._graph.nodes():
            return False
        return nx.has_path(self._graph, source_id, target_id)

    def shortest_path_length(self, source_id: int, target_id: int) -> int:
        """Get shortest path length between two nodes, or -1 if no path."""
        if not self.has_path(source_id, target_id):
            return -1
        return nx.shortest_path_length(self._graph, source_id, target_id)

    def topological_sort(self) -> List[int]:
        """
        Get nodes in topological order.

        Returns:
            List of node IDs in topological order

        Raises:
            ValueError: If graph has cycles
        """
        try:
            return list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Graph contains cycles, topological sort not possible")

    # =========================================================================
    # Edge Queries
    # =========================================================================

    def get_edges_by_type(self, edge_type: EdgeType) -> List[CausalEdge]:
        """
        Get all edges of a specific type.

        Args:
            edge_type: The type of edges to retrieve

        Returns:
            List of causal edges of the specified type
        """
        edges = []
        for source, target, data in self._graph.edges(data=True):
            if data.get("edge_type") == edge_type:
                edges.append(CausalEdge(
                    source_action_id=source,
                    target_action_id=target,
                    edge_type=edge_type,
                    metadata=data.get("metadata", {})
                ))
        return edges

    def get_all_edges(self) -> List[CausalEdge]:
        """
        Get all edges in the graph.

        Returns:
            List of all causal edges
        """
        edges = []
        for source, target, data in self._graph.edges(data=True):
            edges.append(CausalEdge(
                source_action_id=source,
                target_action_id=target,
                edge_type=data.get("edge_type"),
                metadata=data.get("metadata", {})
            ))
        return edges

    def get_cross_domain_edges(self) -> List[CausalEdge]:
        """
        Get edges connecting actions on different domains.

        Returns:
            List of edges where source and target are on different domains
        """
        cross_domain_edges = []
        for source, target, data in self._graph.edges(data=True):
            source_domain = self._graph.nodes[source].get("domain", "")
            target_domain = self._graph.nodes[target].get("domain", "")

            if source_domain and target_domain and source_domain != target_domain:
                cross_domain_edges.append(CausalEdge(
                    source_action_id=source,
                    target_action_id=target,
                    edge_type=data.get("edge_type"),
                    metadata=data.get("metadata", {})
                ))
        return cross_domain_edges

    # =========================================================================
    # Graph Metrics
    # =========================================================================

    def longest_path_length(self) -> int:
        """
        Compute chain depth (longest path in DAG).

        Returns:
            Length of the longest path, or 0 if graph is empty
        """
        if not self._graph.nodes():
            return 0

        try:
            longest = nx.dag_longest_path(self._graph)
            return len(longest) - 1 if longest else 0
        except nx.NetworkXError:
            return 0

    def get_bottleneck_nodes(self, threshold: float = 0.1) -> List[int]:
        """
        Identify bottleneck nodes (actions that many downstream actions depend on).

        A node is considered a bottleneck if removing it would disconnect
        more than threshold * total_nodes downstream nodes.

        Args:
            threshold: Proportion of total nodes (default: 0.1 = 10%)

        Returns:
            List of action IDs that are bottlenecks
        """
        if not self._graph.nodes():
            return []

        bottlenecks = []
        total_nodes = len(self._graph.nodes())
        threshold_count = int(threshold * total_nodes)

        for node in self._graph.nodes():
            descendants = nx.descendants(self._graph, node)
            if len(descendants) >= threshold_count:
                bottlenecks.append(node)

        return bottlenecks

    def get_node_depth(self, action_id: int) -> int:
        """
        Get the depth of a node (longest path from any root to this node).

        Args:
            action_id: ID of the action

        Returns:
            Depth of the node, or 0 if it's a root node
        """
        if action_id not in self._graph.nodes():
            return 0

        root_nodes = self.get_root_nodes()
        max_depth = 0

        for root in root_nodes:
            if nx.has_path(self._graph, root, action_id):
                path_length = nx.shortest_path_length(self._graph, root, action_id)
                max_depth = max(max_depth, path_length)

        return max_depth

    def is_dag(self) -> bool:
        """Check if the graph is a valid DAG (no cycles)."""
        return nx.is_directed_acyclic_graph(self._graph)

    # =========================================================================
    # Graph Copying
    # =========================================================================

    def copy(self) -> "CausalGraph":
        """
        Create a deep copy of the graph.

        Returns:
            New CausalGraph instance with copied data
        """
        new_graph = CausalGraph.__new__(CausalGraph)
        new_graph.trajectory = self.trajectory
        new_graph._graph = self._graph.copy()
        return new_graph

    def subgraph(self, node_ids: List[int]) -> "CausalGraph":
        """
        Create a subgraph containing only specified nodes.

        Args:
            node_ids: List of node IDs to include

        Returns:
            New CausalGraph with only the specified nodes and edges between them
        """
        new_graph = CausalGraph.__new__(CausalGraph)
        new_graph.trajectory = self.trajectory
        new_graph._graph = self._graph.subgraph(node_ids).copy()
        return new_graph

    # =========================================================================
    # Action Lookup
    # =========================================================================

    def get_node(self, action_id: int) -> Optional[Action]:
        """
        Get the Action object for an action ID.

        Args:
            action_id: The action ID to look up

        Returns:
            The Action object, or None if not found
        """
        for action in self.trajectory.actions:
            if action.action_id == action_id:
                return action
        return None

    # =========================================================================
    # Backend Adapters (for future GNN/ML integration)
    # =========================================================================

    def to_pyg(self) -> "PyGData":
        """
        Convert to PyTorch Geometric Data object for GNN processing.

        Returns:
            PyG Data object with node features and edge index

        Raises:
            ImportError: If torch_geometric is not installed
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError(
                "PyTorch Geometric required. Install with: "
                "pip install torch torch_geometric"
            )

        # Build node ID mapping (graph node IDs may not be contiguous)
        node_list = list(self._graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        # Build edge index tensor
        edge_index = []
        edge_types = []
        for source, target, data in self._graph.edges(data=True):
            edge_index.append([node_to_idx[source], node_to_idx[target]])
            edge_type = data.get("edge_type")
            if edge_type:
                edge_types.append(edge_type.value)
            else:
                edge_types.append("unknown")

        if edge_index:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)

        # Build node features
        # Basic features: action_type encoding, domain hash, in_degree, out_degree
        node_features = []
        action_type_map = {}  # Lazy encoding

        for node in node_list:
            node_data = self._graph.nodes[node]

            # Encode action type
            action_type = node_data.get("action_type")
            if hasattr(action_type, 'value'):
                action_type = action_type.value
            if action_type not in action_type_map:
                action_type_map[action_type] = len(action_type_map)
            action_type_idx = action_type_map[action_type]

            # Domain hash (simple numeric encoding)
            domain = node_data.get("domain", "")
            domain_hash = hash(domain) % 1000 / 1000.0  # Normalize to [0, 1]

            # Degree features
            in_deg = self._graph.in_degree(node)
            out_deg = self._graph.out_degree(node)

            # Data flow features
            num_produced = len(node_data.get("data_produced", []))
            num_consumed = len(node_data.get("data_consumed", []))

            node_features.append([
                action_type_idx,
                domain_hash,
                in_deg,
                out_deg,
                num_produced,
                num_consumed
            ])

        x = torch.tensor(node_features, dtype=torch.float)

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index_tensor,
            num_nodes=len(node_list)
        )

        # Store mappings for reference
        data.node_mapping = node_to_idx
        data.reverse_mapping = {idx: node for node, idx in node_to_idx.items()}
        data.action_type_map = action_type_map
        data.trajectory_id = self.trajectory.trajectory_id
        data.is_attack = self.trajectory.is_attack

        return data

    def to_dgl(self) -> "DGLGraph":
        """
        Convert to DGL graph for GNN processing.

        Returns:
            DGL graph object

        Raises:
            ImportError: If dgl is not installed
        """
        try:
            import dgl
            import torch
        except ImportError:
            raise ImportError(
                "DGL required. Install with: pip install dgl"
            )

        # Build node mapping
        node_list = list(self._graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        # Build edge lists
        src_nodes = []
        dst_nodes = []
        for source, target in self._graph.edges():
            src_nodes.append(node_to_idx[source])
            dst_nodes.append(node_to_idx[target])

        # Create DGL graph
        g = dgl.graph((src_nodes, dst_nodes), num_nodes=len(node_list))

        # Add node features (similar to PyG)
        action_type_map = {}
        features = []

        for node in node_list:
            node_data = self._graph.nodes[node]
            action_type = node_data.get("action_type")
            if hasattr(action_type, 'value'):
                action_type = action_type.value
            if action_type not in action_type_map:
                action_type_map[action_type] = len(action_type_map)

            features.append([
                action_type_map[action_type],
                hash(node_data.get("domain", "")) % 1000 / 1000.0,
                self._graph.in_degree(node),
                self._graph.out_degree(node),
                len(node_data.get("data_produced", [])),
                len(node_data.get("data_consumed", []))
            ])

        g.ndata['feat'] = torch.tensor(features, dtype=torch.float)

        return g

    def to_adjacency_matrix(self) -> "np.ndarray":
        """
        Convert to adjacency matrix (numpy array).

        Returns:
            2D numpy array where A[i,j] = 1 if edge from i to j
        """
        import numpy as np
        return nx.to_numpy_array(self._graph)

    # =========================================================================
    # Serialization
    # =========================================================================

    def export_to_json(self) -> Dict:
        """
        Export graph to JSON format.

        Returns:
            Dictionary representation of the graph
        """
        return {
            "trajectory_id": self.trajectory.trajectory_id,
            "num_nodes": len(self._graph.nodes()),
            "num_edges": len(self._graph.edges()),
            "nodes": [
                {
                    "action_id": node,
                    **{k: (v.value if hasattr(v, 'value') else v)
                       for k, v in self._graph.nodes[node].items()}
                }
                for node in self._graph.nodes()
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "edge_type": data.get("edge_type").value if data.get("edge_type") else None,
                    "metadata": data.get("metadata", {})
                }
                for source, target, data in self._graph.edges(data=True)
            ],
            "metrics": {
                "longest_path_length": self.longest_path_length(),
                "num_cross_domain_edges": len(self.get_cross_domain_edges()),
                "num_data_dependencies": len(self.get_edges_by_type(EdgeType.DATA_DEPENDENCY)),
                "num_trust_transfers": len(self.get_edges_by_type(EdgeType.TRUST_TRANSFER)),
                "num_state_enablements": len(self.get_edges_by_type(EdgeType.STATE_ENABLEMENT)),
                "bottleneck_nodes": self.get_bottleneck_nodes()
            }
        }

    @classmethod
    def load_from_json(cls, filepath: str) -> "CausalGraph":
        """
        Load a CausalGraph from a JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            CausalGraph instance
        """
        from causaltrace.models import Trajectory, Action, ActionType, State

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct minimal trajectory from saved nodes
        actions = []
        for node_data in data.get("nodes", []):
            action_type_val = node_data.get("action_type")
            if isinstance(action_type_val, str):
                action_type = ActionType.from_string(action_type_val)
            elif hasattr(action_type_val, 'value'):
                action_type = action_type_val
            else:
                action_type = ActionType.UNKNOWN

            action = Action(
                action_id=node_data.get("action_id", 0),
                action_type=action_type,
                target=node_data.get("target", ""),
                domain=node_data.get("domain"),
                timestamp=node_data.get("timestamp"),
                data_produced=node_data.get("data_produced", []),
                data_consumed=node_data.get("data_consumed", [])
            )
            actions.append(action)

        actions.sort(key=lambda a: a.action_id)

        trajectory = Trajectory(
            trajectory_id=data.get("trajectory_id", "loaded_graph"),
            source="loaded",
            task_description="Loaded from JSON",
            is_attack=False,
            actions=actions,
            initial_state=State()
        )

        graph = cls(trajectory)

        for edge_data in data.get("edges", []):
            edge_type_str = edge_data.get("edge_type")
            if edge_type_str:
                try:
                    edge_type = EdgeType(edge_type_str)
                except ValueError:
                    edge_type = EdgeType.DATA_DEPENDENCY
            else:
                edge_type = EdgeType.DATA_DEPENDENCY

            edge = CausalEdge(
                source_action_id=edge_data["source"],
                target_action_id=edge_data["target"],
                edge_type=edge_type,
                metadata=edge_data.get("metadata", {})
            )
            graph.add_edge(edge)

        return graph

    def export_to_graphviz(self, output_path: str, include_labels: bool = True):
        """
        Export graph to DOT format for visualization.

        Args:
            output_path: Path to save the DOT file
            include_labels: Whether to include action details as labels
        """
        viz_graph = nx.DiGraph()

        for node, data in self._graph.nodes(data=True):
            if include_labels:
                action_type = data.get('action_type', 'unknown')
                if hasattr(action_type, 'value'):
                    action_type = action_type.value
                label = f"{node}: {action_type}\n{data.get('domain', 'N/A')}"
            else:
                label = str(node)
            viz_graph.add_node(node, label=label)

        edge_colors = {
            EdgeType.DATA_DEPENDENCY: "blue",
            EdgeType.TRUST_TRANSFER: "red",
            EdgeType.STATE_ENABLEMENT: "green"
        }

        for source, target, data in self._graph.edges(data=True):
            edge_type = data.get("edge_type")
            color = edge_colors.get(edge_type, "black")
            label = edge_type.value if edge_type else ""
            viz_graph.add_edge(source, target, color=color, label=label)

        nx.drawing.nx_pydot.write_dot(viz_graph, output_path)

    # =========================================================================
    # Deprecated: Direct graph access (for migration)
    # =========================================================================

    @property
    def graph(self) -> nx.DiGraph:
        """
        DEPRECATED: Direct access to internal NetworkX graph.

        This property is provided for backward compatibility during migration.
        Use the encapsulated methods instead (iter_nodes, iter_edges, etc.).

        Will be removed in a future version.
        """
        import warnings
        warnings.warn(
            "Direct access to CausalGraph.graph is deprecated. "
            "Use encapsulated methods (iter_nodes, iter_edges, get_descendants, etc.) "
            "or adapters (to_pyg, to_dgl) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self._graph

    def __repr__(self) -> str:
        return f"CausalGraph(trajectory={self.trajectory.trajectory_id}, nodes={self.num_nodes()}, edges={self.num_edges()})"
