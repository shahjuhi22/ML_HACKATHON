import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from typing import Dict, List, Set, Tuple

class RoadNetworkGraph:
    """
    Directed Road Network Graph constructed from .nodes and .arcs.
    """
    def __init__(self, nodes: np.ndarray, arcs: np.ndarray):
        self.nodes = nodes
        self.arcs = arcs
        self.num_nodes = len(nodes)
        self.num_arcs = len(arcs)

        # Validate node indices
        valid = (
            (arcs[:, 0] >= 0) & (arcs[:, 0] < self.num_nodes) &
            (arcs[:, 1] >= 0) & (arcs[:, 1] < self.num_nodes)
        )
        self.valid_arcs = arcs[valid]

        # Adjacency structures
        self.out_arcs: Dict[int, List[int]] = {i: [] for i in range(self.num_nodes)}
        self.in_arcs: Dict[int, List[int]] = {i: [] for i in range(self.num_nodes)}
        for arc_id, (u, v) in enumerate(arcs):
            if valid[arc_id]:
                self.out_arcs[u].append(arc_id)
                self.in_arcs[v].append(arc_id)

        # Sparse adjacency matrix for components and path analysis
        self._adj_matrix = coo_matrix(
            (np.ones(len(self.valid_arcs)), (self.valid_arcs[:, 0], self.valid_arcs[:, 1])),
            shape=(self.num_nodes, self.num_nodes)
        ).tocsr()

    def get_connected_components(self, connection: str = "weak") -> Tuple[int, np.ndarray]:
        """Compute connected components of the road network."""
        return connected_components(self._adj_matrix, directed=True, connection=connection)

    def are_connected(self, arc_prev: int, arc_curr: int) -> bool:
        """
        Check if arc_curr immediately connects from arc_prev:
        i.e., arc_prev's to_node == arc_curr's from_node.
        """
        if arc_prev < 0 or arc_curr < 0:
            return False
        return self.arcs[arc_prev, 1] == self.arcs[arc_curr, 0]

    def can_transition(self, arc_prev: int, arc_curr: int) -> bool:
        """
        Vehicle can either stay on the same arc, or transition directly to an outgoing connected arc.
        """
        if arc_prev == arc_curr:
            return True
        return bool(self.are_connected(arc_prev, arc_curr))

    def get_candidate_successors(self, arc_id: int) -> List[int]:
        """Get all arcs directly reachable from the end of the specified arc."""
        if arc_id < 0 or arc_id >= self.num_arcs:
            return []
        to_node = self.arcs[arc_id, 1]
        return self.out_arcs.get(to_node, [])
