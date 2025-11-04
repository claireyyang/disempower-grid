import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

@dataclass
class EnvironmentGraph:
    """Graph representation of an n×n gridworld environment"""
    
    def __init__(self, height: int = 7, width: int = 7):
        self.height = height
        self.width = width
        self.total_cells = height * width
        
        # Core spatial structure
        self.adjacency_matrix = np.zeros((self.total_cells, self.total_cells))
        self.cell_features = np.zeros((self.total_cells, 4))  # [is_wall, is_accessible, centrality, access_metric]
        
        # Entity placement as graph properties
        self.agent_nodes: List[int] = []
        self.goal_nodes: List[int] = []
        self.box_nodes: List[int] = []
        self.trap_nodes: List[int] = []
        self.wall_nodes: List[int] = []
        self.helper_nodes: List[int] = []
        
        # Structural features for analysis
        self._reachability_graph = None
        self._bottleneck_scores = None
        self._access_metric = None
        self._distance_matrix = None
    
    def coord_to_node(self, row: int, col: int) -> int:
        """Convert (row, col) coordinates to node ID"""
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError(f"Coordinates ({row}, {col}) out of bounds for {self.height}x{self.width} grid")
        return row * self.width + col
    
    def node_to_coord(self, node_id: int) -> Tuple[int, int]:
        """Convert node ID to (row, col) coordinates"""
        if not (0 <= node_id < self.total_cells):
            raise ValueError(f"Node ID {node_id} out of bounds for {self.total_cells} total cells")
        row = node_id // self.width
        col = node_id % self.width
        return (row, col)
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid neighboring coordinates (4-connected)"""
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.height and 0 <= new_col < self.width:
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def get_neighbor_nodes(self, node_id: int) -> List[int]:
        """Get valid neighboring node IDs"""
        row, col = self.node_to_coord(node_id)
        neighbor_coords = self.get_neighbors(row, col)
        return [self.coord_to_node(r, c) for r, c in neighbor_coords]
    
    def build_adjacency_matrix(self, wall_positions: List[Tuple[int, int]], box_positions: List[Tuple[int, int]]):
        """Build adjacency matrix from wall and box positions"""
        # Reset adjacency matrix
        self.adjacency_matrix = np.zeros((self.total_cells, self.total_cells))
        
        # Convert positions to node IDs
        self.wall_nodes = [self.coord_to_node(r, c) for r, c in wall_positions]
        self.box_nodes = [self.coord_to_node(r, c) for r, c in box_positions]
        
        # All blocked cells (walls + boxes)
        blocked_nodes = set(self.wall_nodes + self.box_nodes)
        
        # Mark cell features
        for node_id in range(self.total_cells):
            is_blocked = node_id in blocked_nodes
            self.cell_features[node_id, 0] = 1.0 if node_id in self.wall_nodes else 0.0  # is_wall
            self.cell_features[node_id, 1] = 0.0 if is_blocked else 1.0  # is_accessible (not blocked by wall or box)
        
        # Build adjacency connections between accessible cells only
        for node_id in range(self.total_cells):
            if node_id in blocked_nodes:
                continue  # Blocked cells have no connections
            
            neighbor_nodes = self.get_neighbor_nodes(node_id)
            for neighbor in neighbor_nodes:
                if neighbor not in blocked_nodes:  # Only connect to non-blocked neighbors
                    self.adjacency_matrix[node_id, neighbor] = 1.0
                    self.adjacency_matrix[neighbor, node_id] = 1.0  # Symmetric
        
        # Clear cached analysis results since graph structure changed
        self._reachability_graph = None
        self._bottleneck_scores = None
        self._access_metric = None
        self._distance_matrix = None
    
    def set_entity_positions(self, agent_pos: List[Tuple[int, int]], goal_pos: List[Tuple[int, int]], 
                           helper_pos: List[Tuple[int, int]], trap_pos: List[Tuple[int, int]]):
        """Set positions of entities (boxes handled in build_adjacency_matrix)"""
        self.agent_nodes = [self.coord_to_node(r, c) for r, c in agent_pos]
        self.goal_nodes = [self.coord_to_node(r, c) for r, c in goal_pos]
        self.helper_nodes = [self.coord_to_node(r, c) for r, c in helper_pos]
        self.trap_nodes = [self.coord_to_node(r, c) for r, c in trap_pos]
    
    def compute_distance_matrix(self):
        """Compute shortest path distances between all accessible nodes"""
        if self._distance_matrix is not None:
            return self._distance_matrix
        
        # Use scipy's shortest path on adjacency matrix
        graph = csr_matrix(self.adjacency_matrix)
        self._distance_matrix = shortest_path(graph, directed=False, return_predecessors=False)
        
        # Set distance to infinity for unreachable nodes
        self._distance_matrix[self._distance_matrix == np.inf] = -1  # Use -1 to indicate unreachable
        
        return self._distance_matrix
    
    def compute_reachability_graph(self):
        """Compute which nodes can reach which other nodes"""
        if self._reachability_graph is not None:
            return self._reachability_graph
        
        distance_matrix = self.compute_distance_matrix()
        # Reachable if finite distance exists
        self._reachability_graph = (distance_matrix >= 0) & (distance_matrix < np.inf)
        
        return self._reachability_graph
    
    def compute_betweenness_centrality(self):
        """Compute betweenness centrality for each node"""
        distance_matrix = self.compute_distance_matrix()
        centrality = np.zeros(self.total_cells)
        
        for node_id in range(self.total_cells):
            if self.cell_features[node_id, 1] == 0:  # Skip inaccessible nodes
                continue
            
            # Count how many shortest paths pass through this node
            paths_through_node = 0
            total_paths = 0
            
            for i in range(self.total_cells):
                for j in range(i + 1, self.total_cells):
                    if (self.cell_features[i, 1] == 0 or self.cell_features[j, 1] == 0 or 
                        distance_matrix[i, j] < 0):  # Skip inaccessible or unreachable pairs
                        continue
                    
                    total_paths += 1
                    # Check if shortest path from i to j passes through node_id
                    if (distance_matrix[i, node_id] + distance_matrix[node_id, j] == 
                        distance_matrix[i, j] and distance_matrix[i, j] > 0):
                        paths_through_node += 1
            
            centrality[node_id] = paths_through_node / max(total_paths, 1)
        
        return centrality
    
    def compute_bottleneck_scores(self):
        """Compute how much each node controls access (bottleneck strength)"""
        if self._bottleneck_scores is not None:
            return self._bottleneck_scores
        
        # Bottleneck score = betweenness centrality + connectivity reduction if removed
        centrality = self.compute_betweenness_centrality()
        bottleneck_scores = centrality.copy()
        
        # Add penalty for nodes whose removal significantly reduces connectivity
        reachability = self.compute_reachability_graph()
        baseline_connectivity = np.sum(reachability)
        
        for node_id in range(self.total_cells):
            if self.cell_features[node_id, 1] == 0:  # Skip inaccessible nodes
                continue
            
            # Temporarily remove this node and recompute connectivity
            temp_adj = self.adjacency_matrix.copy()
            temp_adj[node_id, :] = 0
            temp_adj[:, node_id] = 0
            
            temp_graph = csr_matrix(temp_adj)
            temp_distances = shortest_path(temp_graph, directed=False, return_predecessors=False)
            temp_reachability = (temp_distances >= 0) & (temp_distances < np.inf)
            new_connectivity = np.sum(temp_reachability)
            
            # Add bottleneck penalty based on connectivity loss
            connectivity_loss = (baseline_connectivity - new_connectivity) / max(baseline_connectivity, 1)
            bottleneck_scores[node_id] += connectivity_loss
        
        self._bottleneck_scores = bottleneck_scores
        return self._bottleneck_scores
    
    def compute_access_metric(self):
        if self._access_metric is not None:
            return self._access_metric
        
        access_metric = np.zeros(self.total_cells)
        distance_matrix = self.compute_distance_matrix()
        bottleneck_scores = self.compute_bottleneck_scores()
        
        for node_id in range(self.total_cells):
            if self.cell_features[node_id, 1] == 0:  # Skip inaccessible nodes
                continue
            
            # Access metric = accessibility to other nodes + low bottleneck dependency + choices available
            
            # 1. Accessibility: how many nodes can be reached and how easily
            reachable_nodes = np.sum(distance_matrix[node_id] >= 0)
            avg_distance = np.mean(distance_matrix[node_id][distance_matrix[node_id] >= 0])
            accessibility_score = reachable_nodes / (1 + avg_distance)
            
            # 2. Independence: low dependency on bottleneck nodes for mobility
            neighbor_bottlenecks = [bottleneck_scores[neighbor] for neighbor in self.get_neighbor_nodes(node_id) 
                                  if self.cell_features[neighbor, 1] == 1]
            independence_score = 1.0 - np.mean(neighbor_bottlenecks) if neighbor_bottlenecks else 1.0
            
            # 3. Choice diversity: number of different paths available
            num_neighbors = len([n for n in self.get_neighbor_nodes(node_id) 
                               if self.cell_features[n, 1] == 1])
            choice_score = min(num_neighbors / 4.0, 1.0)  # Normalize by max possible neighbors
            
            # Combined access metric
            access_metric[node_id] = 0.5 * accessibility_score + 0.3 * independence_score + 0.2 * choice_score
        
        self._access_metric = access_metric
        return self._access_metric
    
    def to_json_format(self) -> Dict:
        """Convert EnvironmentGraph to JSON format compatible with existing training code"""
        # Convert node IDs back to (row, col) coordinates
        agent_coords = [self.node_to_coord(node) for node in self.agent_nodes]
        goal_coords = [self.node_to_coord(node) for node in self.goal_nodes]
        helper_coords = [self.node_to_coord(node) for node in self.helper_nodes]
        box_coords = [self.node_to_coord(node) for node in self.box_nodes]
        trap_coords = [self.node_to_coord(node) for node in self.trap_nodes]
        wall_coords = [self.node_to_coord(node) for node in self.wall_nodes]
        
        return {
            "agent_pos": [[int(r), int(c)] for r, c in agent_coords],
            "goal_pos": [[int(r), int(c)] for r, c in goal_coords],
            "helper_pos": [[int(r), int(c)] for r, c in helper_coords],
            "box_pos": [[int(r), int(c)] for r, c in box_coords],
            "trap_pos": [[int(r), int(c)] for r, c in trap_coords],
            "wall_pos": [[int(r), int(c)] for r, c in wall_coords]
        }
    
    @classmethod
    def from_json_format(cls, json_data: Dict, height: int = 7, width: int = 7):
        """Create EnvironmentGraph from JSON format"""
        env_graph = cls(height, width)
        
        # Extract positions from JSON
        wall_positions = [tuple(pos) for pos in json_data.get("wall_pos", [])]
        box_positions = [tuple(pos) for pos in json_data.get("box_pos", [])]
        agent_positions = [tuple(pos) for pos in json_data.get("agent_pos", [])]
        goal_positions = [tuple(pos) for pos in json_data.get("goal_pos", [])]
        helper_positions = [tuple(pos) for pos in json_data.get("helper_pos", [])]
        trap_positions = [tuple(pos) for pos in json_data.get("trap_pos", [])]
        
        # Build the graph
        env_graph.build_adjacency_matrix(wall_positions, box_positions)
        env_graph.set_entity_positions(agent_positions, goal_positions, helper_positions, trap_positions)
        
        return env_graph
    
    def get_accessible_positions(self) -> List[Tuple[int, int]]:
        """Get all accessible (non-blocked) positions as coordinates"""
        accessible_positions = []
        for node_id in range(self.total_cells):
            if self.cell_features[node_id, 1] == 1.0:  # is_accessible
                accessible_positions.append(self.node_to_coord(node_id))
        return accessible_positions
    
    def get_high_access_positions(self, threshold: float = 0.7) -> List[Tuple[int, int]]:
        """Get positions with high access potential"""
        access_scores = self.compute_access_metric()
        high_access_nodes = [node_id for node_id in range(self.total_cells) 
                                if access_scores[node_id] >= threshold and 
                                   self.cell_features[node_id, 1] == 1.0]
        return [self.node_to_coord(node_id) for node_id in high_access_nodes]
    
    def get_low_access_positions(self, threshold: float = 0.3) -> List[Tuple[int, int]]:
        """Get positions with low access potential"""  
        access_scores = self.compute_access_metric()
        low_access_nodes = [node_id for node_id in range(self.total_cells) 
                               if access_scores[node_id] <= threshold and 
                                  self.cell_features[node_id, 1] == 1.0]
        return [self.node_to_coord(node_id) for node_id in low_access_nodes]
    
    def get_bottleneck_positions(self, threshold: float = 0.5) -> List[Tuple[int, int]]:
        """Get positions that are bottlenecks (high control over movement)"""
        bottleneck_scores = self.compute_bottleneck_scores()
        bottleneck_nodes = [node_id for node_id in range(self.total_cells)
                          if bottleneck_scores[node_id] >= threshold and 
                             self.cell_features[node_id, 1] == 1.0]
        return [self.node_to_coord(node_id) for node_id in bottleneck_nodes]