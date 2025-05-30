"""
Memory graph visualization and analysis system.
Uses NetworkX for graph operations and provides various visualization methods.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import community as community_louvain
from sklearn.manifold import TSNE
import torch
import logging

logger = logging.getLogger(__name__)

@dataclass
class MemoryNode:
    """Represents a node in the memory graph"""
    id: str
    content: str
    embedding: np.ndarray
    timestamp: datetime
    memory_type: str  # episodic, semantic, procedural, etc.
    importance: float
    emotional_valence: float
    consolidation_count: int
    metadata: Dict[str, Any]

@dataclass
class MemoryEdge:
    """Represents an edge between memories"""
    source_id: str
    target_id: str
    relationship_type: str  # association, causation, temporal, semantic
    strength: float
    created_at: datetime
    metadata: Dict[str, Any]

class MemoryGraph:
    """Main memory graph structure and operations"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_positions = {}
        self.communities = {}
        self.temporal_layers = {}
        
        # Visualization settings
        self.node_colors = {
            'episodic': '#FF6B6B',      # Red
            'semantic': '#4ECDC4',      # Teal
            'procedural': '#45B7D1',    # Blue
            'prospective': '#96CEB4',   # Green
            'emotional': '#DDA0DD',     # Plum
            'consolidated': '#FFD93D'   # Gold
        }
        
        # Graph metrics cache
        self._metrics_cache = {}
        self._cache_timestamp = None
        
    def add_memory_node(self, memory: MemoryNode):
        """Add a memory node to the graph"""
        self.graph.add_node(
            memory.id,
            content=memory.content,
            embedding=memory.embedding,
            timestamp=memory.timestamp,
            memory_type=memory.memory_type,
            importance=memory.importance,
            emotional_valence=memory.emotional_valence,
            consolidation_count=memory.consolidation_count,
            metadata=memory.metadata
        )
        
        # Invalidate cache
        self._invalidate_cache()
        
    def add_memory_edge(self, edge: MemoryEdge):
        """Add an edge between memories"""
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            relationship_type=edge.relationship_type,
            strength=edge.strength,
            created_at=edge.created_at,
            metadata=edge.metadata
        )
        
        # Invalidate cache
        self._invalidate_cache()
        
    def _invalidate_cache(self):
        """Invalidate metrics cache when graph changes"""
        self._metrics_cache = {}
        self._cache_timestamp = None
        
    def find_memory_clusters(self) -> Dict[str, List[str]]:
        """Find communities/clusters in the memory graph"""
        # Use Louvain algorithm for community detection
        partition = community_louvain.best_partition(
            self.graph.to_undirected(),
            weight='strength'
        )
        
        # Group nodes by community
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
            
        self.communities = communities
        return communities
        
    def get_memory_importance_flow(self) -> Dict[str, float]:
        """Calculate importance flow through the graph using PageRank"""
        # Use edge strengths as weights
        edge_weights = nx.get_edge_attributes(self.graph, 'strength')
        
        # Calculate PageRank
        pagerank = nx.pagerank(
            self.graph,
            weight='strength',
            alpha=0.85
        )
        
        return pagerank
        
    def find_memory_paths(self, source_id: str, target_id: str, max_length: int = 5) -> List[List[str]]:
        """Find all paths between two memories"""
        try:
            paths = list(nx.all_simple_paths(
                self.graph,
                source_id,
                target_id,
                cutoff=max_length
            ))
            return paths
        except nx.NetworkXNoPath:
            return []
            
    def get_memory_context(self, memory_id: str, radius: int = 2) -> nx.Graph:
        """Get subgraph around a specific memory"""
        # Get all nodes within radius
        if memory_id not in self.graph:
            return nx.Graph()
            
        nodes = set([memory_id])
        for _ in range(radius):
            new_nodes = set()
            for node in nodes:
                new_nodes.update(self.graph.predecessors(node))
                new_nodes.update(self.graph.successors(node))
            nodes.update(new_nodes)
            
        # Extract subgraph
        return self.graph.subgraph(nodes)
        
    def calculate_memory_centrality(self) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality measures"""
        if 'centrality' in self._metrics_cache:
            return self._metrics_cache['centrality']
            
        centrality = {
            'degree': nx.degree_centrality(self.graph),
            'betweenness': nx.betweenness_centrality(self.graph, weight='strength'),
            'closeness': nx.closeness_centrality(self.graph, distance='strength'),
            'eigenvector': nx.eigenvector_centrality(self.graph, weight='strength', max_iter=1000)
        }
        
        self._metrics_cache['centrality'] = centrality
        return centrality
        
    def identify_key_memories(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Identify the most important memories based on multiple metrics"""
        # Get centrality measures
        centrality = self.calculate_memory_centrality()
        
        # Combine metrics with weights
        combined_scores = {}
        for node in self.graph.nodes():
            score = (
                0.3 * centrality['degree'].get(node, 0) +
                0.3 * centrality['betweenness'].get(node, 0) +
                0.2 * centrality['eigenvector'].get(node, 0) +
                0.2 * self.graph.nodes[node].get('importance', 0)
            )
            combined_scores[node] = score
            
        # Sort and return top k
        return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
    def create_temporal_layers(self, time_window: timedelta) -> Dict[str, nx.Graph]:
        """Organize memories into temporal layers"""
        layers = {}
        
        # Group nodes by time windows
        for node in self.graph.nodes():
            timestamp = self.graph.nodes[node]['timestamp']
            layer_key = timestamp.strftime('%Y-%m-%d')
            
            if layer_key not in layers:
                layers[layer_key] = []
            layers[layer_key].append(node)
            
        # Create subgraphs for each layer
        self.temporal_layers = {
            key: self.graph.subgraph(nodes)
            for key, nodes in layers.items()
        }
        
        return self.temporal_layers
        
    def visualize_memory_graph(self, output_path: Optional[str] = None, 
                              highlight_nodes: Optional[List[str]] = None) -> plt.Figure:
        """Create a matplotlib visualization of the memory graph"""
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Calculate layout if not exists
        if not self.node_positions:
            self.node_positions = nx.spring_layout(
                self.graph,
                k=2,
                iterations=50,
                weight='strength'
            )
            
        # Prepare node colors and sizes
        node_colors = []
        node_sizes = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            color = self.node_colors.get(node_data['memory_type'], '#CCCCCC')
            
            # Highlight specific nodes
            if highlight_nodes and node in highlight_nodes:
                color = '#FFD700'  # Gold
                
            node_colors.append(color)
            node_sizes.append(300 * node_data['importance'])
            
        # Draw graph
        nx.draw_networkx_nodes(
            self.graph,
            self.node_positions,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.7,
            ax=ax
        )
        
        # Draw edges with varying thickness based on strength
        edges = self.graph.edges()
        weights = [self.graph[u][v]['strength'] for u, v in edges]
        
        nx.draw_networkx_edges(
            self.graph,
            self.node_positions,
            width=weights,
            alpha=0.5,
            ax=ax
        )
        
        # Add labels for important nodes
        important_nodes = dict(self.identify_key_memories(20))
        labels = {
            node: self.graph.nodes[node]['content'][:20] + '...'
            for node in important_nodes
        }
        
        nx.draw_networkx_labels(
            self.graph,
            self.node_positions,
            labels,
            font_size=8,
            ax=ax
        )
        
        plt.title("Memory Graph Visualization", fontsize=16)
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_interactive_visualization(self) -> go.Figure:
        """Create an interactive Plotly visualization"""
        # Calculate 3D layout using t-SNE on embeddings
        embeddings = []
        node_list = list(self.graph.nodes())
        
        for node in node_list:
            embeddings.append(self.graph.nodes[node]['embedding'])
            
        # Use t-SNE for 3D projection
        tsne = TSNE(n_components=3, random_state=42)
        positions_3d = tsne.fit_transform(np.array(embeddings))
        
        # Create edge traces
        edge_trace = []
        for edge in self.graph.edges():
            i = node_list.index(edge[0])
            j = node_list.index(edge[1])
            
            edge_trace.append(go.Scatter3d(
                x=[positions_3d[i, 0], positions_3d[j, 0], None],
                y=[positions_3d[i, 1], positions_3d[j, 1], None],
                z=[positions_3d[i, 2], positions_3d[j, 2], None],
                mode='lines',
                line=dict(
                    width=self.graph[edge[0]][edge[1]]['strength'] * 5,
                    color='#888'
                ),
                hoverinfo='none'
            ))
            
        # Create node trace
        node_trace = go.Scatter3d(
            x=positions_3d[:, 0],
            y=positions_3d[:, 1],
            z=positions_3d[:, 2],
            mode='markers+text',
            marker=dict(
                size=[self.graph.nodes[node]['importance'] * 20 for node in node_list],
                color=[self.node_colors.get(self.graph.nodes[node]['memory_type'], '#CCCCCC') 
                       for node in node_list],
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=[self.graph.nodes[node]['content'][:30] + '...' for node in node_list],
            textposition="top center",
            hoverinfo='text',
            hovertext=[
                f"ID: {node}<br>"
                f"Type: {self.graph.nodes[node]['memory_type']}<br>"
                f"Importance: {self.graph.nodes[node]['importance']:.2f}<br>"
                f"Content: {self.graph.nodes[node]['content'][:50]}..."
                for node in node_list
            ]
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title="Interactive Memory Graph",
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                zaxis=dict(showgrid=False, zeroline=False, visible=False)
            ),
            margin=dict(b=20, l=5, r=5, t=40),
            hovermode='closest'
        )
        
        return fig
        
    def analyze_memory_evolution(self, time_windows: List[timedelta]) -> Dict[str, Any]:
        """Analyze how the memory graph evolves over time"""
        evolution_metrics = {
            'graph_size': [],
            'density': [],
            'clustering_coefficient': [],
            'average_path_length': [],
            'modularity': []
        }
        
        current_time = datetime.now()
        
        for window in time_windows:
            # Get nodes within time window
            cutoff_time = current_time - window
            nodes_in_window = [
                node for node in self.graph.nodes()
                if self.graph.nodes[node]['timestamp'] > cutoff_time
            ]
            
            if len(nodes_in_window) < 2:
                continue
                
            # Extract subgraph
            subgraph = self.graph.subgraph(nodes_in_window)
            
            # Calculate metrics
            evolution_metrics['graph_size'].append(len(nodes_in_window))
            evolution_metrics['density'].append(nx.density(subgraph))
            
            if nx.is_connected(subgraph.to_undirected()):
                evolution_metrics['clustering_coefficient'].append(
                    nx.average_clustering(subgraph.to_undirected())
                )
                evolution_metrics['average_path_length'].append(
                    nx.average_shortest_path_length(subgraph.to_undirected())
                )
            else:
                evolution_metrics['clustering_coefficient'].append(0)
                evolution_metrics['average_path_length'].append(float('inf'))
                
            # Calculate modularity
            communities = community_louvain.best_partition(subgraph.to_undirected())
            modularity = community_louvain.modularity(communities, subgraph.to_undirected())
            evolution_metrics['modularity'].append(modularity)
            
        return evolution_metrics
        
    def export_to_json(self, filepath: str):
        """Export graph to JSON format"""
        data = nx.node_link_data(self.graph)
        
        # Convert numpy arrays and datetime objects to serializable format
        for node in data['nodes']:
            if 'embedding' in node:
                node['embedding'] = node['embedding'].tolist()
            if 'timestamp' in node:
                node['timestamp'] = node['timestamp'].isoformat()
                
        for link in data['links']:
            if 'created_at' in link:
                link['created_at'] = link['created_at'].isoformat()
                
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def import_from_json(self, filepath: str):
        """Import graph from JSON format"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Convert serialized data back to proper types
        for node in data['nodes']:
            if 'embedding' in node:
                node['embedding'] = np.array(node['embedding'])
            if 'timestamp' in node:
                node['timestamp'] = datetime.fromisoformat(node['timestamp'])
                
        for link in data['links']:
            if 'created_at' in link:
                link['created_at'] = datetime.fromisoformat(link['created_at'])
                
        self.graph = nx.node_link_graph(data)
        self._invalidate_cache()
        
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'num_components': nx.number_weakly_connected_components(self.graph),
            'average_degree': np.mean([d for n, d in self.graph.degree()]),
            'memory_type_distribution': {},
            'relationship_type_distribution': {}
        }
        
        # Count memory types
        for node in self.graph.nodes():
            mem_type = self.graph.nodes[node]['memory_type']
            stats['memory_type_distribution'][mem_type] = \
                stats['memory_type_distribution'].get(mem_type, 0) + 1
                
        # Count relationship types
        for edge in self.graph.edges():
            rel_type = self.graph[edge[0]][edge[1]]['relationship_type']
            stats['relationship_type_distribution'][rel_type] = \
                stats['relationship_type_distribution'].get(rel_type, 0) + 1
                
        return stats