"""
Consolidation Network Module
Implements neural network-based memory consolidation mechanisms inspired by
hippocampal-neocortical memory transfer and synaptic consolidation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json

# Additional imports
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


@dataclass
class MemoryTrace:
    """Represents a neural trace of a memory"""
    id: str
    content_embedding: torch.Tensor
    context_embedding: torch.Tensor
    timestamp: datetime
    strength: float = 1.0
    activation_count: int = 0
    last_activation: Optional[datetime] = None
    consolidation_state: str = "labile"  # labile, consolidating, stable
    associated_traces: Set[str] = field(default_factory=set)


@dataclass
class ConsolidationPattern:
    """Represents a pattern discovered during consolidation"""
    pattern_embedding: torch.Tensor
    supporting_traces: List[str]
    emergence_time: datetime
    stability_score: float
    abstraction_level: int  # 0 = concrete, higher = more abstract


class HippocampalBuffer(nn.Module):
    """
    Simulates hippocampal fast learning system for temporary storage
    and initial encoding of episodic memories.
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 768,
        buffer_size: int = 1000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        
        # Pattern separation network (dentate gyrus-like)
        self.pattern_separator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Autoassociative network (CA3-like)
        self.autoassociator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Pattern completion network (CA1-like)
        self.pattern_completion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Memory buffer
        self.memory_buffer = deque(maxlen=buffer_size)
        self.buffer_embeddings = None
    
    def forward(self, x: torch.Tensor, store: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through hippocampal circuits.
        Returns: (separated_pattern, reconstructed_input)
        """
        
        # Pattern separation
        separated = self.pattern_separator(x)
        
        # Autoassociation for learning
        associated = self.autoassociator(separated)
        
        # Pattern completion for retrieval
        reconstructed = self.pattern_completion(associated)
        
        # Store in buffer if requested
        if store:
            self.memory_buffer.append({
                "input": x.detach(),
                "separated": separated.detach(),
                "timestamp": datetime.now()
            })
        
        return separated, reconstructed
    
    def recall_similar(
        self,
        query: torch.Tensor,
        k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Recall similar memories from buffer"""
        
        if not self.memory_buffer:
            return []
        
        # Separate query pattern
        query_separated, _ = self.forward(query, store=False)
        
        # Compare with buffer
        similarities = []
        
        for memory in self.memory_buffer:
            similarity = F.cosine_similarity(
                query_separated,
                memory["separated"],
                dim=-1
            ).mean().item()
            
            if similarity > threshold:
                similarities.append({
                    "memory": memory,
                    "similarity": similarity
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:k]


class NeocorticalNetwork(nn.Module):
    """
    Simulates neocortical slow learning system for long-term
    semantic memory storage and abstraction.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers for hierarchical processing
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        # Hierarchical abstraction layers
        self.abstraction_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(3)  # 3 levels of abstraction
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
        # Learned prototypes (semantic concepts)
        self.num_prototypes = 100
        self.prototypes = nn.Parameter(
            torch.randn(self.num_prototypes, hidden_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_abstractions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Process input through neocortical hierarchy.
        Returns dictionary with processed representations.
        """
        
        # Project to hidden dimension
        h = self.input_projection(x)
        
        # Add positional encoding if sequence
        if len(h.shape) == 3:  # Batch x Seq x Hidden
            h = h + self._positional_encoding(h)
        
        # Process through transformer
        transformed = self.transformer(h)
        
        # Generate abstractions
        abstractions = []
        current = transformed
        
        for layer in self.abstraction_layers:
            current = layer(current)
            abstractions.append(current)
        
        # Compare with learned prototypes
        prototype_similarities = F.cosine_similarity(
            current.unsqueeze(1),  # Add prototype dimension
            self.prototypes.unsqueeze(0),  # Add batch dimension
            dim=-1
        )
        
        # Get closest prototypes
        top_prototypes = torch.topk(prototype_similarities, k=5, dim=-1)
        
        # Output projection
        output = self.output_projection(current)
        
        results = {
            "output": output,
            "transformed": transformed,
            "prototype_indices": top_prototypes.indices,
            "prototype_similarities": top_prototypes.values
        }
        
        if return_abstractions:
            results["abstractions"] = abstractions
        
        return results
    
    def _positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Generate positional encoding for sequences"""
        
        seq_len = x.size(1)
        d_model = x.size(2)
        
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(np.log(10000.0) / d_model))
        
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0).to(x.device)
    
    def update_prototypes(self, new_patterns: torch.Tensor):
        """Update learned prototypes with new patterns"""
        
        # Simple competitive learning
        with torch.no_grad():
            for pattern in new_patterns:
                # Find closest prototype
                similarities = F.cosine_similarity(
                    pattern.unsqueeze(0),
                    self.prototypes,
                    dim=1
                )
                
                closest_idx = similarities.argmax()
                
                # Update closest prototype
                learning_rate = 0.01
                self.prototypes[closest_idx] = (
                    (1 - learning_rate) * self.prototypes[closest_idx] +
                    learning_rate * pattern
                )


class ConsolidationNetwork(nn.Module):
    """
    Main consolidation network that orchestrates memory transfer
    from hippocampal to neocortical systems.
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        hippocampal_hidden: int = 768,
        neocortical_hidden: int = 1024,
        consolidation_rate: float = 0.1,
        replay_temperature: float = 0.8
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.consolidation_rate = consolidation_rate
        self.replay_temperature = replay_temperature
        
        # Initialize subsystems
        self.hippocampus = HippocampalBuffer(
            input_dim=embedding_dim,
            hidden_dim=hippocampal_hidden
        )
        
        self.neocortex = NeocorticalNetwork(
            input_dim=hippocampal_hidden,
            hidden_dim=neocortical_hidden
        )
        
        # Consolidation pathway
        self.consolidation_gate = nn.Sequential(
            nn.Linear(hippocampal_hidden * 2, hippocampal_hidden),
            nn.ReLU(),
            nn.Linear(hippocampal_hidden, 1),
            nn.Sigmoid()
        )
        
        # Replay generator for offline consolidation
        self.replay_generator = nn.Sequential(
            nn.Linear(hippocampal_hidden, hippocampal_hidden * 2),
            nn.ReLU(),
            nn.Linear(hippocampal_hidden * 2, hippocampal_hidden),
            nn.Tanh()
        )
        
        # Memory traces storage
        self.memory_traces: Dict[str, MemoryTrace] = {}
        self.consolidation_patterns: List[ConsolidationPattern] = []
        
        # Consolidation statistics
        self.consolidation_stats = {
            "traces_processed": 0,
            "patterns_discovered": 0,
            "successful_transfers": 0,
            "replay_cycles": 0
        }
    
    def forward(
        self,
        memory_embedding: torch.Tensor,
        memory_id: str,
        consolidate: bool = True
    ) -> Dict[str, Any]:
        """Process memory through consolidation network"""
        
        # Encode in hippocampus
        hip_encoding, reconstructed = self.hippocampus(memory_embedding)
        
        # Calculate reconstruction error
        reconstruction_error = F.mse_loss(reconstructed, memory_embedding)
        
        # Determine if ready for consolidation
        if consolidate:
            consolidation_score = self._calculate_consolidation_readiness(
                hip_encoding,
                reconstruction_error
            )
            
            if consolidation_score > 0.5:
                # Transfer to neocortex
                neo_results = self.neocortex(hip_encoding.unsqueeze(0))
                
                # Create or update memory trace
                self._update_memory_trace(
                    memory_id,
                    hip_encoding,
                    neo_results["transformed"].squeeze(0)
                )
                
                self.consolidation_stats["successful_transfers"] += 1
            else:
                neo_results = None
        else:
            consolidation_score = 0.0
            neo_results = None
        
        self.consolidation_stats["traces_processed"] += 1
        
        return {
            "hippocampal_encoding": hip_encoding,
            "reconstruction": reconstructed,
            "reconstruction_error": reconstruction_error.item(),
            "consolidation_score": consolidation_score,
            "neocortical_results": neo_results
        }
    
    def _calculate_consolidation_readiness(
        self,
        encoding: torch.Tensor,
        reconstruction_error: torch.Tensor
    ) -> float:
        """Calculate if memory is ready for consolidation"""
        
        # Low reconstruction error means stable representation
        error_score = 1.0 - torch.sigmoid(reconstruction_error * 10).item()
        
        # Check encoding stability
        if hasattr(self, '_last_encoding'):
            stability = F.cosine_similarity(
                encoding,
                self._last_encoding,
                dim=0
            ).item()
        else:
            stability = 0.5
        
        self._last_encoding = encoding.detach()
        
        # Combine factors
        readiness = error_score * 0.6 + stability * 0.4
        
        return readiness
    
    def _update_memory_trace(
        self,
        memory_id: str,
        hippocampal_encoding: torch.Tensor,
        neocortical_encoding: torch.Tensor
    ):
        """Update or create memory trace"""
        
        if memory_id in self.memory_traces:
            trace = self.memory_traces[memory_id]
            trace.context_embedding = neocortical_encoding
            trace.strength = min(1.0, trace.strength + 0.1)
            trace.consolidation_state = "consolidating"
            trace.activation_count += 1
            trace.last_activation = datetime.now()
        else:
            trace = MemoryTrace(
                id=memory_id,
                content_embedding=hippocampal_encoding,
                context_embedding=neocortical_encoding,
                timestamp=datetime.now()
            )
            self.memory_traces[memory_id] = trace
    
    async def replay_consolidation(
        self,
        num_replays: int = 50,
        batch_size: int = 10
    ):
        """
        Perform offline consolidation through memory replay.
        Simulates sleep-dependent memory consolidation.
        """
        
        logger.info(f"Starting replay consolidation with {num_replays} cycles")
        
        for cycle in range(num_replays):
            # Select memories for replay
            replay_batch = self._select_replay_batch(batch_size)
            
            if not replay_batch:
                continue
            
            # Generate replay patterns
            replay_patterns = []
            
            for trace in replay_batch:
                # Add noise for exploration
                noise = torch.randn_like(trace.content_embedding) * self.replay_temperature
                replayed = self.replay_generator(trace.content_embedding + noise)
                replay_patterns.append(replayed)
            
            # Process replayed patterns
            replay_tensor = torch.stack(replay_patterns)
            
            # Discover patterns in replay
            patterns = await self._discover_patterns(replay_tensor)
            
            if patterns:
                self.consolidation_patterns.extend(patterns)
                self.consolidation_stats["patterns_discovered"] += len(patterns)
            
            # Update neocortical representations
            neo_results = self.neocortex(replay_tensor)
            
            # Update prototypes with discovered patterns
            if patterns:
                pattern_embeddings = torch.stack([p.pattern_embedding for p in patterns])
                self.neocortex.update_prototypes(pattern_embeddings)
            
            # Strengthen consolidated traces
            for i, trace in enumerate(replay_batch):
                trace.strength *= 1.05  # Strengthen by 5%
                if trace.strength > 0.8:
                    trace.consolidation_state = "stable"
            
            self.consolidation_stats["replay_cycles"] += 1
            
            # Brief pause to simulate time
            await asyncio.sleep(0.01)
        
        logger.info(f"Replay consolidation complete. Discovered {len(patterns)} new patterns")
    
    def _select_replay_batch(self, batch_size: int) -> List[MemoryTrace]:
        """Select memories for replay based on importance and recency"""
        
        if not self.memory_traces:
            return []
        
        # Calculate replay priority
        traces_with_priority = []
        
        for trace in self.memory_traces.values():
            # Prioritize recent and strong memories
            recency = 1.0 / (1.0 + (datetime.now() - trace.timestamp).total_seconds() / 3600)
            priority = trace.strength * 0.6 + recency * 0.4
            
            # Boost priority for consolidating memories
            if trace.consolidation_state == "consolidating":
                priority *= 1.5
            
            traces_with_priority.append((trace, priority))
        
        # Sort by priority
        traces_with_priority.sort(key=lambda x: x[1], reverse=True)
        
        # Select top traces
        selected = [trace for trace, _ in traces_with_priority[:batch_size]]
        
        return selected
    
    async def _discover_patterns(
        self,
        replay_tensor: torch.Tensor
    ) -> List[ConsolidationPattern]:
        """Discover patterns in replayed memories"""
        
        patterns = []
        
        # Simple pattern detection through clustering
        if replay_tensor.size(0) < 3:
            return patterns
        
        # Calculate pairwise similarities
        similarities = F.cosine_similarity(
            replay_tensor.unsqueeze(1),
            replay_tensor.unsqueeze(0),
            dim=-1
        )
        
        # Find highly similar groups
        threshold = 0.8
        pattern_groups = []
        
        for i in range(replay_tensor.size(0)):
            similar_indices = (similarities[i] > threshold).nonzero().squeeze(-1)
            
            if len(similar_indices) >= 3:  # Minimum pattern size
                pattern_groups.append(similar_indices.tolist())
        
        # Merge overlapping groups
        merged_groups = self._merge_overlapping_groups(pattern_groups)
        
        # Create patterns from groups
        for group in merged_groups:
            if len(group) >= 3:
                # Calculate pattern as mean of group
                pattern_embedding = replay_tensor[group].mean(dim=0)
                
                pattern = ConsolidationPattern(
                    pattern_embedding=pattern_embedding,
                    supporting_traces=group,
                    emergence_time=datetime.now(),
                    stability_score=0.5,
                    abstraction_level=1
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _merge_overlapping_groups(self, groups: List[List[int]]) -> List[List[int]]:
        """Merge overlapping groups of indices"""
        
        if not groups:
            return []
        
        # Convert to sets for easier manipulation
        group_sets = [set(g) for g in groups]
        
        merged = []
        
        while group_sets:
            current = group_sets.pop(0)
            
            # Find all overlapping groups
            i = 0
            while i < len(group_sets):
                if current & group_sets[i]:  # Intersection exists
                    current |= group_sets.pop(i)  # Union
                else:
                    i += 1
            
            merged.append(list(current))
        
        return merged
    
    def consolidation_strength(self, memory_id: str) -> float:
        """Get consolidation strength of a memory"""
        
        if memory_id not in self.memory_traces:
            return 0.0
        
        trace = self.memory_traces[memory_id]
        
        # Factor in multiple aspects
        strength_factors = {
            "base_strength": trace.strength,
            "activation_factor": min(1.0, trace.activation_count * 0.1),
            "state_factor": {
                "labile": 0.3,
                "consolidating": 0.6,
                "stable": 1.0
            }.get(trace.consolidation_state, 0.5),
            "association_factor": min(1.0, len(trace.associated_traces) * 0.05)
        }
        
        # Weighted combination
        weights = {
            "base_strength": 0.4,
            "activation_factor": 0.2,
            "state_factor": 0.3,
            "association_factor": 0.1
        }
        
        total_strength = sum(
            strength_factors[factor] * weights[factor]
            for factor in strength_factors
        )
        
        return total_strength
    
    def get_memory_associations(
        self,
        memory_id: str,
        min_strength: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Get associated memories based on consolidation patterns"""
        
        if memory_id not in self.memory_traces:
            return []
        
        trace = self.memory_traces[memory_id]
        associations = []
        
        # Direct associations
        for associated_id in trace.associated_traces:
            if associated_id in self.memory_traces:
                strength = self._calculate_association_strength(memory_id, associated_id)
                if strength >= min_strength:
                    associations.append((associated_id, strength))
        
        # Pattern-based associations
        for pattern in self.consolidation_patterns:
            if memory_id in [str(idx) for idx in pattern.supporting_traces]:
                # Find other memories in same pattern
                for other_idx in pattern.supporting_traces:
                    other_id = str(other_idx)
                    if other_id != memory_id and other_id in self.memory_traces:
                        strength = pattern.stability_score
                        if strength >= min_strength:
                            associations.append((other_id, strength))
        
        # Remove duplicates and sort by strength
        associations = list(set(associations))
        associations.sort(key=lambda x: x[1], reverse=True)
        
        return associations
    
    def _calculate_association_strength(
        self,
        memory1_id: str,
        memory2_id: str
    ) -> float:
        """Calculate strength of association between two memories"""
        
        trace1 = self.memory_traces.get(memory1_id)
        trace2 = self.memory_traces.get(memory2_id)
        
        if not trace1 or not trace2:
            return 0.0
        
        # Embedding similarity
        similarity = F.cosine_similarity(
            trace1.content_embedding,
            trace2.content_embedding,
            dim=0
        ).item()
        
        # Temporal proximity factor
        time_diff = abs((trace1.timestamp - trace2.timestamp).total_seconds())
        temporal_factor = np.exp(-time_diff / 3600)  # Decay over hours
        
        # Co-activation factor
        co_activation = 0.0
        if trace1.last_activation and trace2.last_activation:
            activation_diff = abs(
                (trace1.last_activation - trace2.last_activation).total_seconds()
            )
            co_activation = np.exp(-activation_diff / 300)  # 5-minute window
        
        # Combine factors
        strength = (
            similarity * 0.5 +
            temporal_factor * 0.3 +
            co_activation * 0.2
        )
        
        return strength
    
    def visualize_memory_landscape(
        self,
        save_path: Optional[str] = None,
        show_patterns: bool = True
    ):
        """Visualize the memory consolidation landscape"""
        
        if not self.memory_traces:
            logger.warning("No memory traces to visualize")
            return
        
        # Extract embeddings
        embeddings = []
        labels = []
        colors = []
        
        for trace in self.memory_traces.values():
            embeddings.append(trace.content_embedding.detach().cpu().numpy())
            labels.append(trace.id[:8])  # First 8 chars of ID
            
            # Color by consolidation state
            color_map = {
                "labile": "red",
                "consolidating": "yellow",
                "stable": "green"
            }
            colors.append(color_map.get(trace.consolidation_state, "gray"))
        
        embeddings = np.array(embeddings)
        
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot memory traces
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=colors,
            s=100,
            alpha=0.6,
            edgecolors='black'
        )
        
        # Add labels
        for i, label in enumerate(labels):
            plt.annotate(
                label,
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=8,
                alpha=0.7
            )
        
        # Plot patterns if requested
        if show_patterns and self.consolidation_patterns:
            # Add pattern regions
            for pattern in self.consolidation_patterns[-10:]:  # Last 10 patterns
                # Get positions of supporting traces
                pattern_positions = []
                
                for idx in pattern.supporting_traces:
                    if idx < len(embeddings_2d):
                        pattern_positions.append(embeddings_2d[idx])
                
                if pattern_positions:
                    pattern_positions = np.array(pattern_positions)
                    
                    # Draw convex hull around pattern
                    from scipy.spatial import ConvexHull
                    
                    if len(pattern_positions) >= 3:
                        hull = ConvexHull(pattern_positions)
                        
                        for simplex in hull.simplices:
                            plt.plot(
                                pattern_positions[simplex, 0],
                                pattern_positions[simplex, 1],
                                'b-',
                                alpha=0.2
                            )
        
        plt.title("Memory Consolidation Landscape")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Labile'),
            Patch(facecolor='yellow', label='Consolidating'),
            Patch(facecolor='green', label='Stable')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get comprehensive consolidation statistics"""
        
        stats = self.consolidation_stats.copy()
        
        # Add memory trace statistics
        if self.memory_traces:
            state_counts = {"labile": 0, "consolidating": 0, "stable": 0}
            total_strength = 0.0
            total_associations = 0
            
            for trace in self.memory_traces.values():
                state_counts[trace.consolidation_state] += 1
                total_strength += trace.strength
                total_associations += len(trace.associated_traces)
            
            stats.update({
                "total_traces": len(self.memory_traces),
                "state_distribution": state_counts,
                "average_strength": total_strength / len(self.memory_traces),
                "average_associations": total_associations / len(self.memory_traces),
                "pattern_count": len(self.consolidation_patterns)
            })
        
        return stats
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        
        checkpoint = {
            "hippocampus_state": self.hippocampus.state_dict(),
            "neocortex_state": self.neocortex.state_dict(),
            "consolidation_gate_state": self.consolidation_gate.state_dict(),
            "replay_generator_state": self.replay_generator.state_dict(),
            "memory_traces": {
                k: {
                    "id": v.id,
                    "timestamp": v.timestamp.isoformat(),
                    "strength": v.strength,
                    "activation_count": v.activation_count,
                    "consolidation_state": v.consolidation_state
                }
                for k, v in self.memory_traces.items()
            },
            "consolidation_stats": self.consolidation_stats
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved consolidation network checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        
        checkpoint = torch.load(path)
        
        self.hippocampus.load_state_dict(checkpoint["hippocampus_state"])
        self.neocortex.load_state_dict(checkpoint["neocortex_state"])
        self.consolidation_gate.load_state_dict(checkpoint["consolidation_gate_state"])
        self.replay_generator.load_state_dict(checkpoint["replay_generator_state"])
        self.consolidation_stats = checkpoint["consolidation_stats"]
        
        logger.info(f"Loaded consolidation network checkpoint from {path}")