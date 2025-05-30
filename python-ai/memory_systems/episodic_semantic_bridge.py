"""
Episodic-Semantic Bridge Module
Handles the transformation of episodic memories into semantic knowledge,
implementing theories of memory consolidation and abstraction.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, Counter
import json
from abc import ABC, abstractmethod

# NLP and ML imports
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from transformers import pipeline


logger = logging.getLogger(__name__)


@dataclass
class EpisodicMemory:
    """Represents a single episodic memory"""
    id: str
    content: str
    context: Dict[str, Any]
    timestamp: datetime
    emotional_valence: float
    vividness: float = 1.0  # Decays over time
    rehearsal_count: int = 0
    associations: List[str] = field(default_factory=list)
    
    def decay(self, decay_rate: float = 0.01):
        """Apply time-based decay to vividness"""
        time_elapsed = (datetime.now() - self.timestamp).days
        self.vividness = self.vividness * np.exp(-decay_rate * time_elapsed)


@dataclass
class SemanticConcept:
    """Represents abstract semantic knowledge"""
    id: str
    concept: str
    description: str
    source_episodes: List[str] = field(default_factory=list)
    confidence: float = 0.5
    frequency: int = 1
    attributes: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, float] = field(default_factory=dict)  # concept_id -> strength
    
    def strengthen(self, amount: float = 0.1):
        """Strengthen concept confidence"""
        self.confidence = min(1.0, self.confidence + amount)
        self.frequency += 1


@dataclass
class ConsolidationCandidate:
    """Represents a group of memories ready for consolidation"""
    memories: List[EpisodicMemory]
    theme: str
    coherence_score: float
    importance_score: float
    emotional_consistency: float


class AbstractionStrategy(ABC):
    """Abstract base class for different abstraction strategies"""
    
    @abstractmethod
    async def abstract(
        self,
        memories: List[EpisodicMemory]
    ) -> Optional[SemanticConcept]:
        """Abstract episodic memories into semantic concept"""
        pass


class ThematicAbstraction(AbstractionStrategy):
    """Abstracts memories based on common themes"""
    
    def __init__(self, theme_extractor):
        self.theme_extractor = theme_extractor
    
    async def abstract(
        self,
        memories: List[EpisodicMemory]
    ) -> Optional[SemanticConcept]:
        """Extract common theme from memories"""
        
        # Extract themes from each memory
        themes = []
        for memory in memories:
            theme = await self.theme_extractor.extract(memory.content)
            themes.extend(theme)
        
        # Find most common theme
        theme_counts = Counter(themes)
        if not theme_counts:
            return None
        
        dominant_theme = theme_counts.most_common(1)[0][0]
        
        # Create semantic concept
        concept = SemanticConcept(
            id=f"concept_{dominant_theme}_{datetime.now().timestamp()}",
            concept=dominant_theme,
            description=f"Abstracted knowledge about {dominant_theme}",
            source_episodes=[m.id for m in memories],
            confidence=0.5 + (len(memories) * 0.05),  # More episodes = higher confidence
            attributes={
                "abstraction_method": "thematic",
                "theme_frequency": theme_counts[dominant_theme]
            }
        )
        
        return concept


class PatternAbstraction(AbstractionStrategy):
    """Abstracts memories based on recurring patterns"""
    
    def __init__(self, pattern_detector):
        self.pattern_detector = pattern_detector
    
    async def abstract(
        self,
        memories: List[EpisodicMemory]
    ) -> Optional[SemanticConcept]:
        """Extract patterns from memories"""
        
        # Detect patterns across memories
        patterns = await self.pattern_detector.detect_patterns(
            [m.content for m in memories]
        )
        
        if not patterns:
            return None
        
        # Use strongest pattern
        strongest_pattern = patterns[0]
        
        concept = SemanticConcept(
            id=f"pattern_{strongest_pattern['type']}_{datetime.now().timestamp()}",
            concept=strongest_pattern['name'],
            description=strongest_pattern['description'],
            source_episodes=[m.id for m in memories],
            confidence=strongest_pattern['confidence'],
            attributes={
                "abstraction_method": "pattern",
                "pattern_type": strongest_pattern['type'],
                "pattern_strength": strongest_pattern['strength']
            }
        )
        
        return concept


class EmotionalAbstraction(AbstractionStrategy):
    """Abstracts memories based on emotional significance"""
    
    async def abstract(
        self,
        memories: List[EpisodicMemory]
    ) -> Optional[SemanticConcept]:
        """Create concept from emotionally significant memories"""
        
        # Calculate emotional profile
        avg_valence = np.mean([m.emotional_valence for m in memories])
        emotional_variance = np.var([m.emotional_valence for m in memories])
        
        # Determine emotional category
        if avg_valence > 0.5:
            emotion_category = "positive_experiences"
        elif avg_valence < -0.5:
            emotion_category = "challenging_experiences"
        else:
            emotion_category = "neutral_observations"
        
        # Create concept based on emotional learning
        concept = SemanticConcept(
            id=f"emotional_{emotion_category}_{datetime.now().timestamp()}",
            concept=f"Learned from {emotion_category}",
            description=f"Wisdom gained from {emotion_category}",
            source_episodes=[m.id for m in memories],
            confidence=0.6 + (1 - emotional_variance) * 0.2,  # Consistent emotion = higher confidence
            attributes={
                "abstraction_method": "emotional",
                "average_valence": avg_valence,
                "emotional_consistency": 1 - emotional_variance
            }
        )
        
        return concept


class EpisodicSemanticBridge:
    """
    Main class for transforming episodic memories into semantic knowledge
    through various consolidation and abstraction mechanisms.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        consolidation_threshold: float = 0.7,
        min_episodes_for_consolidation: int = 3,
        enable_sleep_consolidation: bool = True,
        enable_rehearsal_consolidation: bool = True
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.consolidation_threshold = consolidation_threshold
        self.min_episodes_for_consolidation = min_episodes_for_consolidation
        self.enable_sleep_consolidation = enable_sleep_consolidation
        self.enable_rehearsal_consolidation = enable_rehearsal_consolidation
        
        # Initialize components
        self.theme_extractor = self._initialize_theme_extractor()
        self.pattern_detector = self._initialize_pattern_detector()
        
        # Abstraction strategies
        self.abstraction_strategies = [
            ThematicAbstraction(self.theme_extractor),
            PatternAbstraction(self.pattern_detector),
            EmotionalAbstraction()
        ]
        
        # Memory storage
        self.episodic_memories: Dict[str, EpisodicMemory] = {}
        self.semantic_concepts: Dict[str, SemanticConcept] = {}
        self.concept_graph = nx.DiGraph()
        
        # Consolidation tracking
        self.consolidation_history = []
        self.rehearsal_queue = asyncio.Queue()
        
        # Statistics
        self.stats = {
            "total_consolidations": 0,
            "successful_abstractions": 0,
            "failed_abstractions": 0,
            "concepts_created": 0,
            "concepts_strengthened": 0
        }
    
    def _initialize_theme_extractor(self):
        """Initialize theme extraction pipeline"""
        # In production, use more sophisticated NLP
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    def _initialize_pattern_detector(self):
        """Initialize pattern detection system"""
        # Placeholder for pattern detection logic
        return PatternDetector()
    
    async def add_episodic_memory(
        self,
        memory: EpisodicMemory
    ):
        """Add new episodic memory to the system"""
        
        self.episodic_memories[memory.id] = memory
        
        # Check if this memory triggers consolidation
        if await self._should_trigger_consolidation(memory):
            await self.consolidate_memories()
        
        # Add to rehearsal queue if enabled
        if self.enable_rehearsal_consolidation:
            await self.rehearsal_queue.put(memory.id)
    
    async def _should_trigger_consolidation(
        self,
        new_memory: EpisodicMemory
    ) -> bool:
        """Determine if new memory should trigger consolidation"""
        
        # Time-based trigger (sleep consolidation)
        if self.enable_sleep_consolidation:
            last_consolidation = self.consolidation_history[-1] if self.consolidation_history else None
            if last_consolidation:
                time_since_last = datetime.now() - last_consolidation["timestamp"]
                if time_since_last > timedelta(hours=6):  # "Sleep" cycle
                    return True
        
        # Memory count trigger
        unconsolidated_count = sum(
            1 for m in self.episodic_memories.values()
            if not m.associations  # No associations means not consolidated
        )
        
        if unconsolidated_count >= self.min_episodes_for_consolidation * 3:
            return True
        
        # High importance trigger
        if new_memory.emotional_valence > 0.8 or new_memory.emotional_valence < -0.8:
            return True
        
        return False
    
    async def consolidate_memories(
        self,
        strategy: Optional[str] = None
    ) -> List[SemanticConcept]:
        """
        Main consolidation process that transforms episodic memories
        into semantic concepts.
        """
        
        logger.info("Starting memory consolidation process")
        
        # Find consolidation candidates
        candidates = await self._find_consolidation_candidates()
        
        if not candidates:
            logger.info("No suitable candidates for consolidation")
            return []
        
        # Process each candidate group
        new_concepts = []
        
        for candidate in candidates:
            # Try different abstraction strategies
            concept = None
            
            if strategy:
                # Use specific strategy
                for strat in self.abstraction_strategies:
                    if strat.__class__.__name__.lower().startswith(strategy.lower()):
                        concept = await strat.abstract(candidate.memories)
                        break
            else:
                # Try all strategies and use best result
                concepts = []
                for strat in self.abstraction_strategies:
                    try:
                        c = await strat.abstract(candidate.memories)
                        if c:
                            concepts.append(c)
                    except Exception as e:
                        logger.error(f"Abstraction strategy failed: {e}")
                
                # Select best concept based on confidence
                if concepts:
                    concept = max(concepts, key=lambda c: c.confidence)
            
            if concept:
                # Check if similar concept exists
                existing = await self._find_similar_concept(concept)
                
                if existing:
                    # Strengthen existing concept
                    await self._strengthen_concept(existing, candidate.memories)
                    self.stats["concepts_strengthened"] += 1
                else:
                    # Add new concept
                    self.semantic_concepts[concept.id] = concept
                    await self._add_to_concept_graph(concept)
                    new_concepts.append(concept)
                    self.stats["concepts_created"] += 1
                
                # Mark memories as consolidated
                for memory in candidate.memories:
                    memory.associations.append(concept.id)
                
                self.stats["successful_abstractions"] += 1
            else:
                self.stats["failed_abstractions"] += 1
        
        # Record consolidation event
        self.consolidation_history.append({
            "timestamp": datetime.now(),
            "candidates_processed": len(candidates),
            "concepts_created": len(new_concepts),
            "strategy": strategy or "multi"
        })
        
        self.stats["total_consolidations"] += 1
        
        logger.info(f"Consolidation complete: {len(new_concepts)} new concepts created")
        
        return new_concepts
    
    async def _find_consolidation_candidates(self) -> List[ConsolidationCandidate]:
        """Find groups of memories suitable for consolidation"""
        
        # Get unconsolidated memories
        unconsolidated = [
            m for m in self.episodic_memories.values()
            if len(m.associations) == 0  # Not yet consolidated
        ]
        
        if len(unconsolidated) < self.min_episodes_for_consolidation:
            return []
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            [m.content for m in unconsolidated]
        )
        
        # Cluster similar memories
        clustering = DBSCAN(
            eps=1 - self.consolidation_threshold,  # Convert similarity to distance
            min_samples=self.min_episodes_for_consolidation,
            metric='cosine'
        ).fit(embeddings)
        
        # Group memories by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label != -1:  # Not noise
                clusters[label].append(unconsolidated[idx])
        
        # Create candidates from clusters
        candidates = []
        
        for cluster_memories in clusters.values():
            # Calculate cluster metrics
            coherence = await self._calculate_coherence(cluster_memories)
            importance = np.mean([m.emotional_valence for m in cluster_memories])
            emotional_consistency = 1 - np.std([m.emotional_valence for m in cluster_memories])
            
            # Extract theme for cluster
            theme = await self._extract_cluster_theme(cluster_memories)
            
            candidate = ConsolidationCandidate(
                memories=cluster_memories,
                theme=theme,
                coherence_score=coherence,
                importance_score=abs(importance),
                emotional_consistency=emotional_consistency
            )
            
            candidates.append(candidate)
        
        # Sort by consolidation priority
        candidates.sort(
            key=lambda c: c.coherence_score * c.importance_score,
            reverse=True
        )
        
        return candidates
    
    async def _calculate_coherence(
        self,
        memories: List[EpisodicMemory]
    ) -> float:
        """Calculate semantic coherence of memory group"""
        
        if len(memories) < 2:
            return 1.0
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            [m.content for m in memories]
        )
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Average similarity (excluding diagonal)
        n = len(memories)
        total_similarity = (similarities.sum() - n) / (n * (n - 1))
        
        return total_similarity
    
    async def _extract_cluster_theme(
        self,
        memories: List[EpisodicMemory]
    ) -> str:
        """Extract dominant theme from memory cluster"""
        
        # Combine all memory content
        combined_content = " ".join([m.content for m in memories])
        
        # Use theme extractor
        candidate_themes = [
            "learning", "relationship", "achievement", "challenge",
            "discovery", "emotion", "communication", "growth"
        ]
        
        result = self.theme_extractor(
            combined_content,
            candidate_labels=candidate_themes,
            multi_label=False
        )
        
        return result['labels'][0]
    
    async def _find_similar_concept(
        self,
        new_concept: SemanticConcept
    ) -> Optional[SemanticConcept]:
        """Find existing concept similar to new one"""
        
        if not self.semantic_concepts:
            return None
        
        # Compare with existing concepts
        new_embedding = self.embedding_model.encode(new_concept.description)
        
        best_match = None
        best_similarity = 0
        
        for concept_id, concept in self.semantic_concepts.items():
            concept_embedding = self.embedding_model.encode(concept.description)
            similarity = cosine_similarity(
                new_embedding.reshape(1, -1),
                concept_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity and similarity > 0.8:
                best_similarity = similarity
                best_match = concept
        
        return best_match
    
    async def _strengthen_concept(
        self,
        concept: SemanticConcept,
        new_memories: List[EpisodicMemory]
    ):
        """Strengthen existing concept with new evidence"""
        
        # Add source episodes
        concept.source_episodes.extend([m.id for m in new_memories])
        
        # Increase confidence
        concept.strengthen(amount=0.05 * len(new_memories))
        
        # Update attributes
        if "reinforcement_count" not in concept.attributes:
            concept.attributes["reinforcement_count"] = 0
        concept.attributes["reinforcement_count"] += 1
        concept.attributes["last_reinforced"] = datetime.now().isoformat()
        
        logger.debug(f"Strengthened concept {concept.concept} with {len(new_memories)} new memories")
    
    async def _add_to_concept_graph(self, concept: SemanticConcept):
        """Add concept to knowledge graph"""
        
        # Add node
        self.concept_graph.add_node(
            concept.id,
            concept=concept.concept,
            confidence=concept.confidence,
            created_at=datetime.now()
        )
        
        # Find related concepts
        if len(self.semantic_concepts) > 1:
            concept_embedding = self.embedding_model.encode(concept.description)
            
            for other_id, other_concept in self.semantic_concepts.items():
                if other_id != concept.id:
                    other_embedding = self.embedding_model.encode(other_concept.description)
                    similarity = cosine_similarity(
                        concept_embedding.reshape(1, -1),
                        other_embedding.reshape(1, -1)
                    )[0][0]
                    
                    if similarity > 0.5:  # Related concepts
                        self.concept_graph.add_edge(
                            concept.id,
                            other_id,
                            weight=similarity
                        )
                        
                        # Update relationships
                        concept.relationships[other_id] = similarity
                        other_concept.relationships[concept.id] = similarity
    
    async def rehearsal_consolidation(self):
        """
        Consolidate memories through rehearsal (repeated activation).
        Simulates the effect of conscious recall on memory consolidation.
        """
        
        if not self.enable_rehearsal_consolidation:
            return
        
        rehearsed_memories = []
        
        # Process rehearsal queue
        while not self.rehearsal_queue.empty():
            try:
                memory_id = await asyncio.wait_for(
                    self.rehearsal_queue.get(),
                    timeout=0.1
                )
                
                if memory_id in self.episodic_memories:
                    memory = self.episodic_memories[memory_id]
                    memory.rehearsal_count += 1
                    
                    # Strengthen vividness through rehearsal
                    memory.vividness = min(1.0, memory.vividness + 0.1)
                    
                    rehearsed_memories.append(memory)
                    
            except asyncio.TimeoutError:
                break
        
        # Check if rehearsed memories form patterns
        if len(rehearsed_memories) >= self.min_episodes_for_consolidation:
            # Group frequently rehearsed memories
            frequently_rehearsed = [
                m for m in rehearsed_memories
                if m.rehearsal_count >= 3
            ]
            
            if len(frequently_rehearsed) >= self.min_episodes_for_consolidation:
                # These memories are important due to rehearsal
                candidate = ConsolidationCandidate(
                    memories=frequently_rehearsed,
                    theme="frequently_accessed",
                    coherence_score=0.8,  # High coherence due to rehearsal
                    importance_score=0.9,  # High importance due to frequency
                    emotional_consistency=0.7
                )
                
                # Trigger consolidation for rehearsed memories
                concepts = await self._process_single_candidate(candidate)
                
                if concepts:
                    logger.info(f"Rehearsal consolidation created {len(concepts)} concepts")
    
    async def _process_single_candidate(
        self,
        candidate: ConsolidationCandidate
    ) -> List[SemanticConcept]:
        """Process a single consolidation candidate"""
        
        concepts = []
        
        for strategy in self.abstraction_strategies:
            try:
                concept = await strategy.abstract(candidate.memories)
                if concept:
                    concepts.append(concept)
            except Exception as e:
                logger.error(f"Strategy {strategy.__class__.__name__} failed: {e}")
        
        return concepts
    
    async def sleep_consolidation(
        self,
        rem_cycles: int = 4,
        slow_wave_boost: float = 1.5
    ):
        """
        Simulate sleep-like consolidation process.
        REM cycles for emotional processing, slow-wave for memory transfer.
        """
        
        if not self.enable_sleep_consolidation:
            return
        
        logger.info(f"Starting sleep consolidation with {rem_cycles} REM cycles")
        
        for cycle in range(rem_cycles):
            # REM phase - emotional memory processing
            emotional_memories = [
                m for m in self.episodic_memories.values()
                if abs(m.emotional_valence) > 0.5
            ]
            
            if len(emotional_memories) >= self.min_episodes_for_consolidation:
                # Process emotional memories
                emotional_candidates = await self._find_emotional_patterns(emotional_memories)
                
                for candidate in emotional_candidates:
                    concept = await EmotionalAbstraction().abstract(candidate.memories)
                    if concept:
                        concept.confidence *= slow_wave_boost  # Boost from slow-wave
                        self.semantic_concepts[concept.id] = concept
                        await self._add_to_concept_graph(concept)
            
            # Slow-wave phase - memory replay and strengthening
            await self._memory_replay(slow_wave_boost)
            
            # Brief awakening - prune weak memories
            await self._prune_weak_memories()
        
        logger.info("Sleep consolidation complete")
    
    async def _find_emotional_patterns(
        self,
        memories: List[EpisodicMemory]
    ) -> List[ConsolidationCandidate]:
        """Find patterns in emotional memories"""
        
        # Group by emotional valence
        positive_memories = [m for m in memories if m.emotional_valence > 0.5]
        negative_memories = [m for m in memories if m.emotional_valence < -0.5]
        
        candidates = []
        
        for memory_group, emotion_type in [(positive_memories, "positive"), 
                                          (negative_memories, "negative")]:
            if len(memory_group) >= self.min_episodes_for_consolidation:
                candidate = ConsolidationCandidate(
                    memories=memory_group,
                    theme=f"{emotion_type}_emotional_pattern",
                    coherence_score=0.7,
                    importance_score=0.8,
                    emotional_consistency=0.9
                )
                candidates.append(candidate)
        
        return candidates
    
    async def _memory_replay(self, boost_factor: float):
        """Replay and strengthen important memories"""
        
        # Select memories for replay based on importance
        replay_candidates = sorted(
            self.episodic_memories.values(),
            key=lambda m: m.emotional_valence * m.vividness,
            reverse=True
        )[:20]  # Top 20 memories
        
        for memory in replay_candidates:
            # Strengthen associated concepts
            for concept_id in memory.associations:
                if concept_id in self.semantic_concepts:
                    concept = self.semantic_concepts[concept_id]
                    concept.strengthen(amount=0.02 * boost_factor)
    
    async def _prune_weak_memories(self):
        """Remove very weak episodic memories"""
        
        to_remove = []
        
        for memory_id, memory in self.episodic_memories.items():
            # Apply decay
            memory.decay()
            
            # Mark for removal if too weak and already consolidated
            if memory.vividness < 0.1 and len(memory.associations) > 0:
                to_remove.append(memory_id)
        
        # Remove weak memories
        for memory_id in to_remove:
            del self.episodic_memories[memory_id]
        
        if to_remove:
            logger.debug(f"Pruned {len(to_remove)} weak memories")
    
    async def query_semantic_knowledge(
        self,
        query: str,
        include_sources: bool = True
    ) -> List[Dict[str, Any]]:
        """Query semantic knowledge base"""
        
        if not self.semantic_concepts:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Search concepts
        results = []
        
        for concept in self.semantic_concepts.values():
            concept_embedding = self.embedding_model.encode(concept.description)
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                concept_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > 0.5:
                result = {
                    "concept": concept.concept,
                    "description": concept.description,
                    "confidence": concept.confidence,
                    "relevance": similarity
                }
                
                if include_sources:
                    # Include sample source memories
                    source_memories = []
                    for episode_id in concept.source_episodes[:3]:
                        if episode_id in self.episodic_memories:
                            memory = self.episodic_memories[episode_id]
                            source_memories.append({
                                "content": memory.content[:100] + "...",
                                "timestamp": memory.timestamp.isoformat()
                            })
                    result["sources"] = source_memories
                
                results.append(result)
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        return results
    
    def get_concept_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the concept graph"""
        
        if not self.concept_graph:
            return {"nodes": 0, "edges": 0}
        
        stats = {
            "total_concepts": self.concept_graph.number_of_nodes(),
            "total_relationships": self.concept_graph.number_of_edges(),
            "avg_connections": (
                self.concept_graph.number_of_edges() / 
                max(1, self.concept_graph.number_of_nodes())
            ),
            "most_connected": None,
            "isolated_concepts": 0
        }
        
        # Find most connected concept
        if self.concept_graph.number_of_nodes() > 0:
            degrees = dict(self.concept_graph.degree())
            most_connected_id = max(degrees, key=degrees.get)
            stats["most_connected"] = {
                "id": most_connected_id,
                "concept": self.semantic_concepts.get(most_connected_id, {}).concept,
                "connections": degrees[most_connected_id]
            }
            
            # Count isolated concepts
            stats["isolated_concepts"] = sum(
                1 for node, degree in degrees.items() if degree == 0
            )
        
        return stats
    
    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics"""
        
        return {
            **self.stats,
            "episodic_memories": len(self.episodic_memories),
            "semantic_concepts": len(self.semantic_concepts),
            "consolidation_history": self.consolidation_history[-10:],  # Last 10
            "concept_graph_stats": self.get_concept_graph_stats()
        }


class PatternDetector:
    """Placeholder for pattern detection functionality"""
    
    async def detect_patterns(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Detect patterns in text"""
        # Simplified pattern detection
        patterns = []
        
        # Look for repeated phrases
        all_text = " ".join(texts).lower()
        words = all_text.split()
        
        # Find common bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        
        for bigram, count in bigram_counts.most_common(3):
            if count >= 3:
                patterns.append({
                    "type": "phrase_repetition",
                    "name": f"Pattern: {bigram}",
                    "description": f"Repeated phrase '{bigram}' found {count} times",
                    "confidence": min(0.9, count * 0.1),
                    "strength": count / len(texts)
                })
        
        return patterns