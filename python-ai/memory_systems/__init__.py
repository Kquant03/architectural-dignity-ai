"""
Memory Systems Package - Unified interface for all memory components.
Integrates Mem0, Letta, cognitive memory types, consolidation, and visualization.
"""

from typing import Dict, List, Optional, Any, Union
import asyncio
from datetime import datetime, timedelta
import logging

# Import all memory system components
from .mem0_integration import ConsciousnessAwareMemory
from .letta_adapter import LettaAgentMemory
from .episodic_semantic_bridge import EpisodicSemanticBridge
from .consolidation_network import ConsolidationNetwork
from .cognitive_memory import (
    WorkingMemory,
    ProceduralMemory,
    ProspectiveMemory,
    MetaMemory,
    CognitiveMemorySystem
)
from .sleep_consolidation import SleepConsolidationSystem, SleepPhase
from .memory_graph import MemoryGraph, MemoryNode, MemoryEdge

logger = logging.getLogger(__name__)

__all__ = [
    'UnifiedMemorySystem',
    'ConsciousnessAwareMemory',
    'LettaAgentMemory',
    'EpisodicSemanticBridge',
    'ConsolidationNetwork',
    'WorkingMemory',
    'ProceduralMemory',
    'ProspectiveMemory',
    'MetaMemory',
    'CognitiveMemorySystem',
    'SleepConsolidationSystem',
    'MemoryGraph',
    'MemoryNode',
    'MemoryEdge'
]

class UnifiedMemorySystem:
    """
    Unified interface for all memory subsystems.
    Orchestrates interactions between different memory types and processes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize unified memory system with all components.
        
        Args:
            config: Configuration dictionary with database settings, model paths, etc.
        """
        self.config = config or {}
        
        # Initialize core memory systems
        self.consciousness_memory = ConsciousnessAwareMemory(
            db_config=self.config.get('db_config', {}),
            consciousness_threshold=self.config.get('consciousness_threshold', 0.5)
        )
        
        self.letta_memory = LettaAgentMemory(
            agent_name=self.config.get('agent_name', 'consciousness_ai'),
            memory_config=self.config.get('letta_config', {})
        )
        
        # Initialize cognitive memory types
        self.cognitive_memory = CognitiveMemorySystem(
            working_memory_capacity=self.config.get('working_memory_capacity', 100),
            ltm_model_path=self.config.get('ltm_model_path')
        )
        
        # Initialize memory processing systems
        self.episodic_semantic_bridge = EpisodicSemanticBridge()
        self.consolidation_network = ConsolidationNetwork(
            model_path=self.config.get('consolidation_model_path')
        )
        
        # Initialize sleep consolidation
        self.sleep_consolidation = SleepConsolidationSystem(
            memory_system=self,
            consciousness_core=None  # Will be set by consciousness core
        )
        
        # Initialize memory graph
        self.memory_graph = MemoryGraph()
        
        # State tracking
        self.total_memories = 0
        self.last_consolidation = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize all memory subsystems"""
        if self.is_initialized:
            return
            
        logger.info("Initializing unified memory system")
        
        # Initialize databases and connections
        await self.consciousness_memory.initialize()
        await self.letta_memory.initialize()
        
        # Load existing memories into graph
        await self._load_memories_to_graph()
        
        self.is_initialized = True
        logger.info("Memory system initialization complete")
        
    async def store_memory(self, content: str, memory_type: str = 'episodic',
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a memory across all relevant subsystems.
        
        Args:
            content: Memory content
            memory_type: Type of memory (episodic, semantic, procedural, etc.)
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        metadata = metadata or {}
        timestamp = datetime.now()
        
        # Generate embedding
        embedding = await self.consciousness_memory.generate_embedding(content)
        
        # Store in consciousness-aware memory (Mem0)
        mem0_result = await self.consciousness_memory.add_memory(
            content=content,
            metadata={
                **metadata,
                'memory_type': memory_type,
                'timestamp': timestamp
            }
        )
        
        memory_id = mem0_result['id']
        
        # Store in Letta for agent personality
        if memory_type in ['episodic', 'semantic']:
            await self.letta_memory.add_to_archival_memory(content)
            
        # Add to appropriate cognitive memory
        if memory_type == 'procedural':
            self.cognitive_memory.procedural_memory.learn_procedure(
                memory_id, content, metadata
            )
        elif memory_type == 'prospective':
            self.cognitive_memory.prospective_memory.add_intention(
                content, metadata.get('trigger_time'), metadata
            )
            
        # Add to working memory if recent
        self.cognitive_memory.working_memory.add({
            'id': memory_id,
            'content': content,
            'timestamp': timestamp,
            'importance': metadata.get('importance', 0.5)
        })
        
        # Add to memory graph
        memory_node = MemoryNode(
            id=memory_id,
            content=content,
            embedding=embedding,
            timestamp=timestamp,
            memory_type=memory_type,
            importance=metadata.get('importance', 0.5),
            emotional_valence=metadata.get('emotional_valence', 0.0),
            consolidation_count=0,
            metadata=metadata
        )
        self.memory_graph.add_memory_node(memory_node)
        
        # Update meta-memory
        self.cognitive_memory.meta_memory.track_memory_operation(
            'store', memory_id, {'memory_type': memory_type}
        )
        
        self.total_memories += 1
        
        # Check if consolidation needed
        if await self.sleep_consolidation.should_consolidate():
            asyncio.create_task(self.sleep_consolidation.initiate_consolidation())
            
        return memory_id
        
    async def retrieve_memory(self, query: str, limit: int = 10,
                            memory_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve memories using multi-system search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            memory_types: Filter by memory types
            
        Returns:
            List of memory dictionaries
        """
        # Search across different memory systems
        results = []
        
        # Search consciousness-aware memory
        mem0_results = await self.consciousness_memory.search_memories(
            query=query,
            limit=limit,
            filters={'memory_type': memory_types} if memory_types else None
        )
        results.extend(mem0_results)
        
        # Search Letta archival memory
        letta_results = await self.letta_memory.search_archival_memory(
            query=query,
            limit=limit
        )
        results.extend(letta_results)
        
        # Check working memory
        working_mem_results = self.cognitive_memory.working_memory.search(query)
        results.extend(working_mem_results[:limit])
        
        # Deduplicate and sort by relevance
        seen_ids = set()
        unique_results = []
        for result in results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
                
        # Sort by relevance score
        unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Update meta-memory
        self.cognitive_memory.meta_memory.track_memory_operation(
            'retrieve', query, {'num_results': len(unique_results)}
        )
        
        return unique_results[:limit]
        
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]):
        """Update a memory across all systems"""
        # Update in Mem0
        await self.consciousness_memory.update_memory(memory_id, updates)
        
        # Update in memory graph
        if memory_id in self.memory_graph.graph:
            for key, value in updates.items():
                self.memory_graph.graph.nodes[memory_id][key] = value
                
        # Track in meta-memory
        self.cognitive_memory.meta_memory.track_memory_operation(
            'update', memory_id, updates
        )
        
    async def create_memory_association(self, memory_id1: str, memory_id2: str,
                                      relationship_type: str = 'association',
                                      strength: float = 0.5):
        """Create association between memories"""
        # Add edge in memory graph
        edge = MemoryEdge(
            source_id=memory_id1,
            target_id=memory_id2,
            relationship_type=relationship_type,
            strength=strength,
            created_at=datetime.now(),
            metadata={}
        )
        self.memory_graph.add_memory_edge(edge)
        
        # Store association in Mem0
        await self.consciousness_memory.add_memory(
            content=f"Association: {memory_id1} <-> {memory_id2}",
            metadata={
                'memory_type': 'association',
                'source_id': memory_id1,
                'target_id': memory_id2,
                'relationship_type': relationship_type,
                'strength': strength
            }
        )
        
    async def consolidate_memories(self, force: bool = False):
        """Trigger memory consolidation process"""
        if force:
            await self.sleep_consolidation.force_consolidation()
        else:
            await self.sleep_consolidation.initiate_consolidation()
            
        self.last_consolidation = datetime.now()
        
    async def get_memory_context(self, memory_id: str, radius: int = 2) -> Dict[str, Any]:
        """Get contextual memories around a specific memory"""
        # Get graph context
        context_graph = self.memory_graph.get_memory_context(memory_id, radius)
        
        # Get actual memory content for context nodes
        context_memories = []
        for node in context_graph.nodes():
            memory = await self.consciousness_memory.get_memory(node)
            if memory:
                context_memories.append(memory)
                
        return {
            'central_memory': memory_id,
            'context_memories': context_memories,
            'context_graph': context_graph,
            'num_connections': context_graph.number_of_edges()
        }
        
    async def analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in memory system"""
        # Get graph statistics
        graph_stats = self.memory_graph.get_graph_statistics()
        
        # Get cognitive memory statistics
        cognitive_stats = self.cognitive_memory.get_system_stats()
        
        # Get consolidation history
        consolidation_stats = self.sleep_consolidation.get_consolidation_status()
        
        # Identify key memories
        key_memories = self.memory_graph.identify_key_memories(top_k=20)
        
        # Find memory clusters
        clusters = self.memory_graph.find_memory_clusters()
        
        return {
            'graph_statistics': graph_stats,
            'cognitive_statistics': cognitive_stats,
            'consolidation_status': consolidation_stats,
            'key_memories': key_memories,
            'memory_clusters': clusters,
            'total_memories': self.total_memories
        }
        
    async def _load_memories_to_graph(self):
        """Load existing memories into the graph structure"""
        # Get all memories from Mem0
        all_memories = await self.consciousness_memory.get_all_memories(limit=10000)
        
        for memory in all_memories:
            # Create memory node
            node = MemoryNode(
                id=memory['id'],
                content=memory['content'],
                embedding=memory.get('embedding', np.zeros(768)),
                timestamp=memory.get('timestamp', datetime.now()),
                memory_type=memory.get('memory_type', 'episodic'),
                importance=memory.get('importance', 0.5),
                emotional_valence=memory.get('emotional_valence', 0.0),
                consolidation_count=memory.get('consolidation_count', 0),
                metadata=memory.get('metadata', {})
            )
            self.memory_graph.add_memory_node(node)
            
        # Load associations
        associations = await self.consciousness_memory.get_all_associations()
        for assoc in associations:
            edge = MemoryEdge(
                source_id=assoc['source_id'],
                target_id=assoc['target_id'],
                relationship_type=assoc.get('relationship_type', 'association'),
                strength=assoc.get('strength', 0.5),
                created_at=assoc.get('created_at', datetime.now()),
                metadata=assoc.get('metadata', {})
            )
            self.memory_graph.add_memory_edge(edge)
            
    def visualize_memories(self, output_path: Optional[str] = None,
                          highlight_memories: Optional[List[str]] = None):
        """Create visualization of memory graph"""
        return self.memory_graph.visualize_memory_graph(
            output_path=output_path,
            highlight_nodes=highlight_memories
        )
        
    def create_interactive_memory_map(self):
        """Create interactive 3D memory visualization"""
        return self.memory_graph.create_interactive_visualization()
        
    async def export_memories(self, filepath: str):
        """Export all memories to file"""
        self.memory_graph.export_to_json(filepath)
        
    async def import_memories(self, filepath: str):
        """Import memories from file"""
        self.memory_graph.import_from_json(filepath)
        
    async def shutdown(self):
        """Gracefully shutdown memory systems"""
        logger.info("Shutting down memory systems")
        
        # Stop any ongoing consolidation
        await self.sleep_consolidation.stop_consolidation()
        
        # Save current state
        await self.export_memories('memory_backup.json')
        
        # Close connections
        await self.consciousness_memory.close()
        await self.letta_memory.close()
        
        logger.info("Memory systems shutdown complete")