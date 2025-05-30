"""
Sleep-like memory consolidation system for consciousness-aware AI.
Implements REM and slow-wave sleep phases for memory processing.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class SleepPhase:
    """Represents a phase of sleep consolidation"""
    name: str
    duration_seconds: float
    consolidation_rate: float
    replay_probability: float
    creativity_factor: float
    
class MemoryReplayBuffer:
    """Buffer for memory replay during consolidation"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.importance_scores = deque(maxlen=capacity)
        
    def add(self, memory: Dict[str, Any], importance: float):
        """Add memory with importance score"""
        self.buffer.append(memory)
        self.importance_scores.append(importance)
        
    def sample_for_replay(self, n_samples: int) -> List[Dict[str, Any]]:
        """Sample memories weighted by importance"""
        if len(self.buffer) == 0:
            return []
            
        # Convert to numpy for weighted sampling
        scores = np.array(self.importance_scores)
        probabilities = scores / scores.sum()
        
        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=min(n_samples, len(self.buffer)),
            replace=False,
            p=probabilities
        )
        
        return [self.buffer[i] for i in indices]

class ConsolidationNetwork(nn.Module):
    """Neural network for memory transformation during consolidation"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.consolidator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim // 2,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.1
            ),
            num_layers=3
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, memory_embedding: torch.Tensor, sleep_phase: str) -> torch.Tensor:
        """Transform memory based on sleep phase"""
        # Encode
        encoded = self.encoder(memory_embedding)
        
        # Add positional encoding for transformer
        encoded = encoded.unsqueeze(0)  # Add sequence dimension
        
        # Consolidate with phase-specific processing
        if sleep_phase == "REM":
            # More creative transformation
            consolidated = self.consolidator(encoded)
            # Add noise for creativity
            noise = torch.randn_like(consolidated) * 0.1
            consolidated = consolidated + noise
        else:
            # More structured consolidation
            consolidated = self.consolidator(encoded)
            
        # Decode back to memory space
        consolidated = consolidated.squeeze(0)
        return self.decoder(consolidated)

class SleepConsolidationSystem:
    """Main sleep consolidation system"""
    
    def __init__(self, memory_system, consciousness_core=None):
        self.memory_system = memory_system
        self.consciousness_core = consciousness_core
        
        # Sleep phases (simplified sleep architecture)
        self.sleep_phases = [
            SleepPhase("NREM1", 300, 0.1, 0.05, 0.1),  # 5 min light sleep
            SleepPhase("NREM2", 1200, 0.3, 0.1, 0.2),  # 20 min 
            SleepPhase("SWS", 1800, 0.6, 0.3, 0.1),    # 30 min slow-wave
            SleepPhase("REM", 900, 0.4, 0.5, 0.8),     # 15 min REM
        ]
        
        # Replay buffer
        self.replay_buffer = MemoryReplayBuffer()
        
        # Consolidation network
        self.consolidation_network = ConsolidationNetwork()
        
        # State tracking
        self.is_consolidating = False
        self.current_phase = None
        self.consolidation_history = []
        self.last_consolidation = None
        
        # Consolidation parameters
        self.min_memories_for_consolidation = 50
        self.consolidation_interval = timedelta(hours=8)  # Every 8 hours
        
    async def should_consolidate(self) -> bool:
        """Determine if consolidation should occur"""
        # Check if enough new memories
        new_memory_count = await self._count_unconsolidated_memories()
        if new_memory_count < self.min_memories_for_consolidation:
            return False
            
        # Check time since last consolidation
        if self.last_consolidation:
            time_since = datetime.now() - self.last_consolidation
            if time_since < self.consolidation_interval:
                return False
                
        # Check consciousness level if available
        if self.consciousness_core:
            consciousness_level = self.consciousness_core.get_awareness_level()
            # Only consolidate during low consciousness (rest state)
            if consciousness_level > 0.3:
                return False
                
        return True
        
    async def _count_unconsolidated_memories(self) -> int:
        """Count memories that haven't been consolidated"""
        # This would query the memory system for unconsolidated memories
        # For now, return the replay buffer size
        return len(self.replay_buffer.buffer)
        
    async def initiate_consolidation(self):
        """Start the consolidation process"""
        if self.is_consolidating:
            logger.warning("Consolidation already in progress")
            return
            
        self.is_consolidating = True
        self.last_consolidation = datetime.now()
        
        logger.info("Initiating memory consolidation cycle")
        
        try:
            # Load recent memories into replay buffer
            await self._load_memories_for_consolidation()
            
            # Execute sleep phases
            for phase in self.sleep_phases:
                await self._execute_sleep_phase(phase)
                
            # Final integration
            await self._integrate_consolidated_memories()
            
        finally:
            self.is_consolidating = False
            self.current_phase = None
            
    async def _load_memories_for_consolidation(self):
        """Load recent memories into replay buffer"""
        # Get recent episodic memories
        recent_memories = await self.memory_system.get_recent_memories(
            limit=1000,
            include_importance=True
        )
        
        # Add to replay buffer
        for memory in recent_memories:
            self.replay_buffer.add(
                memory['content'],
                memory.get('importance', 0.5)
            )
            
    async def _execute_sleep_phase(self, phase: SleepPhase):
        """Execute a single sleep phase"""
        self.current_phase = phase.name
        logger.info(f"Entering {phase.name} phase")
        
        start_time = time.time()
        phase_memories_processed = 0
        
        # Process memories during this phase
        while time.time() - start_time < phase.duration_seconds:
            # Sample memories for replay
            memories_to_replay = self.replay_buffer.sample_for_replay(
                n_samples=int(10 * phase.replay_probability)
            )
            
            # Process each memory
            for memory in memories_to_replay:
                consolidated = await self._consolidate_memory(memory, phase)
                phase_memories_processed += 1
                
                # Store consolidated version
                await self._store_consolidated_memory(consolidated, phase)
                
            # Brief pause between replay cycles
            await asyncio.sleep(1.0)
            
        logger.info(f"Completed {phase.name}: processed {phase_memories_processed} memories")
        
    async def _consolidate_memory(self, memory: Dict[str, Any], phase: SleepPhase) -> Dict[str, Any]:
        """Consolidate a single memory"""
        # Extract embedding
        embedding = memory.get('embedding')
        if embedding is None:
            # Generate embedding if not present
            embedding = await self.memory_system.generate_embedding(memory['content'])
            
        # Convert to tensor
        memory_tensor = torch.tensor(embedding).float()
        
        # Apply consolidation network
        with torch.no_grad():
            consolidated_embedding = self.consolidation_network(
                memory_tensor,
                phase.name
            )
            
        # Create consolidated memory
        consolidated = {
            'original_id': memory.get('id'),
            'content': memory['content'],
            'embedding': consolidated_embedding.numpy(),
            'consolidation_phase': phase.name,
            'consolidation_timestamp': datetime.now(),
            'creativity_factor': phase.creativity_factor,
            'original_importance': memory.get('importance', 0.5)
        }
        
        # Phase-specific transformations
        if phase.name == "REM":
            # REM phase: Create associations and insights
            consolidated['associations'] = await self._generate_associations(memory, phase)
            consolidated['insights'] = await self._extract_insights(memory, phase)
        elif phase.name == "SWS":
            # Slow-wave sleep: Strengthen important memories
            consolidated['importance'] = min(1.0, memory.get('importance', 0.5) * 1.5)
            consolidated['semantic_category'] = await self._categorize_memory(memory)
            
        return consolidated
        
    async def _generate_associations(self, memory: Dict[str, Any], phase: SleepPhase) -> List[str]:
        """Generate creative associations during REM"""
        # Find similar memories
        similar = await self.memory_system.search_similar(
            memory['embedding'],
            limit=5
        )
        
        # Create novel associations
        associations = []
        for sim_memory in similar:
            # Combine concepts creatively
            association = {
                'related_memory_id': sim_memory['id'],
                'connection_strength': sim_memory['similarity'],
                'connection_type': 'creative' if phase.creativity_factor > 0.5 else 'logical'
            }
            associations.append(association)
            
        return associations
        
    async def _extract_insights(self, memory: Dict[str, Any], phase: SleepPhase) -> List[str]:
        """Extract insights from memory patterns"""
        insights = []
        
        # Pattern recognition across memories
        if 'patterns' in memory:
            for pattern in memory['patterns']:
                insight = {
                    'type': 'pattern_recognition',
                    'content': pattern,
                    'confidence': phase.consolidation_rate
                }
                insights.append(insight)
                
        return insights
        
    async def _categorize_memory(self, memory: Dict[str, Any]) -> str:
        """Categorize memory for semantic organization"""
        # Simple categorization - in practice, this would use NLP
        content = memory.get('content', '')
        
        if 'emotion' in content.lower():
            return 'emotional'
        elif 'fact' in content.lower() or 'know' in content.lower():
            return 'factual'
        elif 'plan' in content.lower() or 'will' in content.lower():
            return 'prospective'
        else:
            return 'episodic'
            
    async def _store_consolidated_memory(self, consolidated: Dict[str, Any], phase: SleepPhase):
        """Store the consolidated memory"""
        # Update original memory with consolidation info
        if consolidated['original_id']:
            await self.memory_system.update_memory(
                consolidated['original_id'],
                {
                    'consolidated': True,
                    'consolidation_phase': phase.name,
                    'consolidation_timestamp': consolidated['consolidation_timestamp'],
                    'enhanced_importance': consolidated.get('importance', 
                                                          consolidated['original_importance'])
                }
            )
            
        # Store new associations and insights
        if 'associations' in consolidated:
            for association in consolidated['associations']:
                await self.memory_system.add_association(
                    consolidated['original_id'],
                    association['related_memory_id'],
                    association['connection_strength'],
                    association['connection_type']
                )
                
        if 'insights' in consolidated:
            for insight in consolidated['insights']:
                await self.memory_system.add_insight(insight)
                
    async def _integrate_consolidated_memories(self):
        """Final integration phase after all sleep phases"""
        logger.info("Performing final memory integration")
        
        # Update semantic memory with consolidated episodic memories
        consolidated_memories = await self.memory_system.get_consolidated_memories()
        
        for memory in consolidated_memories:
            if memory.get('semantic_category'):
                # Promote to semantic memory if sufficiently important
                if memory.get('importance', 0) > 0.7:
                    await self.memory_system.promote_to_semantic(memory)
                    
        # Prune redundant memories
        await self._prune_redundant_memories()
        
        # Update memory statistics
        self.consolidation_history.append({
            'timestamp': datetime.now(),
            'memories_processed': len(consolidated_memories),
            'phases_completed': len(self.sleep_phases)
        })
        
    async def _prune_redundant_memories(self):
        """Remove redundant or low-importance memories"""
        # Find highly similar memories
        redundant_pairs = await self.memory_system.find_redundant_memories(
            similarity_threshold=0.95
        )
        
        for pair in redundant_pairs:
            # Keep the more important/recent one
            if pair[0]['importance'] > pair[1]['importance']:
                await self.memory_system.archive_memory(pair[1]['id'])
            else:
                await self.memory_system.archive_memory(pair[0]['id'])
                
    def get_consolidation_status(self) -> Dict[str, Any]:
        """Get current consolidation status"""
        return {
            'is_consolidating': self.is_consolidating,
            'current_phase': self.current_phase,
            'last_consolidation': self.last_consolidation,
            'total_consolidations': len(self.consolidation_history),
            'replay_buffer_size': len(self.replay_buffer.buffer)
        }
        
    async def force_consolidation(self):
        """Force immediate consolidation (override normal checks)"""
        logger.info("Forcing immediate consolidation")
        await self.initiate_consolidation()
        
    async def stop_consolidation(self):
        """Stop ongoing consolidation"""
        if self.is_consolidating:
            logger.info("Stopping consolidation")
            self.is_consolidating = False
            # Consolidation will stop at next phase transition