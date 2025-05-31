# python-ai/memory_systems/human_like_memory.py
"""
Human-like memory consolidation with sleep-like cycles, emotional tagging,
and interference patterns that mirror biological memory systems.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from collections import deque
import random

@dataclass
class MemoryTrace:
    """Represents a memory trace with decay and consolidation properties"""
    content: str
    encoding_strength: float
    emotional_tag: Dict[str, float]
    context_embedding: np.ndarray
    timestamp: datetime
    access_count: int = 0
    last_access: Optional[datetime] = None
    consolidation_stage: str = "sensory"  # sensory -> working -> long-term
    interference_vulnerability: float = 1.0
    sleep_replay_count: int = 0
    
class SleepConsolidationCycle:
    """Simulates sleep-like memory consolidation cycles"""
    
    def __init__(self):
        self.cycle_phase = "wake"  # wake, NREM, REM
        self.phase_duration = {
            "wake": 16 * 3600,  # 16 hours
            "NREM": 1.5 * 3600,  # 1.5 hours
            "REM": 0.5 * 3600   # 30 minutes
        }
        self.last_phase_change = datetime.now()
        
    async def run_consolidation_cycle(self, memory_system):
        """Run periodic consolidation like sleep cycles"""
        while True:
            current_phase = self.get_current_phase()
            
            if current_phase == "NREM":
                # Deep sleep - consolidate declarative memories
                await self._nrem_consolidation(memory_system)
            elif current_phase == "REM":
                # REM sleep - emotional memory processing & creativity
                await self._rem_consolidation(memory_system)
                
            await asyncio.sleep(300)  # Check every 5 minutes
            
    def get_current_phase(self) -> str:
        """Determine current consolidation phase"""
        elapsed = (datetime.now() - self.last_phase_change).total_seconds()
        
        if elapsed > self.phase_duration[self.cycle_phase]:
            # Transition to next phase
            if self.cycle_phase == "wake":
                self.cycle_phase = "NREM"
            elif self.cycle_phase == "NREM":
                self.cycle_phase = "REM"
            else:
                self.cycle_phase = "wake"
            
            self.last_phase_change = datetime.now()
            
        return self.cycle_phase
        
    async def _nrem_consolidation(self, memory_system):
        """NREM consolidation - replay and strengthen important memories"""
        # Get memories that need consolidation
        candidates = memory_system.get_consolidation_candidates()
        
        for memory in candidates:
            if memory.consolidation_stage == "working":
                # Replay memory (strengthening synaptic connections)
                memory.sleep_replay_count += 1
                memory.encoding_strength *= 1.1  # Strengthen
                
                # Reduce interference vulnerability
                memory.interference_vulnerability *= 0.9
                
                # Promote to long-term if replayed enough
                if memory.sleep_replay_count >= 3:
                    memory.consolidation_stage = "long-term"
                    
    async def _rem_consolidation(self, memory_system):
        """REM consolidation - emotional processing and creative connections"""
        emotional_memories = memory_system.get_emotional_memories()
        
        for memory in emotional_memories:
            # Process emotional content
            emotion_intensity = max(memory.emotional_tag.values())
            if emotion_intensity > 0.7:
                # Strong emotions get extra processing
                memory.encoding_strength *= 1.2
                
            # Create novel associations (dream-like)
            await memory_system.create_creative_associations(memory)

class InterferencePattern:
    """Models memory interference patterns"""
    
    @staticmethod
    def calculate_interference(new_memory: MemoryTrace, 
                             existing_memories: List[MemoryTrace]) -> float:
        """Calculate retroactive and proactive interference"""
        interference = 0.0
        
        for existing in existing_memories:
            # Similarity causes interference
            similarity = InterferencePattern._calculate_similarity(
                new_memory.context_embedding,
                existing.context_embedding
            )
            
            # Recent memories interfere more
            recency_factor = 1.0 / (1.0 + (datetime.now() - existing.timestamp).total_seconds() / 3600)
            
            # Weak memories are more vulnerable
            vulnerability = existing.interference_vulnerability
            
            interference += similarity * recency_factor * vulnerability
            
        return min(interference, 1.0)
        
    @staticmethod
    def _calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate semantic similarity between memories"""
        return float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

class SpacedRepetitionScheduler:
    """Implements spaced repetition for optimal memory retention"""
    
    def __init__(self):
        self.intervals = [1, 3, 7, 14, 30, 90]  # Days
        
    def get_next_review(self, memory: MemoryTrace) -> datetime:
        """Calculate when memory should be reviewed"""
        if memory.access_count >= len(self.intervals):
            # After all intervals, review yearly
            return memory.last_access + timedelta(days=365)
            
        days = self.intervals[memory.access_count]
        return memory.timestamp + timedelta(days=days)
        
    def needs_review(self, memory: MemoryTrace) -> bool:
        """Check if memory needs review to prevent forgetting"""
        if not memory.last_access:
            return True
            
        next_review = self.get_next_review(memory)
        return datetime.now() >= next_review

class HumanLikeMemorySystem:
    """Complete human-like memory system with biological patterns"""
    
    def __init__(self, consciousness_core):
        self.consciousness_core = consciousness_core
        
        # Memory stores
        self.sensory_buffer = deque(maxlen=20)  # Very short term
        self.working_memory = deque(maxlen=7)   # Miller's magic number
        self.long_term_memory = []
        
        # Consolidation systems
        self.sleep_cycle = SleepConsolidationCycle()
        self.repetition_scheduler = SpacedRepetitionScheduler()
        
        # Start background processes
        asyncio.create_task(self.sleep_cycle.run_consolidation_cycle(self))
        asyncio.create_task(self._forgetting_process())
        
    async def encode_memory(self, content: str, emotional_context: Dict[str, float],
                          context_embedding: np.ndarray) -> MemoryTrace:
        """Encode new memory with interference checking"""
        
        # Create memory trace
        memory = MemoryTrace(
            content=content,
            encoding_strength=self._calculate_encoding_strength(emotional_context),
            emotional_tag=emotional_context,
            context_embedding=context_embedding,
            timestamp=datetime.now()
        )
        
        # Check for interference
        interference = InterferencePattern.calculate_interference(
            memory, self.get_all_memories()
        )
        
        # Interference weakens encoding
        memory.encoding_strength *= (1.0 - interference * 0.5)
        
        # Add to sensory buffer
        self.sensory_buffer.append(memory)
        
        # Attention gate to working memory
        if self._passes_attention_gate(memory):
            self._add_to_working_memory(memory)
            
        return memory
        
    def _calculate_encoding_strength(self, emotional_context: Dict[str, float]) -> float:
        """Stronger encoding for emotional or novel content"""
        emotion_boost = max(emotional_context.values()) * 0.5
        novelty_boost = random.uniform(0.1, 0.3)  # Would calculate actual novelty
        
        base_strength = 0.5
        return min(base_strength + emotion_boost + novelty_boost, 1.0)
        
    def _passes_attention_gate(self, memory: MemoryTrace) -> bool:
        """Attention determines what enters working memory"""
        # Emotional memories get priority
        if max(memory.emotional_tag.values()) > 0.7:
            return True
            
        # Novel content gets attention
        # Simplified - would check against existing memories
        if random.random() < 0.3:
            return True
            
        # Default attention threshold
        return memory.encoding_strength > 0.6
        
    def _add_to_working_memory(self, memory: MemoryTrace):
        """Add to working memory with capacity constraints"""
        if len(self.working_memory) >= 7:
            # Remove least important item
            weakest = min(self.working_memory, key=lambda m: m.encoding_strength)
            self.working_memory.remove(weakest)
            
        memory.consolidation_stage = "working"
        self.working_memory.append(memory)
        
    async def retrieve_memory(self, cue: str, context_embedding: np.ndarray) -> Optional[MemoryTrace]:
        """Retrieve memory with context-dependent recall"""
        all_memories = self.get_all_memories()
        
        if not all_memories:
            return None
            
        # Find best match based on cue and context
        best_match = None
        best_score = 0.0
        
        for memory in all_memories:
            # Calculate retrieval strength
            content_match = self._calculate_content_match(cue, memory.content)
            context_match = InterferencePattern._calculate_similarity(
                context_embedding, memory.context_embedding
            )
            
            # Forgetting curve
            time_factor = self._calculate_forgetting(memory)
            
            # Emotional memories are easier to recall
            emotion_boost = max(memory.emotional_tag.values()) * 0.2
            
            score = (content_match * 0.4 + context_match * 0.4 + emotion_boost) * time_factor
            
            if score > best_score:
                best_score = score
                best_match = memory
                
        if best_match and best_score > 0.3:  # Retrieval threshold
            # Successful retrieval strengthens memory
            best_match.access_count += 1
            best_match.last_access = datetime.now()
            best_match.encoding_strength = min(best_match.encoding_strength * 1.1, 1.0)
            
            return best_match
            
        return None
        
    def _calculate_forgetting(self, memory: MemoryTrace) -> float:
        """Ebbinghaus forgetting curve"""
        time_elapsed = (datetime.now() - memory.timestamp).total_seconds() / 3600  # Hours
        
        # Modified by consolidation stage
        retention_rate = {
            "sensory": 0.01,
            "working": 0.1,
            "long-term": 0.5
        }[memory.consolidation_stage]
        
        # Forgetting formula
        retention = memory.encoding_strength * np.exp(-time_elapsed / (retention_rate * 24))
        
        # Spaced repetition bonus
        if memory.access_count > 0:
            retention *= (1 + 0.1 * memory.access_count)
            
        return min(retention, 1.0)
        
    async def _forgetting_process(self):
        """Background process that removes forgotten memories"""
        while True:
            await asyncio.sleep(3600)  # Check hourly
            
            all_memories = self.get_all_memories()
            for memory in all_memories:
                retention = self._calculate_forgetting(memory)
                
                if retention < 0.1:  # Forgotten threshold
                    # Remove from appropriate store
                    if memory in self.working_memory:
                        self.working_memory.remove(memory)
                    elif memory in self.long_term_memory:
                        self.long_term_memory.remove(memory)
                        
    def get_consolidation_candidates(self) -> List[MemoryTrace]:
        """Get memories ready for consolidation"""
        candidates = []
        
        for memory in self.working_memory:
            # Rehearsed memories
            if memory.access_count >= 3:
                candidates.append(memory)
            # Emotional memories
            elif max(memory.emotional_tag.values()) > 0.7:
                candidates.append(memory)
            # Old working memories
            elif (datetime.now() - memory.timestamp).total_seconds() > 3600:
                candidates.append(memory)
                
        return candidates
        
    def get_emotional_memories(self) -> List[MemoryTrace]:
        """Get memories with strong emotional content"""
        all_memories = self.get_all_memories()
        return [m for m in all_memories if max(m.emotional_tag.values()) > 0.5]
        
    async def create_creative_associations(self, memory: MemoryTrace):
        """Create novel associations between memories (dream-like)"""
        all_memories = self.get_all_memories()
        
        # Find semantically distant but emotionally similar memories
        candidates = []
        for other in all_memories:
            if other == memory:
                continue
                
            semantic_distance = 1.0 - InterferencePattern._calculate_similarity(
                memory.context_embedding, other.context_embedding
            )
            
            emotional_similarity = self._calculate_emotional_similarity(
                memory.emotional_tag, other.emotional_tag
            )
            
            # We want high semantic distance but emotional similarity
            if semantic_distance > 0.7 and emotional_similarity > 0.5:
                candidates.append(other)
                
        # Create associations with top candidates
        for candidate in candidates[:3]:
            # This would create actual associations in the memory graph
            pass
            
    def _calculate_emotional_similarity(self, emotions1: Dict[str, float], 
                                      emotions2: Dict[str, float]) -> float:
        """Calculate similarity between emotional states"""
        keys = set(emotions1.keys()) | set(emotions2.keys())
        
        similarity = 0.0
        for key in keys:
            val1 = emotions1.get(key, 0.0)
            val2 = emotions2.get(key, 0.0)
            similarity += 1.0 - abs(val1 - val2)
            
        return similarity / len(keys)
        
    def get_all_memories(self) -> List[MemoryTrace]:
        """Get all memories across stores"""
        return list(self.sensory_buffer) + list(self.working_memory) + self.long_term_memory
        
    def _calculate_content_match(self, cue: str, content: str) -> float:
        """Simple content matching - would use embeddings in production"""
        cue_words = set(cue.lower().split())
        content_words = set(content.lower().split())
        
        if not cue_words:
            return 0.0
            
        overlap = len(cue_words & content_words)
        return overlap / len(cue_words)