"""
Cognitive Memory Module
Implements different types of cognitive memory systems including procedural,
declarative, working memory, and prospective memory with consciousness integration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, OrderedDict
import numpy as np
from abc import ABC, abstractmethod
import json
import heapq

# Additional imports
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of cognitive memory"""
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    PROSPECTIVE = "prospective"
    DECLARATIVE = "declarative"
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"
    SENSORY = "sensory"
    META = "meta"


@dataclass
class WorkingMemoryItem:
    """Item in working memory"""
    content: Any
    timestamp: datetime
    attention_weight: float = 1.0
    rehearsal_count: int = 0
    decay_rate: float = 0.1
    associated_items: List[str] = field(default_factory=list)


@dataclass
class ProceduralSkill:
    """Represents a learned procedure or skill"""
    skill_id: str
    name: str
    steps: List[Dict[str, Any]]
    proficiency: float = 0.0  # 0 to 1
    practice_count: int = 0
    last_practiced: Optional[datetime] = None
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 0.0
    
    def practice(self, success: bool = True):
        """Update skill with practice"""
        self.practice_count += 1
        self.last_practiced = datetime.now()
        
        # Update success rate
        self.success_rate = (
            (self.success_rate * (self.practice_count - 1) + (1.0 if success else 0.0)) /
            self.practice_count
        )
        
        # Update proficiency
        if success:
            self.proficiency = min(1.0, self.proficiency + 0.05)
        else:
            self.proficiency = max(0.0, self.proficiency - 0.02)


@dataclass
class ProspectiveTask:
    """Represents a future intention or task"""
    task_id: str
    content: str
    trigger_condition: Union[datetime, str, Callable]
    priority: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    completed: bool = False
    completion_time: Optional[datetime] = None
    reminders_sent: int = 0
    
    def is_triggered(self, context: Dict[str, Any]) -> bool:
        """Check if task should be triggered"""
        if self.completed:
            return False
        
        if isinstance(self.trigger_condition, datetime):
            return datetime.now() >= self.trigger_condition
        elif isinstance(self.trigger_condition, str):
            # Simple string matching in context
            return self.trigger_condition in str(context)
        elif callable(self.trigger_condition):
            return self.trigger_condition(context)
        
        return False


@dataclass
class MetaMemory:
    """Memory about memory - metacognitive knowledge"""
    memory_id: str
    memory_type: MemoryType
    confidence: float = 0.5
    source_credibility: float = 0.5
    encoding_context: Dict[str, Any] = field(default_factory=dict)
    retrieval_history: List[datetime] = field(default_factory=list)
    modification_history: List[Dict[str, Any]] = field(default_factory=list)
    reliability_score: float = 0.5


class CognitiveMemorySystem:
    """
    Comprehensive cognitive memory system implementing multiple types
    of memory with interactions and consciousness integration.
    """
    
    def __init__(
        self,
        working_memory_capacity: int = 7,  # Miller's magical number
        attention_refresh_rate: float = 0.1,  # seconds
        enable_metacognition: bool = True,
        enable_memory_consolidation: bool = True
    ):
        self.working_memory_capacity = working_memory_capacity
        self.attention_refresh_rate = attention_refresh_rate
        self.enable_metacognition = enable_metacognition
        self.enable_memory_consolidation = enable_memory_consolidation
        
        # Initialize memory subsystems
        self.working_memory = WorkingMemory(capacity=working_memory_capacity)
        self.procedural_memory = ProceduralMemory()
        self.prospective_memory = ProspectiveMemory()
        self.declarative_memory = DeclarativeMemory()
        self.sensory_buffer = SensoryBuffer()
        
        # Meta-memory system
        self.meta_memories: Dict[str, MetaMemory] = {} if enable_metacognition else None
        
        # Consciousness integration
        self.consciousness_state = None
        self.attention_focus = []
        
        # Memory interaction tracking
        self.memory_interactions = []
        self.consolidation_queue = asyncio.Queue()
        
        # Statistics
        self.memory_stats = {
            "total_items_processed": 0,
            "working_memory_hits": 0,
            "procedural_executions": 0,
            "prospective_triggers": 0,
            "consolidations": 0
        }
        
        # Start background processes
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background memory maintenance tasks"""
        asyncio.create_task(self._attention_cycle())
        asyncio.create_task(self._prospective_monitoring())
        
        if self.enable_memory_consolidation:
            asyncio.create_task(self._consolidation_cycle())
    
    async def _attention_cycle(self):
        """Background task for attention and working memory management"""
        while True:
            try:
                await asyncio.sleep(self.attention_refresh_rate)
                
                # Update working memory
                await self.working_memory.update_attention(self.attention_focus)
                
                # Decay items
                self.working_memory.apply_decay()
                
                # Check for items to consolidate
                items_to_consolidate = self.working_memory.get_items_for_consolidation()
                
                for item in items_to_consolidate:
                    await self.consolidation_queue.put(item)
                
            except Exception as e:
                logger.error(f"Error in attention cycle: {e}")
    
    async def _prospective_monitoring(self):
        """Monitor for prospective memory triggers"""
        while True:
            try:
                await asyncio.sleep(1.0)  # Check every second
                
                # Get current context
                context = await self._get_current_context()
                
                # Check prospective tasks
                triggered_tasks = self.prospective_memory.check_triggers(context)
                
                for task in triggered_tasks:
                    await self._handle_prospective_trigger(task)
                    self.memory_stats["prospective_triggers"] += 1
                
            except Exception as e:
                logger.error(f"Error in prospective monitoring: {e}")
    
    async def _consolidation_cycle(self):
        """Background task for memory consolidation"""
        while True:
            try:
                # Get item from consolidation queue
                item = await self.consolidation_queue.get()
                
                # Determine consolidation target
                if self._is_procedural_memory(item):
                    await self._consolidate_to_procedural(item)
                else:
                    await self._consolidate_to_declarative(item)
                
                self.memory_stats["consolidations"] += 1
                
            except Exception as e:
                logger.error(f"Error in consolidation cycle: {e}")
    
    async def process_input(
        self,
        content: Any,
        memory_type: MemoryType = MemoryType.WORKING,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process input through appropriate memory system"""
        
        self.memory_stats["total_items_processed"] += 1
        
        # Create meta-memory if enabled
        memory_id = f"{memory_type.value}_{datetime.now().timestamp()}"
        
        if self.enable_metacognition:
            meta_memory = MetaMemory(
                memory_id=memory_id,
                memory_type=memory_type,
                encoding_context=metadata or {}
            )
            self.meta_memories[memory_id] = meta_memory
        
        # Route to appropriate memory system
        if memory_type == MemoryType.WORKING:
            result = await self._process_working_memory(content, metadata)
        elif memory_type == MemoryType.PROCEDURAL:
            result = await self._process_procedural_memory(content, metadata)
        elif memory_type == MemoryType.PROSPECTIVE:
            result = await self._process_prospective_memory(content, metadata)
        elif memory_type in [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.DECLARATIVE]:
            result = await self._process_declarative_memory(content, memory_type, metadata)
        elif memory_type == MemoryType.SENSORY:
            result = await self._process_sensory_memory(content, metadata)
        else:
            result = {"status": "unsupported_memory_type", "type": memory_type.value}
        
        # Record interaction
        self.memory_interactions.append({
            "timestamp": datetime.now(),
            "memory_id": memory_id,
            "memory_type": memory_type.value,
            "result": result.get("status", "processed")
        })
        
        return result
    
    async def _process_working_memory(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process content through working memory"""
        
        # Add to working memory
        item_id = self.working_memory.add_item(content, metadata)
        
        if item_id:
            self.memory_stats["working_memory_hits"] += 1
            
            # Check for rehearsal effects
            if self.working_memory.items[item_id].rehearsal_count > 3:
                # Queue for consolidation
                await self.consolidation_queue.put(self.working_memory.items[item_id])
            
            return {
                "status": "stored_in_working_memory",
                "item_id": item_id,
                "current_capacity": len(self.working_memory.items),
                "max_capacity": self.working_memory_capacity
            }
        else:
            return {
                "status": "working_memory_full",
                "evicted_item": self.working_memory.get_least_attended_item()
            }
    
    async def _process_procedural_memory(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process procedural memory (skills and procedures)"""
        
        if isinstance(content, dict) and "skill_name" in content:
            # Learning new skill
            skill = self.procedural_memory.learn_skill(
                name=content["skill_name"],
                steps=content.get("steps", []),
                context=metadata
            )
            
            return {
                "status": "skill_learned",
                "skill_id": skill.skill_id,
                "initial_proficiency": skill.proficiency
            }
        
        elif isinstance(content, str):
            # Executing skill
            skill_result = await self.procedural_memory.execute_skill(
                skill_name=content,
                context=metadata
            )
            
            if skill_result:
                self.memory_stats["procedural_executions"] += 1
                
            return {
                "status": "skill_executed" if skill_result else "skill_not_found",
                "result": skill_result
            }
        
        return {"status": "invalid_procedural_input"}
    
    async def _process_prospective_memory(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process prospective memory (future intentions)"""
        
        if isinstance(content, dict) and "task" in content:
            # Create prospective task
            task = self.prospective_memory.add_task(
                content=content["task"],
                trigger=content.get("trigger", datetime.now() + timedelta(hours=1)),
                priority=content.get("priority", 0.5)
            )
            
            return {
                "status": "prospective_task_created",
                "task_id": task.task_id,
                "trigger": str(task.trigger_condition)
            }
        
        return {"status": "invalid_prospective_input"}
    
    async def _process_declarative_memory(
        self,
        content: Any,
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process declarative memory (facts and events)"""
        
        # Store in declarative memory
        memory_id = self.declarative_memory.store(
            content=content,
            memory_type=memory_type,
            metadata=metadata
        )
        
        # Check for semantic extraction if episodic
        if memory_type == MemoryType.EPISODIC:
            semantic_content = await self._extract_semantic_content(content)
            if semantic_content:
                self.declarative_memory.store(
                    content=semantic_content,
                    memory_type=MemoryType.SEMANTIC,
                    metadata={"source": memory_id}
                )
        
        return {
            "status": "stored_in_declarative_memory",
            "memory_id": memory_id,
            "memory_type": memory_type.value
        }
    
    async def _process_sensory_memory(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process sensory memory (brief sensory traces)"""
        
        # Add to sensory buffer
        trace_id = self.sensory_buffer.add_trace(content, metadata)
        
        # Check if trace should be promoted to working memory
        if self.sensory_buffer.should_promote(trace_id):
            await self.process_input(
                content,
                MemoryType.WORKING,
                metadata
            )
            
            return {
                "status": "promoted_to_working_memory",
                "trace_id": trace_id
            }
        
        return {
            "status": "stored_in_sensory_buffer",
            "trace_id": trace_id,
            "duration_ms": self.sensory_buffer.trace_duration_ms
        }
    
    async def retrieve(
        self,
        query: Any,
        memory_types: Optional[List[MemoryType]] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve memories across different memory systems"""
        
        if memory_types is None:
            memory_types = list(MemoryType)
        
        all_results = []
        
        # Search each requested memory type
        for mem_type in memory_types:
            if mem_type == MemoryType.WORKING:
                results = self.working_memory.search(query)
                all_results.extend([
                    {"type": "working", "content": r.content, "score": r.attention_weight}
                    for r in results
                ])
            
            elif mem_type == MemoryType.PROCEDURAL:
                skills = self.procedural_memory.find_relevant_skills(query)
                all_results.extend([
                    {"type": "procedural", "content": s.name, "score": s.proficiency}
                    for s in skills
                ])
            
            elif mem_type == MemoryType.PROSPECTIVE:
                tasks = self.prospective_memory.search_tasks(query)
                all_results.extend([
                    {"type": "prospective", "content": t.content, "score": t.priority}
                    for t in tasks
                ])
            
            elif mem_type in [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.DECLARATIVE]:
                memories = self.declarative_memory.search(query, mem_type)
                all_results.extend([
                    {"type": mem_type.value, "content": m["content"], "score": m["relevance"]}
                    for m in memories
                ])
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Update meta-memory retrieval history
        if self.enable_metacognition:
            for result in all_results[:max_results]:
                if result.get("memory_id") in self.meta_memories:
                    self.meta_memories[result["memory_id"]].retrieval_history.append(
                        datetime.now()
                    )
        
        return all_results[:max_results]
    
    async def _get_current_context(self) -> Dict[str, Any]:
        """Get current context for memory operations"""
        
        context = {
            "timestamp": datetime.now(),
            "working_memory_state": self.working_memory.get_state_summary(),
            "attention_focus": self.attention_focus,
            "active_skills": self.procedural_memory.get_active_skills(),
        }
        
        if self.consciousness_state:
            context["consciousness"] = self.consciousness_state
        
        return context
    
    async def _handle_prospective_trigger(self, task: ProspectiveTask):
        """Handle triggered prospective memory task"""
        
        logger.info(f"Prospective task triggered: {task.content}")
        
        # Mark as completed
        task.completed = True
        task.completion_time = datetime.now()
        
        # Move to working memory for immediate attention
        await self.process_input(
            f"REMINDER: {task.content}",
            MemoryType.WORKING,
            {"priority": task.priority, "source": "prospective_memory"}
        )
    
    def _is_procedural_memory(self, item: Any) -> bool:
        """Determine if item should become procedural memory"""
        
        # Check for action patterns or skill indicators
        if hasattr(item, 'content'):
            content_str = str(item.content).lower()
            procedural_indicators = [
                "how to", "steps", "procedure", "method",
                "technique", "skill", "practice", "perform"
            ]
            
            return any(indicator in content_str for indicator in procedural_indicators)
        
        return False
    
    async def _consolidate_to_procedural(self, item: WorkingMemoryItem):
        """Consolidate working memory item to procedural memory"""
        
        # Extract steps or procedures from content
        steps = self._extract_procedural_steps(item.content)
        
        if steps:
            skill_name = f"Learned_skill_{datetime.now().timestamp()}"
            
            self.procedural_memory.learn_skill(
                name=skill_name,
                steps=steps,
                context={"source": "consolidation", "rehearsal_count": item.rehearsal_count}
            )
            
            logger.debug(f"Consolidated to procedural memory: {skill_name}")
    
    async def _consolidate_to_declarative(self, item: WorkingMemoryItem):
        """Consolidate working memory item to declarative memory"""
        
        # Determine if episodic or semantic based on content
        if self._has_temporal_context(item):
            memory_type = MemoryType.EPISODIC
        else:
            memory_type = MemoryType.SEMANTIC
        
        self.declarative_memory.store(
            content=item.content,
            memory_type=memory_type,
            metadata={
                "source": "working_memory_consolidation",
                "rehearsal_count": item.rehearsal_count,
                "attention_weight": item.attention_weight
            }
        )
        
        logger.debug(f"Consolidated to {memory_type.value} memory")
    
    def _extract_procedural_steps(self, content: Any) -> List[Dict[str, Any]]:
        """Extract procedural steps from content"""
        
        # Simplified extraction - in production, use NLP
        steps = []
        
        if isinstance(content, str):
            # Look for numbered or bulleted lists
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if any(line.strip().startswith(marker) for marker in ['1.', '2.', '-', '*', '•']):
                    steps.append({
                        "step": i + 1,
                        "action": line.strip().lstrip('1234567890.-*• '),
                        "type": "extracted"
                    })
        
        return steps
    
    def _has_temporal_context(self, item: Any) -> bool:
        """Check if item has temporal/episodic context"""
        
        temporal_markers = [
            "yesterday", "today", "tomorrow", "last", "when",
            "remember when", "that time", "once", "then"
        ]
        
        content_str = str(item.content if hasattr(item, 'content') else item).lower()
        
        return any(marker in content_str for marker in temporal_markers)
    
    async def _extract_semantic_content(self, episodic_content: Any) -> Optional[str]:
        """Extract semantic knowledge from episodic memory"""
        
        # Simplified extraction - identify facts or general principles
        if isinstance(episodic_content, str):
            # Look for factual statements
            fact_indicators = ["is", "are", "means", "defined as", "always", "never"]
            
            for indicator in fact_indicators:
                if indicator in episodic_content:
                    # Extract the factual component
                    # In production, use more sophisticated NLP
                    return f"Learned: {episodic_content}"
        
        return None
    
    def update_consciousness_state(self, state: Dict[str, Any]):
        """Update consciousness state for memory modulation"""
        
        self.consciousness_state = state
        
        # Update attention based on consciousness
        if "attention_focus" in state:
            self.attention_focus = state["attention_focus"]
        
        # Modulate working memory capacity based on awareness
        if "awareness_level" in state:
            # Higher awareness = better working memory
            awareness = state["awareness_level"]
            self.working_memory.capacity = int(
                self.working_memory_capacity * (0.5 + awareness * 0.5)
            )
    
    def get_memory_profile(self) -> Dict[str, Any]:
        """Get comprehensive memory system profile"""
        
        profile = {
            "statistics": self.memory_stats,
            "working_memory": {
                "current_items": len(self.working_memory.items),
                "capacity": self.working_memory.capacity,
                "average_attention": np.mean([
                    item.attention_weight 
                    for item in self.working_memory.items.values()
                ]) if self.working_memory.items else 0.0
            },
            "procedural_memory": {
                "total_skills": len(self.procedural_memory.skills),
                "average_proficiency": np.mean([
                    skill.proficiency
                    for skill in self.procedural_memory.skills.values()
                ]) if self.procedural_memory.skills else 0.0
            },
            "prospective_memory": {
                "active_tasks": len([
                    t for t in self.prospective_memory.tasks.values()
                    if not t.completed
                ]),
                "completed_tasks": len([
                    t for t in self.prospective_memory.tasks.values()
                    if t.completed
                ])
            },
            "declarative_memory": self.declarative_memory.get_statistics()
        }
        
        if self.enable_metacognition:
            profile["metacognition"] = {
                "total_meta_memories": len(self.meta_memories),
                "average_confidence": np.mean([
                    m.confidence for m in self.meta_memories.values()
                ]) if self.meta_memories else 0.0
            }
        
        return profile


class WorkingMemory:
    """Working memory implementation with limited capacity"""
    
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: OrderedDict[str, WorkingMemoryItem] = OrderedDict()
        self.phonological_loop = deque(maxlen=20)  # Auditory buffer
        self.visuospatial_sketchpad = deque(maxlen=10)  # Visual buffer
    
    def add_item(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Add item to working memory"""
        
        item_id = f"wm_{datetime.now().timestamp()}"
        
        # Check capacity
        if len(self.items) >= self.capacity:
            # Remove least attended item
            self._evict_least_attended()
        
        item = WorkingMemoryItem(
            content=content,
            timestamp=datetime.now()
        )
        
        self.items[item_id] = item
        
        # Categorize into subsystems
        if metadata:
            if metadata.get("modality") == "auditory":
                self.phonological_loop.append(item_id)
            elif metadata.get("modality") == "visual":
                self.visuospatial_sketchpad.append(item_id)
        
        return item_id
    
    def _evict_least_attended(self):
        """Remove item with lowest attention weight"""
        
        if not self.items:
            return
        
        min_item_id = min(
            self.items.keys(),
            key=lambda k: self.items[k].attention_weight
        )
        
        del self.items[min_item_id]
    
    def update_attention(self, focus_items: List[str]):
        """Update attention weights based on focus"""
        
        for item_id, item in self.items.items():
            if item_id in focus_items:
                item.attention_weight = min(1.0, item.attention_weight + 0.1)
                item.rehearsal_count += 1
            else:
                item.attention_weight = max(0.0, item.attention_weight - 0.05)
    
    def apply_decay(self):
        """Apply time-based decay to all items"""
        
        current_time = datetime.now()
        items_to_remove = []
        
        for item_id, item in self.items.items():
            time_elapsed = (current_time - item.timestamp).total_seconds()
            decay_factor = np.exp(-item.decay_rate * time_elapsed)
            
            item.attention_weight *= decay_factor
            
            if item.attention_weight < 0.1:
                items_to_remove.append(item_id)
        
        for item_id in items_to_remove:
            del self.items[item_id]
    
    def search(self, query: Any) -> List[WorkingMemoryItem]:
        """Search working memory"""
        
        results = []
        query_str = str(query).lower()
        
        for item in self.items.values():
            if query_str in str(item.content).lower():
                results.append(item)
        
        return sorted(results, key=lambda x: x.attention_weight, reverse=True)
    
    def get_items_for_consolidation(self) -> List[WorkingMemoryItem]:
        """Get items ready for long-term consolidation"""
        
        candidates = []
        
        for item in self.items.values():
            if item.rehearsal_count >= 5 or item.attention_weight > 0.8:
                candidates.append(item)
        
        return candidates
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of working memory state"""
        
        return {
            "item_count": len(self.items),
            "capacity_used": len(self.items) / self.capacity,
            "phonological_items": len(self.phonological_loop),
            "visuospatial_items": len(self.visuospatial_sketchpad),
            "high_attention_items": sum(
                1 for item in self.items.values()
                if item.attention_weight > 0.7
            )
        }
    
    def get_least_attended_item(self) -> Optional[Any]:
        """Get content of least attended item"""
        
        if not self.items:
            return None
        
        min_item = min(
            self.items.values(),
            key=lambda x: x.attention_weight
        )
        
        return min_item.content


class ProceduralMemory:
    """Procedural memory for skills and procedures"""
    
    def __init__(self):
        self.skills: Dict[str, ProceduralSkill] = {}
        self.skill_hierarchy = {}  # Skill dependencies
        self.active_skills = set()
    
    def learn_skill(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> ProceduralSkill:
        """Learn a new skill or procedure"""
        
        skill_id = f"skill_{name}_{datetime.now().timestamp()}"
        
        skill = ProceduralSkill(
            skill_id=skill_id,
            name=name,
            steps=steps,
            context_requirements=context or {}
        )
        
        self.skills[skill_id] = skill
        
        # Also index by name for easier retrieval
        self.skills[name] = skill
        
        return skill
    
    async def execute_skill(
        self,
        skill_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute a learned skill"""
        
        if skill_name not in self.skills:
            return None
        
        skill = self.skills[skill_name]
        
        # Check context requirements
        if not self._check_context_requirements(skill, context):
            return None
        
        # Mark as active
        self.active_skills.add(skill.skill_id)
        
        # Execute steps (simplified)
        results = []
        success = True
        
        for step in skill.steps:
            try:
                # In real implementation, execute actual procedures
                step_result = {
                    "step": step,
                    "status": "completed",
                    "timestamp": datetime.now()
                }
                results.append(step_result)
                
            except Exception as e:
                success = False
                results.append({
                    "step": step,
                    "status": "failed",
                    "error": str(e)
                })
                break
        
        # Update skill proficiency
        skill.practice(success)
        
        # Remove from active
        self.active_skills.discard(skill.skill_id)
        
        return {
            "skill": skill_name,
            "success": success,
            "proficiency": skill.proficiency,
            "results": results
        }
    
    def _check_context_requirements(
        self,
        skill: ProceduralSkill,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if context meets skill requirements"""
        
        if not skill.context_requirements:
            return True
        
        if not context:
            return False
        
        # Simple requirement checking
        for req_key, req_value in skill.context_requirements.items():
            if req_key not in context:
                return False
            
            if context[req_key] != req_value:
                return False
        
        return True
    
    def find_relevant_skills(self, query: Any) -> List[ProceduralSkill]:
        """Find skills relevant to query"""
        
        query_str = str(query).lower()
        relevant = []
        
        for skill in self.skills.values():
            if isinstance(skill, ProceduralSkill):  # Skip duplicate name entries
                if query_str in skill.name.lower():
                    relevant.append(skill)
                elif any(query_str in str(step).lower() for step in skill.steps):
                    relevant.append(skill)
        
        # Sort by proficiency
        relevant.sort(key=lambda x: x.proficiency, reverse=True)
        
        return relevant
    
    def get_active_skills(self) -> List[str]:
        """Get currently active skills"""
        
        return [
            self.skills[skill_id].name
            for skill_id in self.active_skills
            if skill_id in self.skills
        ]


class ProspectiveMemory:
    """Prospective memory for future intentions"""
    
    def __init__(self):
        self.tasks: Dict[str, ProspectiveTask] = {}
        self.task_queue = []  # Priority queue
    
    def add_task(
        self,
        content: str,
        trigger: Union[datetime, str, Callable],
        priority: float = 0.5
    ) -> ProspectiveTask:
        """Add a prospective memory task"""
        
        task_id = f"task_{datetime.now().timestamp()}"
        
        task = ProspectiveTask(
            task_id=task_id,
            content=content,
            trigger_condition=trigger,
            priority=priority
        )
        
        self.tasks[task_id] = task
        
        # Add to priority queue if time-based
        if isinstance(trigger, datetime):
            heapq.heappush(self.task_queue, (trigger, task_id))
        
        return task
    
    def check_triggers(self, context: Dict[str, Any]) -> List[ProspectiveTask]:
        """Check for triggered tasks"""
        
        triggered = []
        
        # Check all tasks
        for task in self.tasks.values():
            if task.is_triggered(context):
                triggered.append(task)
        
        # Check time-based queue
        current_time = datetime.now()
        
        while self.task_queue and self.task_queue[0][0] <= current_time:
            _, task_id = heapq.heappop(self.task_queue)
            
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if not task.completed:
                    triggered.append(task)
        
        return triggered
    
    def search_tasks(self, query: Any) -> List[ProspectiveTask]:
        """Search prospective tasks"""
        
        query_str = str(query).lower()
        matches = []
        
        for task in self.tasks.values():
            if query_str in task.content.lower():
                matches.append(task)
        
        # Sort by priority and completion status
        matches.sort(
            key=lambda x: (x.completed, -x.priority)
        )
        
        return matches


class DeclarativeMemory:
    """Declarative memory for facts and events"""
    
    def __init__(self):
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.semantic_network = {}  # Concept relationships
        self.episodic_timeline = []  # Temporal ordering
    
    def store(
        self,
        content: Any,
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store declarative memory"""
        
        memory_id = f"{memory_type.value}_{datetime.now().timestamp()}"
        
        memory = {
            "id": memory_id,
            "content": content,
            "type": memory_type.value,
            "timestamp": datetime.now(),
            "metadata": metadata or {},
            "access_count": 0
        }
        
        self.memories[memory_id] = memory
        
        # Add to appropriate structure
        if memory_type == MemoryType.EPISODIC:
            self.episodic_timeline.append(memory_id)
        elif memory_type == MemoryType.SEMANTIC:
            self._add_to_semantic_network(memory)
        
        return memory_id
    
    def _add_to_semantic_network(self, memory: Dict[str, Any]):
        """Add semantic memory to concept network"""
        
        # Extract concepts (simplified)
        content_str = str(memory["content"])
        concepts = content_str.lower().split()[:5]  # First 5 words as concepts
        
        for concept in concepts:
            if concept not in self.semantic_network:
                self.semantic_network[concept] = []
            
            self.semantic_network[concept].append(memory["id"])
    
    def search(
        self,
        query: Any,
        memory_type: Optional[MemoryType] = None
    ) -> List[Dict[str, Any]]:
        """Search declarative memories"""
        
        query_str = str(query).lower()
        results = []
        
        for memory in self.memories.values():
            # Filter by type if specified
            if memory_type and memory["type"] != memory_type.value:
                continue
            
            # Simple string matching
            if query_str in str(memory["content"]).lower():
                memory["access_count"] += 1
                
                results.append({
                    "memory_id": memory["id"],
                    "content": memory["content"],
                    "type": memory["type"],
                    "relevance": 1.0  # Simple binary relevance
                })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get declarative memory statistics"""
        
        type_counts = {}
        
        for memory in self.memories.values():
            mem_type = memory["type"]
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
        
        return {
            "total_memories": len(self.memories),
            "type_distribution": type_counts,
            "semantic_concepts": len(self.semantic_network),
            "episodic_memories": len(self.episodic_timeline)
        }


class SensoryBuffer:
    """Brief sensory memory buffer"""
    
    def __init__(self, trace_duration_ms: int = 500):
        self.trace_duration_ms = trace_duration_ms
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.trace_times: Dict[str, datetime] = {}
    
    def add_trace(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add sensory trace"""
        
        trace_id = f"sensory_{datetime.now().timestamp()}"
        
        self.traces[trace_id] = {
            "content": content,
            "metadata": metadata or {},
            "strength": 1.0
        }
        
        self.trace_times[trace_id] = datetime.now()
        
        # Start decay timer
        asyncio.create_task(self._decay_trace(trace_id))
        
        return trace_id
    
    async def _decay_trace(self, trace_id: str):
        """Decay and remove sensory trace"""
        
        await asyncio.sleep(self.trace_duration_ms / 1000)
        
        if trace_id in self.traces:
            del self.traces[trace_id]
            del self.trace_times[trace_id]
    
    def should_promote(self, trace_id: str) -> bool:
        """Check if trace should be promoted to working memory"""
        
        if trace_id not in self.traces:
            return False
        
        # Promote if accessed multiple times or flagged as important
        metadata = self.traces[trace_id].get("metadata", {})
        
        return (
            metadata.get("importance", 0) > 0.7 or
            metadata.get("attention_capture", False)
        )