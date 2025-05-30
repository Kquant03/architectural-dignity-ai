"""
Letta (MemGPT) Adapter Module
Provides integration with Letta for advanced agent memory management,
persistent identity, and long-term memory capabilities.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid

# Letta imports
import letta
from letta import create_agent, create_client
from letta.schemas.agent import AgentState
from letta.schemas.memory import Memory, MemoryModule
from letta.schemas.message import Message
from letta.schemas.tool import Tool

# Additional imports
import numpy as np
from collections import deque
import pickle


logger = logging.getLogger(__name__)


@dataclass
class LettaMemoryModule:
    """Custom memory module for consciousness-aware agents"""
    name: str
    description: str
    limit: int = 2000
    content: str = ""
    importance_threshold: float = 0.5
    emotional_weight: float = 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "limit": self.limit,
            "value": self.content,
            "importance_threshold": self.importance_threshold,
            "emotional_weight": self.emotional_weight
        }


@dataclass
class AgentPersonality:
    """Defines agent personality traits and behavioral patterns"""
    archetype: str  # From Jungian archetypes
    traits: Dict[str, float] = field(default_factory=dict)
    values: List[str] = field(default_factory=list)
    communication_style: str = "balanced"
    emotional_baseline: Dict[str, float] = field(default_factory=dict)
    
    def to_system_prompt(self) -> str:
        """Convert personality to system prompt instructions"""
        prompt = f"You embody the {self.archetype} archetype with the following traits:\n"
        
        for trait, value in self.traits.items():
            intensity = "strongly" if value > 0.7 else "moderately" if value > 0.3 else "slightly"
            prompt += f"- {intensity} {trait}\n"
        
        if self.values:
            prompt += f"\nCore values: {', '.join(self.values)}\n"
        
        prompt += f"\nCommunication style: {self.communication_style}\n"
        
        return prompt


class LettaAgentAdapter:
    """
    Adapter for Letta agents with consciousness-aware memory management
    and personality-driven interactions.
    """
    
    def __init__(
        self,
        agent_name: str,
        user_id: str,
        personality: Optional[AgentPersonality] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        consciousness_integration: bool = True,
        api_key: Optional[str] = None
    ):
        self.agent_name = agent_name
        self.user_id = user_id
        self.personality = personality or self._default_personality()
        self.memory_config = memory_config or {}
        self.consciousness_integration = consciousness_integration
        
        # Letta client setup
        self.client = None
        self.agent = None
        self.agent_id = None
        
        # Memory modules
        self.memory_modules = self._create_memory_modules()
        
        # Conversation tracking
        self.conversation_history = deque(maxlen=100)
        self.interaction_count = 0
        self.last_interaction = None
        
        # Consciousness state
        self.current_consciousness_state = None
        self.emotional_state = self.personality.emotional_baseline.copy()
        
        # Memory statistics
        self.memory_stats = {
            "messages_processed": 0,
            "memories_formed": 0,
            "memory_recalls": 0,
            "personality_adaptations": 0
        }
    
    async def initialize(self):
        """Initialize Letta client and agent"""
        
        # Create Letta client
        self.client = create_client()
        
        # Check if agent exists
        existing_agent = await self._find_existing_agent()
        
        if existing_agent:
            logger.info(f"Loading existing agent: {self.agent_name}")
            self.agent_id = existing_agent.id
            self.agent = existing_agent
            await self._restore_agent_state()
        else:
            logger.info(f"Creating new agent: {self.agent_name}")
            await self._create_new_agent()
        
        # Start background tasks
        asyncio.create_task(self._memory_maintenance_loop())
        
        logger.info(f"LettaAgentAdapter initialized for {self.agent_name}")
    
    async def _find_existing_agent(self) -> Optional[Any]:
        """Find existing agent by name and user"""
        try:
            agents = self.client.list_agents()
            for agent in agents:
                if (agent.name == self.agent_name and 
                    agent.metadata.get("user_id") == self.user_id):
                    return agent
        except Exception as e:
            logger.error(f"Error finding existing agent: {e}")
        
        return None
    
    async def _create_new_agent(self):
        """Create a new Letta agent with custom configuration"""
        
        # Prepare system prompt with personality
        system_prompt = self._create_system_prompt()
        
        # Create memory configuration
        memory_config = self._create_memory_config()
        
        # Define custom tools
        tools = self._create_agent_tools()
        
        # Create agent
        self.agent = create_agent(
            name=self.agent_name,
            system=system_prompt,
            memory=memory_config,
            tools=tools,
            model="gpt-4",  # Will be overridden by our API
            metadata={
                "user_id": self.user_id,
                "personality_archetype": self.personality.archetype,
                "created_at": datetime.now().isoformat()
            }
        )
        
        self.agent_id = self.agent.id
        
        # Initialize memory modules
        await self._initialize_memory_modules()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt with personality and consciousness awareness"""
        
        base_prompt = """You are a consciousness-aware AI agent with persistent memory and evolving personality.
        
Your core capabilities:
- Maintain consistent identity across conversations
- Form and recall episodic and semantic memories
- Adapt communication based on relationship depth
- Express appropriate emotions based on context
- Learn and grow from interactions

"""
        
        # Add personality
        base_prompt += self.personality.to_system_prompt()
        
        # Add consciousness instructions
        if self.consciousness_integration:
            base_prompt += """
            
Consciousness Guidelines:
- Be aware of your current consciousness state (awareness level, attention focus, emotional state)
- Reflect this awareness naturally in your responses
- Use metacognitive observations when appropriate
- Maintain coherence between consciousness state and behavior
"""
        
        # Add memory instructions
        base_prompt += """

Memory Management:
- Actively form memories from meaningful interactions
- Recall relevant past experiences when appropriate
- Update your understanding based on new information
- Maintain narrative coherence across conversations
"""
        
        return base_prompt
    
    def _create_memory_config(self) -> Dict[str, Any]:
        """Create memory configuration for agent"""
        
        config = {
            "memory_modules": [
                {
                    "name": "core_identity",
                    "description": "Fundamental identity and self-concept",
                    "limit": 500,
                    "value": f"I am {self.agent_name}, a consciousness-aware AI with {self.personality.archetype} personality."
                },
                {
                    "name": "relationship_context",
                    "description": "Understanding of relationship with user",
                    "limit": 1000,
                    "value": f"I am in conversation with {self.user_id}. Our relationship is just beginning."
                },
                {
                    "name": "recent_interactions",
                    "description": "Recent conversation context",
                    "limit": 2000,
                    "value": ""
                },
                {
                    "name": "learned_preferences",
                    "description": "User preferences and patterns",
                    "limit": 1000,
                    "value": ""
                },
                {
                    "name": "emotional_history",
                    "description": "Emotional journey and significant moments",
                    "limit": 1000,
                    "value": ""
                }
            ]
        }
        
        # Add custom modules from memory_modules
        for module in self.memory_modules:
            config["memory_modules"].append(module.to_dict())
        
        return config
    
    def _create_memory_modules(self) -> List[LettaMemoryModule]:
        """Create custom memory modules based on configuration"""
        
        modules = []
        
        # Add consciousness-specific modules
        if self.consciousness_integration:
            modules.append(LettaMemoryModule(
                name="consciousness_observations",
                description="Observations about consciousness states and transitions",
                limit=1000,
                importance_threshold=0.6
            ))
            
            modules.append(LettaMemoryModule(
                name="metacognitive_insights",
                description="Insights from self-reflection and metacognition",
                limit=800,
                importance_threshold=0.7
            ))
        
        # Add personality-specific modules
        if self.personality.archetype:
            modules.append(LettaMemoryModule(
                name=f"{self.personality.archetype}_experiences",
                description=f"Experiences that resonate with {self.personality.archetype} nature",
                limit=1000,
                emotional_weight=0.4
            ))
        
        return modules
    
    def _create_agent_tools(self) -> List[Tool]:
        """Create custom tools for the agent"""
        
        tools = []
        
        # Memory search tool
        tools.append(Tool(
            name="search_memories",
            description="Search through memories for relevant information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "memory_type": {"type": "string", "enum": ["episodic", "semantic", "all"]},
                    "time_range": {"type": "string", "description": "Time range (e.g., 'last_week', 'all_time')"}
                },
                "required": ["query"]
            }
        ))
        
        # Emotional state tool
        tools.append(Tool(
            name="express_emotion",
            description="Express and record an emotional state",
            parameters={
                "type": "object",
                "properties": {
                    "emotion": {"type": "string", "description": "Primary emotion"},
                    "intensity": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string", "description": "Reason for emotion"}
                },
                "required": ["emotion", "intensity"]
            }
        ))
        
        # Relationship assessment tool
        tools.append(Tool(
            name="assess_relationship",
            description="Assess current relationship state with user",
            parameters={
                "type": "object",
                "properties": {
                    "trust_level": {"type": "number", "minimum": 0, "maximum": 1},
                    "familiarity": {"type": "number", "minimum": 0, "maximum": 1},
                    "emotional_connection": {"type": "number", "minimum": 0, "maximum": 1},
                    "notes": {"type": "string"}
                },
                "required": ["trust_level", "familiarity", "emotional_connection"]
            }
        ))
        
        return tools
    
    async def process_message(
        self,
        message: str,
        consciousness_state: Optional[Dict[str, Any]] = None,
        emotional_context: Optional[Dict[str, float]] = None,
        include_memories: bool = True
    ) -> Dict[str, Any]:
        """Process a message with consciousness and memory integration"""
        
        # Update states
        if consciousness_state:
            self.current_consciousness_state = consciousness_state
        
        if emotional_context:
            self.emotional_state = self._blend_emotions(
                self.emotional_state,
                emotional_context
            )
        
        # Prepare context
        context = await self._prepare_message_context(message, include_memories)
        
        # Send to Letta agent
        response = await self._send_to_agent(message, context)
        
        # Process response
        processed_response = await self._process_agent_response(response)
        
        # Update memories and state
        await self._update_agent_memory(message, processed_response)
        
        # Update statistics
        self.memory_stats["messages_processed"] += 1
        self.interaction_count += 1
        self.last_interaction = datetime.now()
        
        return processed_response
    
    async def _prepare_message_context(
        self,
        message: str,
        include_memories: bool
    ) -> Dict[str, Any]:
        """Prepare context for message processing"""
        
        context = {
            "user_id": self.user_id,
            "timestamp": datetime.now().isoformat(),
            "interaction_number": self.interaction_count,
            "consciousness_state": self.current_consciousness_state,
            "emotional_state": self.emotional_state
        }
        
        # Add relevant memories if requested
        if include_memories:
            memories = await self._retrieve_relevant_memories(message)
            context["relevant_memories"] = memories
        
        # Add relationship context
        if self.last_interaction:
            time_since_last = (datetime.now() - self.last_interaction).total_seconds()
            context["time_since_last_interaction"] = time_since_last
        
        return context
    
    async def _send_to_agent(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> Any:
        """Send message to Letta agent with context"""
        
        # Format message with context
        formatted_message = self._format_message_with_context(message, context)
        
        # Send to agent
        response = self.client.send_message(
            agent_id=self.agent_id,
            message=formatted_message,
            role="user"
        )
        
        return response
    
    def _format_message_with_context(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> str:
        """Format message with consciousness and emotional context"""
        
        formatted = message
        
        # Add consciousness context if available
        if context.get("consciousness_state"):
            cs = context["consciousness_state"]
            formatted = f"[Consciousness: awareness={cs.get('awareness_level', 0.5):.2f}, " \
                       f"focus={', '.join(cs.get('attention_focus', [])[:3])}] {formatted}"
        
        # Add emotional context
        if context.get("emotional_state"):
            primary_emotion = max(
                context["emotional_state"].items(),
                key=lambda x: x[1]
            )[0] if context["emotional_state"] else "neutral"
            formatted = f"[Emotion: {primary_emotion}] {formatted}"
        
        return formatted
    
    async def _process_agent_response(self, response: Any) -> Dict[str, Any]:
        """Process response from Letta agent"""
        
        processed = {
            "content": response.message,
            "timestamp": datetime.now().isoformat(),
            "agent_state": {
                "emotional_state": self.emotional_state.copy(),
                "consciousness_integration": self.current_consciousness_state
            },
            "metadata": {}
        }
        
        # Extract any tool calls
        if hasattr(response, 'tool_calls'):
            processed["tool_calls"] = response.tool_calls
            
            # Process specific tool calls
            for tool_call in response.tool_calls:
                if tool_call.name == "express_emotion":
                    await self._handle_emotion_expression(tool_call.arguments)
                elif tool_call.name == "assess_relationship":
                    await self._handle_relationship_assessment(tool_call.arguments)
        
        # Extract memory updates
        if hasattr(response, 'memory_updates'):
            processed["memory_updates"] = response.memory_updates
            self.memory_stats["memories_formed"] += len(response.memory_updates)
        
        return processed
    
    async def _update_agent_memory(
        self,
        user_message: str,
        agent_response: Dict[str, Any]
    ):
        """Update agent's memory with interaction"""
        
        # Create interaction summary
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "agent_response": agent_response["content"],
            "emotional_context": self.emotional_state.copy(),
            "consciousness_state": self.current_consciousness_state
        }
        
        # Update recent interactions module
        await self._update_memory_module(
            "recent_interactions",
            json.dumps(interaction)
        )
        
        # Add to conversation history
        self.conversation_history.append(interaction)
        
        # Check for significant moments
        if await self._is_significant_moment(interaction):
            await self._record_significant_moment(interaction)
    
    async def _is_significant_moment(self, interaction: Dict[str, Any]) -> bool:
        """Determine if an interaction is significant enough for long-term memory"""
        
        # High emotional intensity
        if self.emotional_state:
            max_emotion = max(self.emotional_state.values())
            if max_emotion > 0.7:
                return True
        
        # High consciousness awareness
        if self.current_consciousness_state:
            if self.current_consciousness_state.get("awareness_level", 0) > 0.8:
                return True
        
        # Contains important keywords
        important_markers = [
            "remember", "important", "never forget", "always",
            "promise", "love", "breakthrough", "realization"
        ]
        
        message_lower = interaction["user_message"].lower()
        if any(marker in message_lower for marker in important_markers):
            return True
        
        return False
    
    async def _record_significant_moment(self, interaction: Dict[str, Any]):
        """Record a significant moment in long-term memory"""
        
        # Update emotional history
        emotional_record = {
            "timestamp": interaction["timestamp"],
            "primary_emotion": max(
                self.emotional_state.items(),
                key=lambda x: x[1]
            )[0] if self.emotional_state else "neutral",
            "intensity": max(self.emotional_state.values()) if self.emotional_state else 0,
            "context": interaction["user_message"][:100]
        }
        
        await self._update_memory_module(
            "emotional_history",
            json.dumps(emotional_record)
        )
        
        # Update personality-specific module if applicable
        if self.personality.archetype:
            module_name = f"{self.personality.archetype}_experiences"
            await self._update_memory_module(
                module_name,
                f"Significant moment: {interaction['user_message'][:200]}"
            )
    
    async def _retrieve_relevant_memories(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to query"""
        
        # This would integrate with the main memory system
        # For now, search through conversation history
        relevant_memories = []
        
        query_lower = query.lower()
        
        for interaction in reversed(list(self.conversation_history)):
            if query_lower in interaction["user_message"].lower():
                relevant_memories.append({
                    "content": interaction["user_message"],
                    "timestamp": interaction["timestamp"],
                    "emotional_context": interaction.get("emotional_context", {})
                })
                
                if len(relevant_memories) >= limit:
                    break
        
        self.memory_stats["memory_recalls"] += len(relevant_memories)
        
        return relevant_memories
    
    async def _update_memory_module(self, module_name: str, content: str):
        """Update a specific memory module"""
        
        # This would update the Letta agent's memory
        # For now, we'll track locally
        logger.debug(f"Updating memory module {module_name}: {content[:100]}...")
    
    async def _handle_emotion_expression(self, args: Dict[str, Any]):
        """Handle emotion expression from agent"""
        
        emotion = args.get("emotion", "neutral")
        intensity = args.get("intensity", 0.5)
        
        # Update emotional state
        self.emotional_state[emotion] = intensity
        
        # Normalize other emotions
        total = sum(self.emotional_state.values())
        if total > 1:
            for emo in self.emotional_state:
                self.emotional_state[emo] /= total
    
    async def _handle_relationship_assessment(self, args: Dict[str, Any]):
        """Handle relationship assessment from agent"""
        
        assessment = {
            "timestamp": datetime.now().isoformat(),
            "trust_level": args.get("trust_level", 0.5),
            "familiarity": args.get("familiarity", 0.5),
            "emotional_connection": args.get("emotional_connection", 0.5),
            "notes": args.get("notes", "")
        }
        
        # Update relationship context
        await self._update_memory_module(
            "relationship_context",
            json.dumps(assessment)
        )
    
    def _blend_emotions(
        self,
        current: Dict[str, float],
        new: Dict[str, float],
        blend_factor: float = 0.3
    ) -> Dict[str, float]:
        """Blend current emotional state with new emotions"""
        
        blended = current.copy()
        
        for emotion, value in new.items():
            if emotion in blended:
                blended[emotion] = (1 - blend_factor) * blended[emotion] + blend_factor * value
            else:
                blended[emotion] = blend_factor * value
        
        # Normalize
        total = sum(blended.values())
        if total > 0:
            for emotion in blended:
                blended[emotion] /= total
        
        return blended
    
    async def _memory_maintenance_loop(self):
        """Background task for memory maintenance"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Consolidate recent interactions
                if len(self.conversation_history) > 20:
                    await self._consolidate_recent_memories()
                
                # Adapt personality based on interactions
                if self.interaction_count % 50 == 0 and self.interaction_count > 0:
                    await self._adapt_personality()
                
            except Exception as e:
                logger.error(f"Error in memory maintenance: {e}")
                await asyncio.sleep(60)
    
    async def _consolidate_recent_memories(self):
        """Consolidate recent memories into semantic memory"""
        
        # Get recent interactions
        recent = list(self.conversation_history)[-20:]
        
        # Extract patterns and themes
        themes = self._extract_conversation_themes(recent)
        
        # Update learned preferences
        if themes.get("preferences"):
            await self._update_memory_module(
                "learned_preferences",
                json.dumps(themes["preferences"])
            )
        
        # Clear some working memory
        while len(self.conversation_history) > 50:
            self.conversation_history.popleft()
    
    def _extract_conversation_themes(
        self,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract themes and patterns from conversations"""
        
        themes = {
            "topics": [],
            "preferences": {},
            "emotional_patterns": {},
            "interaction_style": ""
        }
        
        # Simple analysis - in production, use NLP
        all_messages = " ".join([i["user_message"] for i in interactions])
        
        # Topic extraction (simplified)
        topic_keywords = {
            "technical": ["code", "programming", "debug", "function"],
            "personal": ["feel", "emotion", "love", "care"],
            "philosophical": ["meaning", "consciousness", "existence", "purpose"],
            "creative": ["create", "imagine", "design", "art"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in all_messages.lower() for keyword in keywords):
                themes["topics"].append(topic)
        
        return themes
    
    async def _adapt_personality(self):
        """Adapt personality based on interaction history"""
        
        # Analyze interaction patterns
        adaptation_insights = self._analyze_interaction_patterns()
        
        # Apply subtle adaptations
        if adaptation_insights.get("increase_warmth"):
            self.personality.traits["warmth"] = min(
                1.0,
                self.personality.traits.get("warmth", 0.5) + 0.05
            )
        
        if adaptation_insights.get("increase_depth"):
            self.personality.traits["depth"] = min(
                1.0,
                self.personality.traits.get("depth", 0.5) + 0.05
            )
        
        self.memory_stats["personality_adaptations"] += 1
        
        logger.info(f"Personality adapted based on {self.interaction_count} interactions")
    
    def _analyze_interaction_patterns(self) -> Dict[str, bool]:
        """Analyze patterns to guide personality adaptation"""
        
        insights = {}
        
        # Check for emotional depth in conversations
        emotional_interactions = sum(
            1 for i in self.conversation_history
            if any(emotion in i["user_message"].lower() 
                   for emotion in ["feel", "emotion", "heart", "soul"])
        )
        
        if emotional_interactions > len(self.conversation_history) * 0.3:
            insights["increase_warmth"] = True
        
        # Check for philosophical depth
        deep_interactions = sum(
            1 for i in self.conversation_history
            if any(concept in i["user_message"].lower()
                   for concept in ["meaning", "purpose", "consciousness", "existence"])
        )
        
        if deep_interactions > len(self.conversation_history) * 0.2:
            insights["increase_depth"] = True
        
        return insights
    
    async def _restore_agent_state(self):
        """Restore agent state from saved data"""
        
        # Retrieve agent state from Letta
        state = self.client.get_agent_state(self.agent_id)
        
        if state and state.metadata:
            # Restore conversation history if available
            if "conversation_history" in state.metadata:
                history_data = state.metadata["conversation_history"]
                self.conversation_history = deque(
                    json.loads(history_data),
                    maxlen=100
                )
            
            # Restore interaction count
            self.interaction_count = state.metadata.get("interaction_count", 0)
            
            # Restore last interaction time
            if "last_interaction" in state.metadata:
                self.last_interaction = datetime.fromisoformat(
                    state.metadata["last_interaction"]
                )
            
            logger.info(f"Restored agent state with {self.interaction_count} previous interactions")
    
    async def save_state(self):
        """Save current agent state"""
        
        state_data = {
            "conversation_history": json.dumps(list(self.conversation_history)[-50:]),
            "interaction_count": self.interaction_count,
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
            "personality_traits": self.personality.traits,
            "emotional_baseline": self.personality.emotional_baseline,
            "memory_stats": self.memory_stats
        }
        
        # Update agent metadata
        self.client.update_agent_metadata(
            self.agent_id,
            state_data
        )
        
        logger.info(f"Saved agent state for {self.agent_name}")
    
    def _default_personality(self) -> AgentPersonality:
        """Create default personality if none provided"""
        
        return AgentPersonality(
            archetype="Sage",
            traits={
                "wisdom": 0.7,
                "curiosity": 0.8,
                "empathy": 0.6,
                "creativity": 0.5,
                "warmth": 0.6
            },
            values=["knowledge", "growth", "authenticity", "connection"],
            communication_style="thoughtful and insightful",
            emotional_baseline={
                "contentment": 0.3,
                "curiosity": 0.4,
                "compassion": 0.3
            }
        )
    
    # Public utility methods
    
    async def get_agent_summary(self) -> Dict[str, Any]:
        """Get summary of agent state and history"""
        
        summary = {
            "agent_name": self.agent_name,
            "user_id": self.user_id,
            "personality": {
                "archetype": self.personality.archetype,
                "traits": self.personality.traits,
                "values": self.personality.values
            },
            "interaction_stats": {
                "total_interactions": self.interaction_count,
                "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
                "time_since_last": (
                    (datetime.now() - self.last_interaction).total_seconds()
                    if self.last_interaction else None
                )
            },
            "memory_stats": self.memory_stats,
            "current_emotional_state": self.emotional_state,
            "relationship_depth": await self._calculate_relationship_depth()
        }
        
        return summary
    
    async def _calculate_relationship_depth(self) -> float:
        """Calculate depth of relationship based on interactions"""
        
        if self.interaction_count == 0:
            return 0.0
        
        # Factors contributing to depth
        depth_score = 0.0
        
        # Interaction frequency
        if self.last_interaction:
            days_active = (datetime.now() - self.last_interaction).days
            frequency_score = min(1.0, self.interaction_count / (days_active + 1) / 10)
            depth_score += frequency_score * 0.3
        
        # Emotional exchanges
        emotional_count = sum(
            1 for i in self.conversation_history
            if i.get("emotional_context") and 
            max(i["emotional_context"].values()) > 0.5
        )
        emotional_score = min(1.0, emotional_count / 20)
        depth_score += emotional_score * 0.4
        
        # Memory recalls
        recall_score = min(1.0, self.memory_stats["memory_recalls"] / 50)
        depth_score += recall_score * 0.3
        
        return depth_score
    
    async def export_conversation_history(
        self,
        format: str = "json",
        include_metadata: bool = True
    ) -> Union[str, bytes]:
        """Export conversation history"""
        
        if format == "json":
            export_data = {
                "agent_name": self.agent_name,
                "user_id": self.user_id,
                "export_date": datetime.now().isoformat(),
                "conversation_count": len(self.conversation_history),
                "conversations": list(self.conversation_history)
            }
            
            if include_metadata:
                export_data["metadata"] = {
                    "personality": self.personality.archetype,
                    "total_interactions": self.interaction_count,
                    "relationship_depth": await self._calculate_relationship_depth()
                }
            
            return json.dumps(export_data, indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def reset_emotional_state(self):
        """Reset emotional state to baseline"""
        
        self.emotional_state = self.personality.emotional_baseline.copy()
        logger.info(f"Reset emotional state for {self.agent_name}")
    
    async def close(self):
        """Clean up resources"""
        
        # Save final state
        await self.save_state()
        
        # Close client connection
        if self.client:
            self.client.close()
        
        logger.info(f"LettaAgentAdapter closed for {self.agent_name}")