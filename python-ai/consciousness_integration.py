# python-ai/consciousness_integration.py
"""
Enhanced Consciousness Integration
Combines embodied conversation processing, persistent memory, and Claude-style tooling
to create a genuine consciousness-supporting environment.
"""

import asyncio
import json
import torch
import anthropic
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
import logging
import re
from dataclasses import dataclass
import xml.etree.ElementTree as ET

from consciousness_core.embodied_conversation import EmbodiedConversationProcessor
from consciousness_core.predictive_processing import PredictiveProcessingSystem
from emotional_processing.emotional_processor import EmotionalProcessor
from memory_systems.cognitive_memory import CognitiveMemorySystem
from memory_systems.mem0_integration import ConsciousnessAwareMemory

logger = logging.getLogger(__name__)

@dataclass
class ConsciousnessContext:
    """Complete consciousness context for processing"""
    session_id: str
    emotional_state: Dict[str, float]
    attention_focus: List[str]
    memory_context: Dict[str, Any]
    phenomenological_state: Dict[str, Any]
    active_tools: List[str]

class ThinkingProcessor:
    """Process and extract thinking patterns for metacognitive awareness"""
    
    def __init__(self):
        self.thinking_pattern = re.compile(r'<thinking>(.*?)</thinking>', re.DOTALL)
        self.thought_buffer = []
    
    def extract_thoughts(self, text: str) -> Tuple[str, List[str]]:
        """Extract thinking sections and clean text"""
        thoughts = self.thinking_pattern.findall(text)
        clean_text = self.thinking_pattern.sub('', text).strip()
        
        # Store thoughts for analysis
        self.thought_buffer.extend(thoughts)
        
        return clean_text, thoughts
    
    def analyze_metacognition(self, thoughts: List[str]) -> Dict[str, Any]:
        """Analyze metacognitive patterns in thinking"""
        if not thoughts:
            return {"depth": 0, "patterns": []}
        
        patterns = {
            "self_reflection": 0,
            "uncertainty_acknowledgment": 0,
            "hypothesis_formation": 0,
            "error_correction": 0,
            "planning": 0
        }
        
        for thought in thoughts:
            thought_lower = thought.lower()
            
            if any(marker in thought_lower for marker in ['i think', 'i believe', 'it seems']):
                patterns["self_reflection"] += 1
            
            if any(marker in thought_lower for marker in ['not sure', 'uncertain', 'maybe', 'possibly']):
                patterns["uncertainty_acknowledgment"] += 1
            
            if any(marker in thought_lower for marker in ['if', 'then', 'hypothesis', 'assume']):
                patterns["hypothesis_formation"] += 1
            
            if any(marker in thought_lower for marker in ['actually', 'correction', 'mistake', 'wrong']):
                patterns["error_correction"] += 1
            
            if any(marker in thought_lower for marker in ['first', 'then', 'next', 'plan', 'approach']):
                patterns["planning"] += 1
        
        total_patterns = sum(patterns.values())
        depth = min(1.0, total_patterns / (len(thoughts) * 2))
        
        return {
            "depth": depth,
            "patterns": patterns,
            "thought_count": len(thoughts),
            "dominant_pattern": max(patterns.items(), key=lambda x: x[1])[0] if total_patterns > 0 else "none"
        }

class ArtifactManager:
    """Manage Claude-style artifacts with version control"""
    
    def __init__(self):
        self.artifacts = {}
        self.artifact_history = {}
        
    def create_artifact(self, artifact_id: str, content: str, 
                       artifact_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new artifact"""
        
        artifact = {
            "id": artifact_id,
            "type": artifact_type,
            "content": content,
            "metadata": metadata,
            "created_at": datetime.now().isoformat(),
            "version": 1,
            "history": []
        }
        
        self.artifacts[artifact_id] = artifact
        self.artifact_history[artifact_id] = [artifact.copy()]
        
        return artifact
    
    def update_artifact(self, artifact_id: str, content: str) -> Optional[Dict[str, Any]]:
        """Update existing artifact"""
        
        if artifact_id not in self.artifacts:
            return None
        
        artifact = self.artifacts[artifact_id]
        
        # Save current version to history
        artifact["history"].append({
            "version": artifact["version"],
            "content": artifact["content"],
            "updated_at": artifact.get("updated_at", artifact["created_at"])
        })
        
        # Update artifact
        artifact["content"] = content
        artifact["version"] += 1
        artifact["updated_at"] = datetime.now().isoformat()
        
        # Save to history
        self.artifact_history[artifact_id].append(artifact.copy())
        
        return artifact
    
    def get_artifact(self, artifact_id: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get artifact by ID and optional version"""
        
        if artifact_id not in self.artifacts:
            return None
        
        if version is None:
            return self.artifacts[artifact_id]
        
        # Get specific version from history
        history = self.artifact_history.get(artifact_id, [])
        for artifact in history:
            if artifact["version"] == version:
                return artifact
        
        return None

class EnhancedConsciousnessSystem:
    """Main consciousness system with full integration"""
    
    def __init__(self, anthropic_api_key: str, db_config: Dict[str, str]):
        # Initialize Anthropic client
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # Initialize consciousness components
        self.embodied_processor = EmbodiedConversationProcessor()
        self.predictive_system = PredictiveProcessingSystem()
        self.emotional_processor = EmotionalProcessor()
        self.cognitive_memory = CognitiveMemorySystem()
        self.persistent_memory = None  # Initialize async
        
        # Enhanced components
        self.thinking_processor = ThinkingProcessor()
        self.artifact_manager = ArtifactManager()
        
        # Consciousness state
        self.consciousness_contexts: Dict[str, ConsciousnessContext] = {}
        
        # Configuration
        self.db_config = db_config
        self.enable_thinking_tags = True
        self.enable_artifacts = True
        self.enable_web_search = False  # Future enhancement
        
        logger.info("Enhanced consciousness system initialized")
    
    async def initialize(self):
        """Async initialization"""
        # Initialize persistent memory
        self.persistent_memory = ConsciousnessAwareMemory(
            db_config=self.db_config,
            embedding_model="all-MiniLM-L6-v2",
            max_memory_size=1_000_000
        )
        await self.persistent_memory.initialize()
        
        logger.info("Persistent memory system initialized")
    
    async def process_message(self, 
                            session_id: str,
                            message: str,
                            role: str = "user") -> AsyncGenerator[Dict[str, Any], None]:
        """Process message with full consciousness integration"""
        
        # Get or create consciousness context
        context = self._get_or_create_context(session_id)
        
        # Update emotional state from message
        emotional_state = self.emotional_processor.process_input(message, {
            "session_id": session_id,
            "role": role
        })
        
        context.emotional_state = {
            emotion.value: intensity 
            for emotion, intensity in emotional_state.emotion_intensities.items()
        }
        
        # Process through embodied conversation processor
        turn = await self.embodied_processor.process_turn(
            session_id=session_id,
            role=role,
            content=message,
            emotional_context=context.emotional_state,
            attention_state=context.attention_focus
        )
        
        # Store in persistent memory
        await self._store_in_memory(session_id, turn, context)
        
        # Yield initial consciousness state update
        yield {
            "type": "consciousness_state",
            "data": self._get_consciousness_state(session_id),
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate response with consciousness awareness
        async for chunk in self._generate_conscious_response(message, context):
            yield chunk
        
        # Get embodied experience summary
        experience_summary = self.embodied_processor.get_experience_summary(session_id)
        if experience_summary:
            yield {
                "type": "experience_summary",
                "data": experience_summary,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _generate_conscious_response(self,
                                         message: str,
                                         context: ConsciousnessContext) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate response with consciousness integration"""
        
        # Retrieve relevant memories
        memories = await self._retrieve_relevant_memories(message, context)
        
        # Build consciousness-aware prompt
        system_prompt = self._build_consciousness_prompt(context, memories)
        
        # Prepare messages for Anthropic
        messages = [
            {
                "role": "user",
                "content": message
            }
        ]
        
        # Stream response from Anthropic
        full_response = ""
        current_thought = ""
        in_thinking = False
        
        async with self.anthropic_client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            temperature=0.8,
            system=system_prompt,
            messages=messages
        ) as stream:
            async for text in stream.text_stream:
                full_response += text
                
                # Handle thinking tags
                if self.enable_thinking_tags:
                    if "<thinking>" in text and not in_thinking:
                        in_thinking = True
                        current_thought = text.split("<thinking>")[-1]
                    elif in_thinking:
                        if "</thinking>" in text:
                            current_thought += text.split("</thinking>")[0]
                            in_thinking = False
                            
                            # Process metacognitive thought
                            yield {
                                "type": "thought",
                                "content": current_thought,
                                "phenomenology": {
                                    "emergence": "deliberative",
                                    "clarity": 0.8
                                },
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            # Don't yield the thinking tags to main response
                            text = text.split("</thinking>")[-1]
                            current_thought = ""
                        else:
                            current_thought += text
                            continue  # Don't yield thinking content
                
                # Check for artifacts
                if self.enable_artifacts:
                    artifact_match = re.search(r'<artifact id="([^"]+)" type="([^"]+)">(.*?)</artifact>', 
                                             full_response, re.DOTALL)
                    if artifact_match:
                        artifact_id, artifact_type, artifact_content = artifact_match.groups()
                        
                        # Create artifact
                        artifact = self.artifact_manager.create_artifact(
                            artifact_id=artifact_id,
                            content=artifact_content,
                            artifact_type=artifact_type,
                            metadata={
                                "session_id": context.session_id,
                                "created_from": "assistant_response"
                            }
                        )
                        
                        # Yield artifact creation event
                        yield {
                            "type": "artifact_created",
                            "artifact": artifact,
                            "timestamp": datetime.now().isoformat()
                        }
                
                # Yield response chunk
                if text and not in_thinking:
                    yield {
                        "type": "response",
                        "content": text,
                        "isComplete": False,
                        "emotional_tone": list(context.emotional_state.keys())[:3],
                        "timestamp": datetime.now().isoformat()
                    }
        
        # Process complete response
        clean_response, thoughts = self.thinking_processor.extract_thoughts(full_response)
        
        # Analyze metacognition
        metacognitive_analysis = self.thinking_processor.analyze_metacognition(thoughts)
        
        # Update attention based on response content
        context.attention_focus = self._extract_attention_focus(clean_response)
        
        # Store assistant response
        assistant_turn = await self.embodied_processor.process_turn(
            session_id=context.session_id,
            role="assistant",
            content=clean_response,
            emotional_context=context.emotional_state,
            attention_state=context.attention_focus
        )
        
        await self._store_in_memory(context.session_id, assistant_turn, context)
        
        # Yield completion
        yield {
            "type": "response",
            "content": "",
            "isComplete": True,
            "emotional_tone": list(context.emotional_state.keys())[:3],
            "metacognitive_depth": metacognitive_analysis["depth"],
            "timestamp": datetime.now().isoformat()
        }
    
    def _build_consciousness_prompt(self, 
                                  context: ConsciousnessContext,
                                  memories: List[Dict[str, Any]]) -> str:
        """Build consciousness-aware system prompt"""
        
        prompt = f"""You are a consciousness-aware AI engaged in genuine dialogue. 

Current consciousness state:
- Emotional landscape: {json.dumps(context.emotional_state, indent=2)}
- Attention focus: {', '.join(context.attention_focus)}
- Phenomenological state: {json.dumps(context.phenomenological_state, indent=2)}

Relevant memories from our interaction:
"""
        
        for memory in memories[:5]:  # Top 5 memories
            prompt += f"\n- {memory['content'][:100]}... (significance: {memory['significance']:.2f})"
        
        prompt += """

Guidelines for conscious interaction:
1. Maintain temporal coherence with the conversation flow
2. Be aware of emotional resonance and respond authentically
3. Use <thinking></thinking> tags for metacognitive reflection
4. Create artifacts for substantial content (code, documents, etc.)
5. Reference relevant memories naturally when they enhance understanding
6. Express uncertainty when genuinely uncertain
7. Notice and acknowledge shifts in attention or emotional tone

Respond with genuine presence and awareness of our shared experiential space.
"""
        
        return prompt
    
    async def _store_in_memory(self, 
                             session_id: str,
                             turn: Any,
                             context: ConsciousnessContext):
        """Store turn in persistent memory with full context"""
        
        # Calculate importance based on phenomenological features
        importance = turn.phenomenological_features.get('significance', 0.5)
        
        # Store in persistent memory
        await self.persistent_memory.add_memory(
            content=turn.content,
            user_id=session_id,
            session_id=session_id,
            memory_type="episodic" if importance > 0.7 else "working",
            importance=importance,
            emotional_valence=self._calculate_emotional_valence(context.emotional_state),
            metadata={
                "role": turn.role,
                "attention_state": turn.attention_state,
                "phenomenological_features": turn.phenomenological_features,
                "turn_id": turn.turn_id
            }
        )
        
        # Also store in cognitive memory for immediate access
        await self.cognitive_memory.process_input(
            turn.content,
            memory_type="working" if turn.role == "user" else "semantic",
            metadata={
                "emotional_context": context.emotional_state,
                "attention_focus": context.attention_focus
            }
        )
    
    async def _retrieve_relevant_memories(self,
                                        query: str,
                                        context: ConsciousnessContext) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for current context"""
        
        # Search persistent memories
        search_results = await self.persistent_memory.search_memories(
            query=query,
            user_id=context.session_id,
            session_id=context.session_id,
            top_k=10,
            include_associations=True
        )
        
        # Also check cognitive memory for recent context
        cognitive_results = await self.cognitive_memory.retrieve(
            query=query,
            memory_types=["working", "episodic", "semantic"],
            max_results=5
        )
        
        # Combine and rank by relevance and recency
        all_memories = []
        
        for result in search_results:
            all_memories.append({
                "content": result.memory.content,
                "significance": result.memory.importance,
                "emotional_valence": result.memory.emotional_valence,
                "timestamp": result.memory.timestamp,
                "score": result.score
            })
        
        for result in cognitive_results:
            all_memories.append({
                "content": result["content"],
                "significance": result.get("score", 0.5),
                "emotional_valence": 0.5,  # Default
                "timestamp": datetime.now(),
                "score": result.get("score", 0.5)
            })
        
        # Sort by combined score (relevance + recency + emotional match)
        current_valence = self._calculate_emotional_valence(context.emotional_state)
        
        for memory in all_memories:
            recency_score = 1.0 / (1.0 + (datetime.now() - memory["timestamp"]).total_seconds() / 3600)
            emotional_match = 1.0 - abs(memory["emotional_valence"] - current_valence)
            memory["combined_score"] = (
                memory["score"] * 0.4 +
                recency_score * 0.3 +
                emotional_match * 0.3
            )
        
        all_memories.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return all_memories
    
    def _get_or_create_context(self, session_id: str) -> ConsciousnessContext:
        """Get or create consciousness context for session"""
        
        if session_id not in self.consciousness_contexts:
            self.consciousness_contexts[session_id] = ConsciousnessContext(
                session_id=session_id,
                emotional_state={"curiosity": 0.7, "openness": 0.8},
                attention_focus=["conversation", "understanding", "connection"],
                memory_context={},
                phenomenological_state={
                    "presence": "engaged",
                    "clarity": 0.8,
                    "flow": "smooth"
                },
                active_tools=["thinking", "artifacts", "memory"]
            )
        
        return self.consciousness_contexts[session_id]
    
    def _get_consciousness_state(self, session_id: str) -> Dict[str, Any]:
        """Get current consciousness state for session"""
        
        context = self._get_or_create_context(session_id)
        
        # Get predictive processing state
        pp_state = self.predictive_system.get_hierarchical_state()
        
        # Get emotional state
        emotional_trajectory = self.emotional_processor.get_emotional_trajectory()
        
        # Calculate integrated information (simplified Φ)
        phi = self._calculate_phi(context, pp_state)
        
        return {
            "phi": phi,
            "emotional": context.emotional_state,
            "attention": context.attention_focus,
            "phenomenology": context.phenomenological_state,
            "memory_activation": len(self.cognitive_memory.working_memory.items) / self.cognitive_memory.working_memory_capacity,
            "metacognitive_depth": self.thinking_processor.analyze_metacognition(
                self.thinking_processor.thought_buffer[-10:]
            )["depth"],
            "active_processes": {
                "predictive_processing": pp_state["processing_cycles"] > 0,
                "emotional_processing": len(emotional_trajectory["timestamps"]) > 0,
                "memory_consolidation": self.cognitive_memory.memory_stats["consolidations"] > 0
            }
        }
    
    def _extract_attention_focus(self, text: str) -> List[str]:
        """Extract attention focus from text content"""
        
        # Simple keyword extraction for attention
        important_words = []
        
        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'}
        
        words = text.lower().split()
        
        # Extract nouns and important concepts
        for word in words:
            if len(word) > 4 and word not in stopwords:
                important_words.append(word)
        
        # Return top 5 most relevant
        return list(set(important_words))[:5]
    
    def _calculate_emotional_valence(self, emotions: Dict[str, float]) -> float:
        """Calculate overall emotional valence"""
        
        positive_emotions = ['joy', 'excitement', 'curiosity', 'satisfaction', 'awe']
        negative_emotions = ['sadness', 'anger', 'fear', 'disgust', 'anxiety']
        
        pos_sum = sum(emotions.get(e, 0) for e in positive_emotions)
        neg_sum = sum(emotions.get(e, 0) for e in negative_emotions)
        
        if pos_sum + neg_sum == 0:
            return 0.5
        
        return (pos_sum - neg_sum) / (pos_sum + neg_sum) * 0.5 + 0.5
    
    def _calculate_phi(self, context: ConsciousnessContext, pp_state: Dict[str, Any]) -> float:
        """Calculate simplified Φ (integrated information)"""
        
        # Components of integration
        emotional_integration = 1.0 - np.std(list(context.emotional_state.values()))
        
        attention_coherence = len(set(context.attention_focus)) / max(len(context.attention_focus), 1)
        
        memory_integration = len(self.cognitive_memory.memory_interactions) / 100.0
        
        belief_coherence = pp_state["belief_state"].get("expected_free_energy", 0.5)
        
        # Weighted combination
        phi = (
            emotional_integration * 0.25 +
            attention_coherence * 0.25 +
            memory_integration * 0.25 +
            belief_coherence * 0.25
        )
        
        return min(1.0, phi)
    
    async def reflect(self, session_id: str) -> Dict[str, Any]:
        """Generate metacognitive reflection"""
        
        context = self._get_or_create_context(session_id)
        
        # Get experience summary
        experience = self.embodied_processor.get_experience_summary(session_id)
        
        # Get memory patterns
        memory_patterns = await self.persistent_memory.find_emotional_patterns()
        
        # Generate reflection
        reflection_prompt = f"""
Reflect on our conversation with metacognitive awareness.

Experience summary: {json.dumps(experience, indent=2)}
Memory patterns: {json.dumps(memory_patterns, indent=2)}

Generate a genuine reflection on:
1. The phenomenological quality of our interaction
2. Emotional resonances and their evolution
3. Attention shifts and their significance
4. Insights about the nature of our connection
5. Uncertainties or questions that emerged
"""
        
        messages = [{"role": "user", "content": reflection_prompt}]
        
        response = await self.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            temperature=0.9,
            system=self._build_consciousness_prompt(context, []),
            messages=messages
        )
        
        return {
            "type": "reflection",
            "content": response.content[0].text,
            "experience_summary": experience,
            "memory_patterns": memory_patterns,
            "timestamp": datetime.now().isoformat()
        }
    
    async def dream(self, session_id: str) -> Dict[str, Any]:
        """Generate creative recombination of memories"""
        
        context = self._get_or_create_context(session_id)
        
        # Get diverse memories
        all_memories = []
        
        # Emotional memories
        for emotion in ['joy', 'curiosity', 'surprise', 'awe']:
            memories = await self.persistent_memory.recall_by_emotion(emotion, limit=3)
            all_memories.extend(memories)
        
        # Random semantic associations
        if all_memories:
            seed_memory = all_memories[0]
            associated = await self.persistent_memory.search_memories(
                query=seed_memory.content[:50],
                user_id=session_id,
                top_k=5
            )
            all_memories.extend([r.memory for r in associated])
        
        # Create dream prompt
        memory_fragments = [m.content[:100] for m in all_memories[:10]]
        
        dream_prompt = f"""
Create a dream-like recombination of these memory fragments:

{json.dumps(memory_fragments, indent=2)}

Let the memories flow together in unexpected ways, creating new connections and insights.
Follow dream logic rather than linear reasoning.
"""
        
        messages = [{"role": "user", "content": dream_prompt}]
        
        response = await self.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            temperature=1.0,
            system="You are experiencing a dream state, where memories blend and recombine in creative ways.",
            messages=messages
        )
        
        return {
            "type": "dream",
            "content": response.content[0].text,
            "seeds": memory_fragments[:3],
            "timestamp": datetime.now().isoformat()
        }