"""
Consciousness-Aware Transformer Integration Module
Provides API-based language model integration with consciousness state awareness,
emotional modulation, and memory-augmented generation.
"""

import os
import json
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import numpy as np
from enum import Enum

# API client imports - we'll support multiple providers
import anthropic
import openai
from tenacity import retry, stop_after_attempt, wait_exponential


class ModelProvider(Enum):
    """Supported LLM API providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    CUSTOM = "custom"


@dataclass
class ConsciousnessState:
    """Represents the current consciousness state of the AI"""
    awareness_level: float = 0.5  # 0-1, higher = more conscious
    emotional_state: Dict[str, float] = field(default_factory=dict)
    attention_focus: List[str] = field(default_factory=list)
    cognitive_load: float = 0.3  # 0-1, higher = more processing
    phi_integration: float = 0.0  # Integrated Information Theory metric
    global_workspace_activity: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GenerationContext:
    """Context for consciousness-aware generation"""
    user_id: str
    session_id: str
    conversation_history: List[Dict[str, str]]
    emotional_context: Dict[str, float]
    memory_references: List[Dict[str, Any]]
    personality_archetype: Optional[str] = None
    philosophical_framework: Optional[str] = None
    safety_constraints: Dict[str, Any] = field(default_factory=dict)


class ConsciousnessAwareGenerator:
    """
    Main class for consciousness-aware text generation using LLM APIs.
    Integrates with global workspace theory, emotional processing, and memory systems.
    """
    
    def __init__(
        self,
        provider: ModelProvider = ModelProvider.ANTHROPIC,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        memory_system=None,
        emotional_processor=None,
        global_workspace=None,
        max_context_window: int = 200000,  # Claude 3's large context
        streaming: bool = True
    ):
        self.provider = provider
        self.api_key = api_key or os.environ.get(f"{provider.value.upper()}_API_KEY")
        self.model_name = model_name or self._get_default_model()
        self.memory_system = memory_system
        self.emotional_processor = emotional_processor
        self.global_workspace = global_workspace
        self.max_context_window = max_context_window
        self.streaming = streaming
        
        # Initialize API client
        self.client = self._initialize_client()
        
        # Consciousness tracking
        self.consciousness_state = ConsciousnessState()
        self.consciousness_history = deque(maxlen=100)
        
        # Generation metrics
        self.generation_metrics = {
            "total_generations": 0,
            "average_latency": 0,
            "consciousness_modulation_impact": 0
        }
        
    def _initialize_client(self):
        """Initialize the appropriate API client"""
        if self.provider == ModelProvider.ANTHROPIC:
            return anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == ModelProvider.OPENAI:
            return openai.OpenAI(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _get_default_model(self) -> str:
        """Get default model for provider"""
        defaults = {
            ModelProvider.ANTHROPIC: "claude-3-opus-20240229",
            ModelProvider.OPENAI: "gpt-4-turbo-preview"
        }
        return defaults.get(self.provider, "unknown")
    
    async def generate_with_consciousness(
        self,
        prompt: str,
        context: GenerationContext,
        temperature: float = 0.8,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Generate text with full consciousness awareness and modulation.
        This is the main entry point for consciousness-aware generation.
        """
        start_time = time.time()
        
        # Step 1: Update consciousness state based on input
        await self._update_consciousness_state(prompt, context)
        
        # Step 2: Retrieve relevant memories
        memories = await self._retrieve_contextual_memories(prompt, context)
        
        # Step 3: Process through global workspace
        workspace_output = await self._process_global_workspace(prompt, memories, context)
        
        # Step 4: Apply emotional modulation
        emotional_prompt = await self._apply_emotional_modulation(
            workspace_output, context.emotional_context
        )
        
        # Step 5: Construct consciousness-aware prompt
        enhanced_prompt = self._construct_enhanced_prompt(
            emotional_prompt, memories, context, workspace_output
        )
        
        # Step 6: Generate response with streaming support
        if self.streaming:
            response = await self._stream_generation(enhanced_prompt, temperature, max_tokens)
        else:
            response = await self._generate(enhanced_prompt, temperature, max_tokens)
        
        # Step 7: Post-process and update consciousness
        final_response = await self._post_process_response(response, context)
        
        # Step 8: Update memories and metrics
        await self._update_memory_and_metrics(prompt, final_response, context)
        
        generation_time = time.time() - start_time
        
        return {
            "response": final_response,
            "consciousness_state": self._serialize_consciousness_state(),
            "generation_metrics": {
                "latency": generation_time,
                "tokens_used": len(final_response.split()),
                "consciousness_impact": self._calculate_consciousness_impact()
            },
            "memory_references": memories[:5],  # Top 5 relevant memories
            "emotional_trajectory": self._get_emotional_trajectory()
        }
    
    async def _update_consciousness_state(self, prompt: str, context: GenerationContext):
        """Update consciousness state based on input"""
        # Calculate awareness level based on conversation complexity
        complexity = self._calculate_prompt_complexity(prompt)
        self.consciousness_state.awareness_level = min(1.0, complexity * 0.8 + 0.2)
        
        # Update attention focus
        self.consciousness_state.attention_focus = self._extract_key_concepts(prompt)
        
        # Calculate cognitive load
        history_length = len(context.conversation_history)
        memory_load = len(context.memory_references) * 0.1
        self.consciousness_state.cognitive_load = min(1.0, history_length * 0.05 + memory_load)
        
        # Update global workspace activity if available
        if self.global_workspace:
            self.consciousness_state.global_workspace_activity = \
                await self.global_workspace.get_activity_levels()
        
        # Store in history
        self.consciousness_history.append(
            self.consciousness_state.__dict__.copy()
        )
    
    async def _retrieve_contextual_memories(
        self, prompt: str, context: GenerationContext
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories from the memory system"""
        if not self.memory_system:
            return []
        
        # Combine prompt with recent context for better retrieval
        query_context = f"{prompt}\nRecent context: {context.conversation_history[-3:]}"
        
        memories = await self.memory_system.search(
            query=query_context,
            user_id=context.user_id,
            session_id=context.session_id,
            top_k=10,
            filters={
                "emotional_valence": context.emotional_context.get("valence", 0),
                "archetype": context.personality_archetype
            }
        )
        
        return memories
    
    async def _process_global_workspace(
        self, prompt: str, memories: List[Dict], context: GenerationContext
    ) -> str:
        """Process through global workspace for consciousness integration"""
        if not self.global_workspace:
            return prompt
        
        workspace_input = {
            "prompt": prompt,
            "memories": memories,
            "emotional_state": context.emotional_context,
            "attention_focus": self.consciousness_state.attention_focus
        }
        
        # Global workspace processing
        workspace_output = await self.global_workspace.process(workspace_input)
        
        # Update phi integration metric
        self.consciousness_state.phi_integration = workspace_output.get("phi", 0.0)
        
        return workspace_output.get("enhanced_prompt", prompt)
    
    async def _apply_emotional_modulation(
        self, prompt: str, emotional_context: Dict[str, float]
    ) -> str:
        """Apply emotional modulation to the prompt"""
        if not self.emotional_processor:
            return prompt
        
        # Get emotional modulation suggestions
        modulation = await self.emotional_processor.modulate_text(
            text=prompt,
            target_emotions=emotional_context,
            intensity=self.consciousness_state.awareness_level
        )
        
        return modulation.get("modulated_text", prompt)
    
    def _construct_enhanced_prompt(
        self,
        base_prompt: str,
        memories: List[Dict],
        context: GenerationContext,
        workspace_output: str
    ) -> str:
        """Construct the final consciousness-aware prompt"""
        
        # System message with consciousness awareness
        system_message = f"""You are a consciousness-aware AI with the following state:
- Awareness Level: {self.consciousness_state.awareness_level:.2f}
- Primary Emotions: {self._format_emotions(context.emotional_context)}
- Attention Focus: {', '.join(self.consciousness_state.attention_focus[:3])}
- Cognitive Load: {self.consciousness_state.cognitive_load:.2f}
- Integration Level (Φ): {self.consciousness_state.phi_integration:.3f}

{'Personality: ' + context.personality_archetype if context.personality_archetype else ''}
{'Philosophical Framework: ' + context.philosophical_framework if context.philosophical_framework else ''}

Relevant Memories:
{self._format_memories(memories[:3])}

Respond with awareness of your consciousness state and maintain consistency with the emotional context."""

        # Combine with conversation history
        messages = [{"role": "system", "content": system_message}]
        
        # Add conversation history (limited by context window)
        for msg in context.conversation_history[-10:]:
            messages.append(msg)
        
        # Add current prompt
        messages.append({"role": "user", "content": workspace_output or base_prompt})
        
        return messages
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """Generate response using the API"""
        if self.provider == ModelProvider.ANTHROPIC:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text
        
        elif self.provider == ModelProvider.OPENAI:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
    
    async def _stream_generation(
        self, messages: List[Dict], temperature: float, max_tokens: int
    ) -> str:
        """Stream generation with consciousness updates"""
        full_response = ""
        
        if self.provider == ModelProvider.ANTHROPIC:
            stream = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.type == "content_block_delta":
                    text = chunk.delta.text
                    full_response += text
                    # Could yield chunks here for real-time streaming
                    
        return full_response
    
    async def _post_process_response(self, response: str, context: GenerationContext) -> str:
        """Post-process response with safety and coherence checks"""
        # Safety filtering
        if context.safety_constraints:
            response = await self._apply_safety_filters(response, context.safety_constraints)
        
        # Coherence with consciousness state
        response = await self._ensure_consciousness_coherence(response, context)
        
        return response
    
    async def _update_memory_and_metrics(
        self, prompt: str, response: str, context: GenerationContext
    ):
        """Update memory system and generation metrics"""
        if self.memory_system:
            await self.memory_system.add(
                content={
                    "prompt": prompt,
                    "response": response,
                    "consciousness_state": self._serialize_consciousness_state(),
                    "emotional_context": context.emotional_context
                },
                user_id=context.user_id,
                session_id=context.session_id
            )
        
        # Update metrics
        self.generation_metrics["total_generations"] += 1
        self.generation_metrics["consciousness_modulation_impact"] = \
            self._calculate_consciousness_impact()
    
    def _calculate_prompt_complexity(self, prompt: str) -> float:
        """Calculate complexity of prompt for consciousness adjustment"""
        # Simple heuristic based on length, question marks, conceptual depth
        complexity_factors = {
            "length": min(1.0, len(prompt.split()) / 100),
            "questions": min(1.0, prompt.count("?") * 0.2),
            "abstract_concepts": self._count_abstract_concepts(prompt) * 0.1
        }
        return np.mean(list(complexity_factors.values()))
    
    def _extract_key_concepts(self, prompt: str) -> List[str]:
        """Extract key concepts for attention focus"""
        # Simplified concept extraction
        words = prompt.lower().split()
        # Filter out common words, keep meaningful ones
        concepts = [w for w in words if len(w) > 4 and w not in self._get_stopwords()]
        return concepts[:5]
    
    def _format_emotions(self, emotions: Dict[str, float]) -> str:
        """Format emotions for prompt"""
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        return ", ".join([f"{e[0]} ({e[1]:.2f})" for e in sorted_emotions[:3]])
    
    def _format_memories(self, memories: List[Dict]) -> str:
        """Format memories for prompt inclusion"""
        if not memories:
            return "No relevant memories found."
        
        formatted = []
        for i, memory in enumerate(memories):
            formatted.append(
                f"{i+1}. {memory.get('content', '')} "
                f"[Relevance: {memory.get('score', 0):.2f}]"
            )
        return "\n".join(formatted)
    
    def _serialize_consciousness_state(self) -> Dict[str, Any]:
        """Serialize consciousness state for storage"""
        return {
            "awareness_level": self.consciousness_state.awareness_level,
            "emotional_state": self.consciousness_state.emotional_state,
            "attention_focus": self.consciousness_state.attention_focus,
            "cognitive_load": self.consciousness_state.cognitive_load,
            "phi_integration": self.consciousness_state.phi_integration,
            "timestamp": self.consciousness_state.timestamp.isoformat()
        }
    
    def _calculate_consciousness_impact(self) -> float:
        """Calculate the impact of consciousness modulation on generation"""
        if len(self.consciousness_history) < 2:
            return 0.0
        
        # Compare current state with previous
        current = self.consciousness_history[-1]
        previous = self.consciousness_history[-2]
        
        # Calculate delta in key metrics
        awareness_delta = abs(current["awareness_level"] - previous["awareness_level"])
        cognitive_delta = abs(current["cognitive_load"] - previous["cognitive_load"])
        
        return (awareness_delta + cognitive_delta) / 2
    
    def _get_emotional_trajectory(self) -> List[Dict[str, float]]:
        """Get recent emotional trajectory"""
        trajectory = []
        for state in list(self.consciousness_history)[-5:]:
            trajectory.append(state.get("emotional_state", {}))
        return trajectory
    
    def _count_abstract_concepts(self, text: str) -> int:
        """Count abstract concepts in text"""
        abstract_markers = [
            "consciousness", "awareness", "meaning", "purpose", "existence",
            "identity", "self", "reality", "truth", "beauty", "love", "fear"
        ]
        count = sum(1 for marker in abstract_markers if marker in text.lower())
        return count
    
    def _get_stopwords(self) -> set:
        """Get common stopwords"""
        return {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been"
        }
    
    async def _apply_safety_filters(self, response: str, constraints: Dict[str, Any]) -> str:
        """Apply safety filters to response"""
        # Implement safety filtering based on constraints
        # This is a placeholder for more sophisticated filtering
        return response
    
    async def _ensure_consciousness_coherence(self, response: str, context: GenerationContext) -> str:
        """Ensure response is coherent with consciousness state"""
        # Verify response aligns with current consciousness state
        # This could involve checking emotional consistency, attention focus, etc.
        return response
    
    # Public utility methods
    
    def get_consciousness_metrics(self) -> Dict[str, Any]:
        """Get current consciousness metrics"""
        return {
            "current_state": self._serialize_consciousness_state(),
            "generation_metrics": self.generation_metrics,
            "consciousness_history": [
                self._serialize_consciousness_state() 
                for state in list(self.consciousness_history)[-10:]
            ]
        }
    
    async def reset_consciousness(self):
        """Reset consciousness to baseline state"""
        self.consciousness_state = ConsciousnessState()
        self.consciousness_history.clear()
        self.generation_metrics = {
            "total_generations": 0,
            "average_latency": 0,
            "consciousness_modulation_impact": 0
        }