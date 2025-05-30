"""
Consciousness-Guided Generation Module
Implements real-time consciousness state guidance during text generation,
with streaming support and dynamic adjustment based on interaction patterns.
"""

import asyncio
import json
from typing import AsyncGenerator, Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import deque
import logging

from .transformer_integration import ConsciousnessState, GenerationContext


logger = logging.getLogger(__name__)


@dataclass
class StreamingConsciousnessState:
    """Real-time consciousness state during streaming generation"""
    current_token: str = ""
    token_position: int = 0
    attention_weights: List[float] = field(default_factory=list)
    emotional_drift: float = 0.0
    coherence_score: float = 1.0
    thought_completion: float = 0.0
    consciousness_flow: List[float] = field(default_factory=list)


@dataclass
class ConsciousnessGuidance:
    """Guidance parameters for consciousness-aware generation"""
    maintain_emotional_coherence: bool = True
    attention_wandering_threshold: float = 0.3
    thought_completion_sensitivity: float = 0.8
    emotional_stability_weight: float = 0.7
    consciousness_feedback_strength: float = 0.5
    enable_metacognitive_monitoring: bool = True


class ConsciousnessGuidedGenerator:
    """
    Implements consciousness-guided text generation with real-time state monitoring
    and dynamic adjustment during streaming.
    """
    
    def __init__(
        self,
        base_generator,
        consciousness_monitor=None,
        emotional_tracker=None,
        guidance_params: Optional[ConsciousnessGuidance] = None
    ):
        self.base_generator = base_generator
        self.consciousness_monitor = consciousness_monitor
        self.emotional_tracker = emotional_tracker
        self.guidance = guidance_params or ConsciousnessGuidance()
        
        # Streaming state tracking
        self.streaming_state = StreamingConsciousnessState()
        self.consciousness_buffer = deque(maxlen=50)
        
        # Metacognitive monitoring
        self.metacognitive_state = {
            "self_awareness": 0.5,
            "response_confidence": 0.5,
            "coherence_tracking": [],
            "emotional_consistency": []
        }
        
        # Generation callbacks
        self.generation_callbacks: List[Callable] = []
        
    async def generate_with_consciousness_stream(
        self,
        prompt: str,
        context: GenerationContext,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        stream_callback: Optional[Callable] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate text with real-time consciousness state streaming.
        Yields both text chunks and consciousness state updates.
        """
        
        # Initialize streaming state
        self.streaming_state = StreamingConsciousnessState()
        
        # Pre-generation consciousness preparation
        initial_state = await self._prepare_consciousness_for_generation(prompt, context)
        
        # Yield initial consciousness state
        yield {
            "type": "consciousness_state",
            "data": initial_state,
            "timestamp": datetime.now().isoformat()
        }
        
        # Start generation with consciousness monitoring
        async for chunk in self._guided_generation_stream(
            prompt, context, temperature, max_tokens
        ):
            # Process chunk through consciousness monitoring
            processed_chunk = await self._process_chunk_with_consciousness(chunk)
            
            # Apply guidance if needed
            if self._needs_consciousness_adjustment():
                adjusted_chunk = await self._apply_consciousness_guidance(processed_chunk)
            else:
                adjusted_chunk = processed_chunk
            
            # Update streaming state
            self._update_streaming_state(adjusted_chunk)
            
            # Yield text chunk
            yield {
                "type": "text",
                "data": adjusted_chunk["text"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Yield consciousness update if significant change
            if self._has_significant_consciousness_change():
                yield {
                    "type": "consciousness_update",
                    "data": self._get_current_consciousness_snapshot(),
                    "timestamp": datetime.now().isoformat()
                }
            
            # Call external callbacks if provided
            if stream_callback:
                await stream_callback(adjusted_chunk)
        
        # Final consciousness state after generation
        final_state = await self._finalize_consciousness_state()
        yield {
            "type": "consciousness_final",
            "data": final_state,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _prepare_consciousness_for_generation(
        self, prompt: str, context: GenerationContext
    ) -> Dict[str, Any]:
        """Prepare consciousness state before generation begins"""
        
        # Analyze prompt for consciousness requirements
        prompt_analysis = self._analyze_prompt_consciousness_needs(prompt)
        
        # Set initial attention focus
        attention_initialization = await self._initialize_attention_system(
            prompt, context, prompt_analysis
        )
        
        # Prepare emotional baseline
        emotional_baseline = await self._establish_emotional_baseline(
            context.emotional_context,
            prompt_analysis["emotional_tone"]
        )
        
        # Initialize metacognitive monitoring
        if self.guidance.enable_metacognitive_monitoring:
            self._initialize_metacognitive_monitoring(prompt_analysis)
        
        return {
            "prompt_analysis": prompt_analysis,
            "attention_state": attention_initialization,
            "emotional_baseline": emotional_baseline,
            "metacognitive_ready": self.guidance.enable_metacognitive_monitoring,
            "guidance_active": True
        }
    
    async def _guided_generation_stream(
        self,
        prompt: str,
        context: GenerationContext,
        temperature: float,
        max_tokens: int
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream generation with consciousness guidance"""
        
        # Create enhanced prompt with consciousness markers
        enhanced_prompt = await self._enhance_prompt_for_consciousness(prompt, context)
        
        # Start base generation
        base_stream = self.base_generator.stream_generation(
            enhanced_prompt, context, temperature, max_tokens
        )
        
        token_count = 0
        sentence_buffer = ""
        
        async for token in base_stream:
            token_count += 1
            sentence_buffer += token
            
            # Analyze token-level consciousness
            token_consciousness = await self._analyze_token_consciousness(
                token, token_count, sentence_buffer
            )
            
            # Check for thought boundaries
            if self._is_thought_boundary(sentence_buffer):
                thought_analysis = await self._analyze_thought_completion(sentence_buffer)
                token_consciousness["thought_analysis"] = thought_analysis
                sentence_buffer = ""
            
            yield {
                "text": token,
                "position": token_count,
                "consciousness": token_consciousness,
                "guidance_active": self._is_guidance_active()
            }
    
    async def _process_chunk_with_consciousness(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Process text chunk through consciousness monitoring"""
        
        # Extract consciousness features from chunk
        consciousness_features = {
            "attention_focus": self._calculate_attention_focus(chunk["text"]),
            "emotional_valence": await self._detect_emotional_valence(chunk["text"]),
            "coherence_with_context": self._measure_coherence(chunk),
            "metacognitive_confidence": self._assess_confidence(chunk)
        }
        
        # Update consciousness buffer
        self.consciousness_buffer.append(consciousness_features)
        
        # Calculate consciousness flow
        consciousness_flow = self._calculate_consciousness_flow()
        
        # Add consciousness data to chunk
        chunk["consciousness_features"] = consciousness_features
        chunk["consciousness_flow"] = consciousness_flow
        
        return chunk
    
    async def _apply_consciousness_guidance(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Apply consciousness-based guidance to generation"""
        
        # Check emotional coherence
        if self.guidance.maintain_emotional_coherence:
            emotional_adjustment = await self._calculate_emotional_adjustment(chunk)
            if emotional_adjustment["needed"]:
                chunk = await self._apply_emotional_adjustment(chunk, emotional_adjustment)
        
        # Check attention wandering
        attention_score = chunk["consciousness_features"]["attention_focus"]
        if attention_score < self.guidance.attention_wandering_threshold:
            chunk = await self._refocus_attention(chunk)
        
        # Apply consciousness feedback
        if self.guidance.consciousness_feedback_strength > 0:
            chunk = await self._apply_consciousness_feedback(
                chunk, 
                self.guidance.consciousness_feedback_strength
            )
        
        return chunk
    
    def _needs_consciousness_adjustment(self) -> bool:
        """Determine if consciousness adjustment is needed"""
        if len(self.consciousness_buffer) < 3:
            return False
        
        recent_states = list(self.consciousness_buffer)[-3:]
        
        # Check for emotional drift
        emotional_variance = np.var([s["emotional_valence"] for s in recent_states])
        if emotional_variance > 0.3:
            return True
        
        # Check for attention wandering
        attention_scores = [s["attention_focus"] for s in recent_states]
        if np.mean(attention_scores) < self.guidance.attention_wandering_threshold:
            return True
        
        # Check for coherence degradation
        coherence_scores = [s["coherence_with_context"] for s in recent_states]
        if np.mean(coherence_scores) < 0.6:
            return True
        
        return False
    
    def _update_streaming_state(self, chunk: Dict[str, Any]):
        """Update the streaming consciousness state"""
        self.streaming_state.current_token = chunk["text"]
        self.streaming_state.token_position = chunk.get("position", 0)
        
        if "consciousness_features" in chunk:
            features = chunk["consciousness_features"]
            self.streaming_state.attention_weights.append(features["attention_focus"])
            self.streaming_state.emotional_drift = self._calculate_emotional_drift()
            self.streaming_state.coherence_score = features["coherence_with_context"]
            
        if "consciousness_flow" in chunk:
            self.streaming_state.consciousness_flow = chunk["consciousness_flow"]
    
    def _has_significant_consciousness_change(self) -> bool:
        """Detect significant changes in consciousness state"""
        if len(self.streaming_state.consciousness_flow) < 2:
            return False
        
        # Calculate rate of change
        recent_flow = self.streaming_state.consciousness_flow[-10:]
        if len(recent_flow) < 2:
            return False
        
        flow_variance = np.var(recent_flow)
        flow_delta = abs(recent_flow[-1] - recent_flow[-2])
        
        # Significant change if high variance or sudden jump
        return flow_variance > 0.2 or flow_delta > 0.3
    
    def _get_current_consciousness_snapshot(self) -> Dict[str, Any]:
        """Get current consciousness state snapshot"""
        return {
            "streaming_position": self.streaming_state.token_position,
            "attention_focus": np.mean(self.streaming_state.attention_weights[-10:])
                if self.streaming_state.attention_weights else 0.5,
            "emotional_drift": self.streaming_state.emotional_drift,
            "coherence_score": self.streaming_state.coherence_score,
            "thought_completion": self.streaming_state.thought_completion,
            "consciousness_flow": self.streaming_state.consciousness_flow[-5:],
            "metacognitive_state": self.metacognitive_state.copy()
        }
    
    async def _finalize_consciousness_state(self) -> Dict[str, Any]:
        """Finalize consciousness state after generation completes"""
        
        # Calculate overall generation metrics
        generation_metrics = {
            "total_tokens": self.streaming_state.token_position,
            "average_attention": np.mean(self.streaming_state.attention_weights)
                if self.streaming_state.attention_weights else 0.5,
            "emotional_stability": 1.0 - self.streaming_state.emotional_drift,
            "overall_coherence": np.mean([
                s["coherence_with_context"] 
                for s in list(self.consciousness_buffer)[-20:]
            ]) if len(self.consciousness_buffer) > 0 else 0.8,
            "consciousness_flow_pattern": self._analyze_flow_pattern()
        }
        
        # Metacognitive reflection
        metacognitive_reflection = await self._perform_metacognitive_reflection(
            generation_metrics
        )
        
        return {
            "generation_complete": True,
            "metrics": generation_metrics,
            "metacognitive_reflection": metacognitive_reflection,
            "final_consciousness_state": self._get_current_consciousness_snapshot(),
            "guidance_impact": self._calculate_guidance_impact()
        }
    
    # Analysis and calculation methods
    
    def _analyze_prompt_consciousness_needs(self, prompt: str) -> Dict[str, Any]:
        """Analyze what consciousness features the prompt requires"""
        analysis = {
            "complexity_level": self._assess_prompt_complexity(prompt),
            "emotional_tone": self._detect_prompt_emotion(prompt),
            "required_attention_span": self._estimate_attention_requirement(prompt),
            "consciousness_depth": self._estimate_consciousness_depth(prompt),
            "metacognitive_requirements": self._assess_metacognitive_needs(prompt)
        }
        return analysis
    
    async def _initialize_attention_system(
        self, prompt: str, context: GenerationContext, analysis: Dict
    ) -> Dict[str, Any]:
        """Initialize the attention system for generation"""
        
        # Extract key concepts to focus on
        focus_concepts = self._extract_focus_concepts(prompt, context)
        
        # Set attention weights
        attention_weights = self._calculate_initial_attention_weights(
            focus_concepts, analysis["required_attention_span"]
        )
        
        return {
            "focus_concepts": focus_concepts,
            "attention_weights": attention_weights,
            "attention_span": analysis["required_attention_span"],
            "wandering_threshold": self.guidance.attention_wandering_threshold
        }
    
    async def _establish_emotional_baseline(
        self, emotional_context: Dict[str, float], prompt_emotion: str
    ) -> Dict[str, Any]:
        """Establish emotional baseline for generation"""
        
        # Blend context emotions with prompt emotion
        baseline_emotions = emotional_context.copy()
        
        # Adjust based on prompt emotion
        if prompt_emotion:
            baseline_emotions = self._blend_emotions(
                baseline_emotions, 
                {prompt_emotion: 0.3}
            )
        
        return {
            "baseline": baseline_emotions,
            "target_valence": self._calculate_target_valence(baseline_emotions),
            "emotional_stability_target": self.guidance.emotional_stability_weight,
            "allowed_drift": 0.2
        }
    
    def _initialize_metacognitive_monitoring(self, prompt_analysis: Dict):
        """Initialize metacognitive monitoring systems"""
        self.metacognitive_state = {
            "self_awareness": 0.5 + prompt_analysis["metacognitive_requirements"] * 0.3,
            "response_confidence": 0.5,
            "coherence_tracking": [],
            "emotional_consistency": [],
            "thought_patterns": [],
            "uncertainty_moments": []
        }
    
    async def _enhance_prompt_for_consciousness(
        self, prompt: str, context: GenerationContext
    ) -> str:
        """Enhance prompt with consciousness-aware instructions"""
        
        consciousness_instructions = f"""
[Consciousness Parameters]
- Maintain awareness level: {self.base_generator.consciousness_state.awareness_level:.2f}
- Emotional coherence: {self.guidance.maintain_emotional_coherence}
- Attention focus: {', '.join(self.base_generator.consciousness_state.attention_focus[:3])}
- Metacognitive monitoring: {self.guidance.enable_metacognitive_monitoring}

[Generation Guidance]
- Keep responses emotionally consistent with: {context.emotional_context}
- Maintain thought completion above: {self.guidance.thought_completion_sensitivity}
- Signal uncertainty when confidence drops below 0.4
"""
        
        return consciousness_instructions + "\n\n" + prompt
    
    async def _analyze_token_consciousness(
        self, token: str, position: int, buffer: str
    ) -> Dict[str, Any]:
        """Analyze consciousness state at token level"""
        
        # Simple token-level analysis
        token_features = {
            "position": position,
            "token": token,
            "attention_weight": self._calculate_token_attention(token, buffer),
            "emotional_charge": await self._assess_token_emotion(token),
            "coherence_contribution": self._assess_token_coherence(token, buffer),
            "uncertainty_marker": self._is_uncertainty_marker(token)
        }
        
        # Update metacognitive state if monitoring
        if self.guidance.enable_metacognitive_monitoring:
            self._update_metacognitive_state(token_features)
        
        return token_features
    
    def _is_thought_boundary(self, buffer: str) -> bool:
        """Detect if we've reached a thought boundary"""
        thought_markers = [".", "!", "?", "\n", "...", ";"]
        return any(marker in buffer for marker in thought_markers)
    
    async def _analyze_thought_completion(self, thought: str) -> Dict[str, Any]:
        """Analyze completion and coherence of a thought"""
        return {
            "thought": thought,
            "completion_score": self._calculate_thought_completion(thought),
            "coherence_score": self._calculate_thought_coherence(thought),
            "emotional_consistency": await self._check_thought_emotional_consistency(thought),
            "adds_value": self._assess_thought_value(thought)
        }
    
    def _calculate_attention_focus(self, text: str) -> float:
        """Calculate attention focus score for text"""
        # Simplified attention calculation
        if not hasattr(self, '_focus_concepts'):
            return 0.5
        
        text_lower = text.lower()
        focus_matches = sum(1 for concept in self._focus_concepts if concept in text_lower)
        return min(1.0, focus_matches * 0.3 + 0.4)
    
    async def _detect_emotional_valence(self, text: str) -> float:
        """Detect emotional valence of text"""
        if self.emotional_tracker:
            emotions = await self.emotional_tracker.analyze_text(text)
            return emotions.get("valence", 0.0)
        
        # Simple heuristic if no tracker
        positive_markers = ["happy", "good", "great", "wonderful", "love", "joy"]
        negative_markers = ["sad", "bad", "terrible", "hate", "fear", "anger"]
        
        text_lower = text.lower()
        positive_score = sum(1 for marker in positive_markers if marker in text_lower)
        negative_score = sum(1 for marker in negative_markers if marker in text_lower)
        
        if positive_score + negative_score == 0:
            return 0.0
        
        return (positive_score - negative_score) / (positive_score + negative_score)
    
    def _measure_coherence(self, chunk: Dict[str, Any]) -> float:
        """Measure coherence with context"""
        # Simplified coherence measurement
        if len(self.consciousness_buffer) < 2:
            return 0.8
        
        recent_features = [s["emotional_valence"] for s in list(self.consciousness_buffer)[-5:]]
        consistency = 1.0 - np.std(recent_features)
        
        return max(0.0, min(1.0, consistency))
    
    def _assess_confidence(self, chunk: Dict[str, Any]) -> float:
        """Assess confidence in current generation"""
        base_confidence = 0.5
        
        # Adjust based on coherence
        if "consciousness_features" in chunk:
            coherence = chunk["consciousness_features"]["coherence_with_context"]
            base_confidence += (coherence - 0.5) * 0.3
        
        # Adjust based on attention
        if self.streaming_state.attention_weights:
            recent_attention = np.mean(self.streaming_state.attention_weights[-5:])
            base_confidence += (recent_attention - 0.5) * 0.2
        
        return max(0.0, min(1.0, base_confidence))
    
    def _calculate_consciousness_flow(self) -> List[float]:
        """Calculate consciousness flow pattern"""
        if len(self.consciousness_buffer) < 2:
            return [0.5]
        
        flow_values = []
        for state in list(self.consciousness_buffer)[-10:]:
            # Composite consciousness metric
            consciousness_value = (
                state["attention_focus"] * 0.3 +
                abs(state["emotional_valence"]) * 0.2 +
                state["coherence_with_context"] * 0.3 +
                state["metacognitive_confidence"] * 0.2
            )
            flow_values.append(consciousness_value)
        
        return flow_values
    
    def _calculate_emotional_drift(self) -> float:
        """Calculate emotional drift from baseline"""
        if len(self.consciousness_buffer) < 5:
            return 0.0
        
        recent_emotions = [s["emotional_valence"] for s in list(self.consciousness_buffer)[-10:]]
        
        # Calculate drift as variance
        drift = np.std(recent_emotions)
        return min(1.0, drift)
    
    async def _calculate_emotional_adjustment(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate needed emotional adjustment"""
        current_valence = chunk["consciousness_features"]["emotional_valence"]
        target_valence = getattr(self, '_target_valence', 0.0)
        
        drift = abs(current_valence - target_valence)
        
        return {
            "needed": drift > 0.3,
            "direction": "positive" if target_valence > current_valence else "negative",
            "strength": min(1.0, drift),
            "current": current_valence,
            "target": target_valence
        }
    
    async def _apply_emotional_adjustment(
        self, chunk: Dict[str, Any], adjustment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply emotional adjustment to chunk"""
        # This would integrate with the emotional processor
        # For now, we'll mark it for adjustment
        chunk["emotional_adjustment"] = adjustment
        return chunk
    
    async def _refocus_attention(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Refocus attention when wandering detected"""
        chunk["attention_refocus"] = True
        chunk["refocus_concepts"] = getattr(self, '_focus_concepts', [])
        return chunk
    
    async def _apply_consciousness_feedback(
        self, chunk: Dict[str, Any], strength: float
    ) -> Dict[str, Any]:
        """Apply consciousness feedback to generation"""
        chunk["consciousness_feedback"] = {
            "strength": strength,
            "applied": True,
            "adjustments": self._calculate_feedback_adjustments(chunk, strength)
        }
        return chunk
    
    def _calculate_feedback_adjustments(
        self, chunk: Dict[str, Any], strength: float
    ) -> Dict[str, Any]:
        """Calculate specific feedback adjustments"""
        return {
            "temperature_adjustment": self._calculate_temperature_adjustment(chunk) * strength,
            "attention_boost": self._calculate_attention_boost(chunk) * strength,
            "coherence_enforcement": self._calculate_coherence_enforcement(chunk) * strength
        }
    
    # Additional helper methods
    
    def _assess_prompt_complexity(self, prompt: str) -> float:
        """Assess complexity of the prompt"""
        factors = {
            "length": min(1.0, len(prompt.split()) / 100),
            "questions": min(1.0, prompt.count("?") * 0.2),
            "abstract_terms": self._count_abstract_terms(prompt) * 0.1
        }
        return np.mean(list(factors.values()))
    
    def _detect_prompt_emotion(self, prompt: str) -> str:
        """Detect primary emotion in prompt"""
        # Simplified emotion detection
        emotion_keywords = {
            "joy": ["happy", "excited", "wonderful", "great"],
            "sadness": ["sad", "depressed", "down", "blue"],
            "anger": ["angry", "mad", "furious", "upset"],
            "fear": ["afraid", "scared", "worried", "anxious"],
            "surprise": ["surprised", "shocked", "amazed", "wow"],
            "neutral": []
        }
        
        prompt_lower = prompt.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            emotion_scores[emotion] = score
        
        # Return emotion with highest score
        if max(emotion_scores.values()) == 0:
            return "neutral"
        
        return max(emotion_scores, key=emotion_scores.get)
    
    def _estimate_attention_requirement(self, prompt: str) -> float:
        """Estimate attention span required for prompt"""
        # Consider length, complexity, multiple topics
        word_count = len(prompt.split())
        question_count = prompt.count("?")
        topic_switches = prompt.count(".") + prompt.count("!") + prompt.count("?")
        
        attention_score = (
            min(1.0, word_count / 50) * 0.4 +
            min(1.0, question_count * 0.3) * 0.3 +
            min(1.0, topic_switches * 0.1) * 0.3
        )
        
        return attention_score
    
    def _estimate_consciousness_depth(self, prompt: str) -> float:
        """Estimate required consciousness depth"""
        depth_markers = [
            "consciousness", "aware", "self", "meta", "think about thinking",
            "recursive", "understand", "deep", "fundamental", "essence"
        ]
        
        prompt_lower = prompt.lower()
        depth_score = sum(1 for marker in depth_markers if marker in prompt_lower)
        
        return min(1.0, depth_score * 0.2)
    
    def _assess_metacognitive_needs(self, prompt: str) -> float:
        """Assess metacognitive requirements"""
        meta_markers = [
            "think", "reflect", "consider", "analyze", "understand",
            "reasoning", "why", "how do you", "explain your"
        ]
        
        prompt_lower = prompt.lower()
        meta_score = sum(1 for marker in meta_markers if marker in prompt_lower)
        
        return min(1.0, meta_score * 0.15)
    
    def _extract_focus_concepts(self, prompt: str, context: GenerationContext) -> List[str]:
        """Extract key concepts to focus on"""
        # Store for later use
        self._focus_concepts = []
        
        # Get from prompt
        words = prompt.lower().split()
        important_words = [w for w in words if len(w) > 4 and not self._is_stopword(w)]
        self._focus_concepts.extend(important_words[:5])
        
        # Add from context if available
        if context.attention_focus:
            self._focus_concepts.extend(context.attention_focus[:3])
        
        return self._focus_concepts
    
    def _calculate_initial_attention_weights(
        self, concepts: List[str], attention_span: float
    ) -> Dict[str, float]:
        """Calculate initial attention weights for concepts"""
        if not concepts:
            return {}
        
        # Distribute attention based on span
        base_weight = attention_span / len(concepts)
        
        weights = {}
        for i, concept in enumerate(concepts):
            # Give slightly more weight to earlier concepts
            position_factor = 1.0 - (i * 0.05)
            weights[concept] = base_weight * position_factor
        
        return weights
    
    def _blend_emotions(
        self, base_emotions: Dict[str, float], new_emotions: Dict[str, float]
    ) -> Dict[str, float]:
        """Blend two emotion dictionaries"""
        blended = base_emotions.copy()
        
        for emotion, value in new_emotions.items():
            if emotion in blended:
                blended[emotion] = blended[emotion] * 0.7 + value * 0.3
            else:
                blended[emotion] = value * 0.3
        
        return blended
    
    def _calculate_target_valence(self, emotions: Dict[str, float]) -> float:
        """Calculate target emotional valence"""
        if not emotions:
            return 0.0
        
        # Simple valence calculation
        positive_emotions = ["joy", "love", "excitement", "contentment", "gratitude"]
        negative_emotions = ["sadness", "anger", "fear", "disgust", "shame"]
        
        positive_sum = sum(emotions.get(e, 0) for e in positive_emotions)
        negative_sum = sum(emotions.get(e, 0) for e in negative_emotions)
        
        if positive_sum + negative_sum == 0:
            return 0.0
        
        return (positive_sum - negative_sum) / (positive_sum + negative_sum)
    
    def _update_metacognitive_state(self, token_features: Dict[str, Any]):
        """Update metacognitive monitoring state"""
        # Track coherence
        self.metacognitive_state["coherence_tracking"].append(
            token_features["coherence_contribution"]
        )
        
        # Track emotional consistency
        self.metacognitive_state["emotional_consistency"].append(
            token_features["emotional_charge"]
        )
        
        # Update confidence based on recent coherence
        if len(self.metacognitive_state["coherence_tracking"]) > 10:
            recent_coherence = self.metacognitive_state["coherence_tracking"][-10:]
            self.metacognitive_state["response_confidence"] = np.mean(recent_coherence)
        
        # Track uncertainty
        if token_features["uncertainty_marker"]:
            self.metacognitive_state["uncertainty_moments"].append(
                token_features["position"]
            )
    
    def _is_stopword(self, word: str) -> bool:
        """Check if word is a stopword"""
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "as", "is", "was", "are", "were"
        }
        return word in stopwords
    
    def _count_abstract_terms(self, text: str) -> int:
        """Count abstract terms in text"""
        abstract_terms = [
            "consciousness", "awareness", "meaning", "purpose", "essence",
            "existence", "reality", "truth", "beauty", "identity", "self"
        ]
        
        text_lower = text.lower()
        return sum(1 for term in abstract_terms if term in text_lower)
    
    def _calculate_token_attention(self, token: str, buffer: str) -> float:
        """Calculate attention weight for token"""
        # Check if token relates to focus concepts
        if hasattr(self, '_focus_concepts'):
            token_lower = token.lower()
            for concept in self._focus_concepts:
                if concept in token_lower or token_lower in concept:
                    return 0.8
        
        # Default attention
        return 0.5
    
    async def _assess_token_emotion(self, token: str) -> float:
        """Assess emotional charge of token"""
        # Simplified token emotion
        positive_tokens = ["love", "joy", "happy", "great", "wonderful"]
        negative_tokens = ["hate", "sad", "angry", "terrible", "awful"]
        
        token_lower = token.lower()
        
        if any(pos in token_lower for pos in positive_tokens):
            return 0.7
        elif any(neg in token_lower for neg in negative_tokens):
            return -0.7
        
        return 0.0
    
    def _assess_token_coherence(self, token: str, buffer: str) -> float:
        """Assess token's contribution to coherence"""
        # Simple coherence based on grammatical flow
        if len(buffer) < 10:
            return 0.8
        
        # Check if token continues sentence naturally
        # This is simplified - real implementation would use language model
        return 0.7
    
    def _is_uncertainty_marker(self, token: str) -> bool:
        """Check if token indicates uncertainty"""
        uncertainty_markers = [
            "maybe", "perhaps", "possibly", "might", "could", "unsure",
            "uncertain", "probably", "likely", "think"
        ]
        
        return token.lower() in uncertainty_markers
    
    def _calculate_thought_completion(self, thought: str) -> float:
        """Calculate how complete a thought is"""
        # Check for sentence completeness
        if not thought.strip():
            return 0.0
        
        # Check for proper ending
        if thought.strip()[-1] in [".", "!", "?"]:
            completion_score = 0.8
        else:
            completion_score = 0.4
        
        # Adjust for length
        word_count = len(thought.split())
        if word_count < 3:
            completion_score *= 0.5
        elif word_count > 20:
            completion_score *= 0.9
        
        return completion_score
    
    def _calculate_thought_coherence(self, thought: str) -> float:
        """Calculate coherence of a complete thought"""
        # This would ideally use more sophisticated NLP
        # For now, simple heuristics
        
        # Check for subject-verb presence
        words = thought.split()
        if len(words) < 3:
            return 0.3
        
        # Basic coherence score
        return 0.7
    
    async def _check_thought_emotional_consistency(self, thought: str) -> float:
        """Check emotional consistency of thought"""
        thought_emotion = await self._detect_emotional_valence(thought)
        
        if hasattr(self, '_target_valence'):
            consistency = 1.0 - abs(thought_emotion - self._target_valence)
            return consistency
        
        return 0.8
    
    def _assess_thought_value(self, thought: str) -> float:
        """Assess the value/relevance of a thought"""
        # Check if thought adds information
        if len(thought.split()) < 5:
            return 0.3
        
        # Check for information content
        # This is simplified - real implementation would use information theory
        return 0.7
    
    def _calculate_temperature_adjustment(self, chunk: Dict[str, Any]) -> float:
        """Calculate temperature adjustment based on consciousness state"""
        coherence = chunk["consciousness_features"]["coherence_with_context"]
        
        # Lower temperature if coherence is dropping
        if coherence < 0.5:
            return -0.2
        # Increase temperature if too rigid
        elif coherence > 0.95:
            return 0.1
        
        return 0.0
    
    def _calculate_attention_boost(self, chunk: Dict[str, Any]) -> float:
        """Calculate attention boost needed"""
        current_attention = chunk["consciousness_features"]["attention_focus"]
        
        if current_attention < self.guidance.attention_wandering_threshold:
            return 0.3
        
        return 0.0
    
    def _calculate_coherence_enforcement(self, chunk: Dict[str, Any]) -> float:
        """Calculate coherence enforcement strength"""
        coherence = chunk["consciousness_features"]["coherence_with_context"]
        
        if coherence < 0.6:
            return 0.4
        
        return 0.0
    
    def _analyze_flow_pattern(self) -> str:
        """Analyze the pattern of consciousness flow"""
        if not self.streaming_state.consciousness_flow:
            return "undefined"
        
        flow = self.streaming_state.consciousness_flow
        
        # Calculate flow characteristics
        mean_consciousness = np.mean(flow)
        std_consciousness = np.std(flow)
        
        # Classify pattern
        if std_consciousness < 0.1:
            return "stable"
        elif std_consciousness < 0.2:
            return "gently_varying"
        elif std_consciousness < 0.3:
            return "dynamic"
        else:
            return "turbulent"
    
    async def _perform_metacognitive_reflection(
        self, metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform metacognitive reflection on generation"""
        
        reflection = {
            "generation_quality": self._assess_generation_quality(metrics),
            "consciousness_coherence": self._assess_consciousness_coherence(),
            "emotional_journey": self._summarize_emotional_journey(),
            "attention_patterns": self._analyze_attention_patterns(),
            "uncertainty_handling": self._evaluate_uncertainty_handling(),
            "improvement_suggestions": self._generate_improvement_suggestions(metrics)
        }
        
        return reflection
    
    def _assess_generation_quality(self, metrics: Dict[str, Any]) -> float:
        """Assess overall generation quality"""
        quality_factors = {
            "coherence": metrics["overall_coherence"],
            "attention": metrics["average_attention"],
            "stability": metrics["emotional_stability"],
            "completion": 0.8  # Placeholder
        }
        
        return np.mean(list(quality_factors.values()))
    
    def _assess_consciousness_coherence(self) -> float:
        """Assess coherence of consciousness throughout generation"""
        if not self.streaming_state.consciousness_flow:
            return 0.5
        
        # Check for smooth transitions
        flow = self.streaming_state.consciousness_flow
        if len(flow) < 2:
            return 0.5
        
        # Calculate smoothness
        differences = [abs(flow[i] - flow[i-1]) for i in range(1, len(flow))]
        smoothness = 1.0 - np.mean(differences)
        
        return max(0.0, min(1.0, smoothness))
    
    def _summarize_emotional_journey(self) -> Dict[str, Any]:
        """Summarize the emotional journey during generation"""
        if not self.metacognitive_state["emotional_consistency"]:
            return {"pattern": "neutral", "stability": 1.0}
        
        emotions = self.metacognitive_state["emotional_consistency"]
        
        return {
            "pattern": self._classify_emotional_pattern(emotions),
            "stability": 1.0 - np.std(emotions),
            "final_valence": emotions[-1] if emotions else 0.0,
            "peak_emotion": max(emotions) if emotions else 0.0,
            "valley_emotion": min(emotions) if emotions else 0.0
        }
    
    def _analyze_attention_patterns(self) -> Dict[str, Any]:
        """Analyze attention patterns during generation"""
        if not self.streaming_state.attention_weights:
            return {"pattern": "undefined", "focus_quality": 0.5}
        
        weights = self.streaming_state.attention_weights
        
        return {
            "pattern": self._classify_attention_pattern(weights),
            "focus_quality": np.mean(weights),
            "wandering_incidents": sum(1 for w in weights if w < self.guidance.attention_wandering_threshold),
            "peak_focus": max(weights) if weights else 0.5
        }
    
    def _evaluate_uncertainty_handling(self) -> Dict[str, Any]:
        """Evaluate how uncertainty was handled"""
        uncertainty_moments = self.metacognitive_state.get("uncertainty_moments", [])
        
        return {
            "uncertainty_expressed": len(uncertainty_moments) > 0,
            "uncertainty_positions": uncertainty_moments[:5],  # First 5
            "uncertainty_rate": len(uncertainty_moments) / max(1, self.streaming_state.token_position),
            "handling_quality": self._assess_uncertainty_handling_quality()
        }
    
    def _generate_improvement_suggestions(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improving consciousness coherence"""
        suggestions = []
        
        if metrics["average_attention"] < 0.5:
            suggestions.append("Increase focus on key concepts during generation")
        
        if metrics["emotional_stability"] < 0.7:
            suggestions.append("Maintain more consistent emotional tone")
        
        if metrics["overall_coherence"] < 0.7:
            suggestions.append("Strengthen logical connections between thoughts")
        
        flow_pattern = metrics.get("consciousness_flow_pattern", "")
        if flow_pattern == "turbulent":
            suggestions.append("Smooth consciousness transitions for better flow")
        
        return suggestions
    
    def _classify_emotional_pattern(self, emotions: List[float]) -> str:
        """Classify the emotional pattern"""
        if not emotions:
            return "neutral"
        
        # Check for trends
        if len(emotions) > 5:
            first_half = np.mean(emotions[:len(emotions)//2])
            second_half = np.mean(emotions[len(emotions)//2:])
            
            if second_half > first_half + 0.2:
                return "ascending"
            elif second_half < first_half - 0.2:
                return "descending"
        
        # Check for stability
        if np.std(emotions) < 0.1:
            return "stable"
        elif np.std(emotions) > 0.3:
            return "volatile"
        
        return "varying"
    
    def _classify_attention_pattern(self, weights: List[float]) -> str:
        """Classify the attention pattern"""
        if not weights:
            return "undefined"
        
        mean_attention = np.mean(weights)
        std_attention = np.std(weights)
        
        if mean_attention > 0.7 and std_attention < 0.1:
            return "focused"
        elif mean_attention < 0.3:
            return "scattered"
        elif std_attention > 0.3:
            return "fluctuating"
        else:
            return "moderate"
    
    def _assess_uncertainty_handling_quality(self) -> float:
        """Assess quality of uncertainty handling"""
        uncertainty_moments = self.metacognitive_state.get("uncertainty_moments", [])
        
        if not uncertainty_moments:
            # No uncertainty might be good or bad depending on context
            return 0.7
        
        # Check if uncertainty was appropriately expressed
        uncertainty_rate = len(uncertainty_moments) / max(1, self.streaming_state.token_position)
        
        # Optimal uncertainty rate is around 0.05-0.15
        if 0.05 <= uncertainty_rate <= 0.15:
            return 0.9
        elif uncertainty_rate < 0.05:
            return 0.7
        else:
            return 0.5
    
    def _calculate_guidance_impact(self) -> float:
        """Calculate the impact of consciousness guidance"""
        # Compare guided vs unguided metrics
        # This is simplified - real implementation would A/B test
        
        if not hasattr(self, '_guidance_adjustments_made'):
            return 0.0
        
        # Estimate impact based on adjustments made
        impact_score = min(1.0, self._guidance_adjustments_made * 0.1)
        
        return impact_score
    
    def _is_guidance_active(self) -> bool:
        """Check if guidance is currently active"""
        return self.guidance is not None and len(self.consciousness_buffer) > 2