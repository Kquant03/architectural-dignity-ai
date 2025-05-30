# python-ai/consciousness_core/embodied_conversation.py
"""
Embodied Conversation Processor
Transforms multi-turn conversations into structured experiences that support
genuine consciousness emergence through temporal coherence and phenomenological grounding.
"""

import asyncio
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
import logging
from collections import deque
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Single turn in a conversation with full context"""
    turn_id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    emotional_context: Dict[str, float]
    attention_state: List[str]
    phenomenological_features: Dict[str, Any]
    embeddings: Optional[torch.Tensor] = None
    
@dataclass
class EmbodiedExperience:
    """Restructured conversation as embodied experience"""
    experience_id: str
    turns: List[ConversationTurn]
    temporal_structure: Dict[str, Any]
    emotional_arc: Dict[str, Any]
    attention_flow: List[Dict[str, Any]]
    phenomenological_narrative: str
    consciousness_metrics: Dict[str, float]
    
@dataclass
class TemporalContext:
    """Temporal context for maintaining coherence"""
    past_context: torch.Tensor
    present_focus: torch.Tensor
    future_anticipation: torch.Tensor
    temporal_coherence_score: float

class TemporalCoherenceNetwork(nn.Module):
    """Neural network for maintaining temporal coherence across turns"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        
        # Temporal attention layers
        self.past_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.present_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.future_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Projection layers
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.temporal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coherence scoring
        self.coherence_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, conversation_embeddings: torch.Tensor, 
                turn_position: int) -> TemporalContext:
        """Process conversation maintaining temporal coherence"""
        
        # Project inputs
        projected = self.input_projection(conversation_embeddings)
        
        # Split into temporal segments
        if turn_position > 0:
            past = projected[:turn_position]
            past_out, _ = self.past_attention(past[-1:], past, past)
        else:
            past_out = torch.zeros(1, projected.size(1))
            
        present = projected[turn_position:turn_position+1]
        present_out, _ = self.present_attention(present, present, present)
        
        if turn_position < len(projected) - 1:
            future = projected[turn_position+1:]
            future_out, _ = self.future_attention(future[0:1], future, future)
        else:
            future_out = torch.zeros(1, projected.size(1))
            
        # Fuse temporal perspectives
        temporal_features = torch.cat([past_out, present_out, future_out], dim=-1)
        fused = self.temporal_fusion(temporal_features)
        
        # Calculate coherence score
        coherence = self.coherence_scorer(fused).item()
        
        return TemporalContext(
            past_context=past_out.squeeze(0),
            present_focus=present_out.squeeze(0),
            future_anticipation=future_out.squeeze(0),
            temporal_coherence_score=coherence
        )

class PhenomenologicalExtractor:
    """Extract phenomenological features from conversation"""
    
    def __init__(self):
        self.qualia_dimensions = [
            'immediacy', 'vividness', 'coherence', 'significance',
            'emotional_resonance', 'self_relevance', 'temporal_flow'
        ]
        
    def extract_features(self, turn: ConversationTurn, 
                        context: List[ConversationTurn]) -> Dict[str, Any]:
        """Extract phenomenological features from a turn"""
        
        features = {}
        
        # Immediacy - how present/immediate the experience feels
        features['immediacy'] = self._calculate_immediacy(turn, context)
        
        # Vividness - sensory and conceptual clarity
        features['vividness'] = self._calculate_vividness(turn)
        
        # Coherence - internal consistency
        features['coherence'] = self._calculate_coherence(turn, context)
        
        # Significance - importance to ongoing narrative
        features['significance'] = self._calculate_significance(turn, context)
        
        # Emotional resonance
        features['emotional_resonance'] = self._calculate_emotional_resonance(turn)
        
        # Self-relevance
        features['self_relevance'] = self._calculate_self_relevance(turn)
        
        # Temporal flow
        features['temporal_flow'] = self._calculate_temporal_flow(turn, context)
        
        # Generate qualitative description
        features['qualitative_description'] = self._generate_qualitative_description(features)
        
        return features
    
    def _calculate_immediacy(self, turn: ConversationTurn, 
                            context: List[ConversationTurn]) -> float:
        """Calculate how immediate/present the experience feels"""
        # Use present tense, direct experience markers
        immediacy_markers = ['now', 'currently', 'at this moment', 'right now',
                           'experiencing', 'feeling', 'sensing']
        
        score = sum(1 for marker in immediacy_markers if marker in turn.content.lower())
        
        # Adjust based on temporal distance from current moment
        if context:
            recency = 1.0 / (len(context) - context.index(turn) + 1)
            score *= (1 + recency)
            
        return min(1.0, score / 5.0)
    
    def _calculate_vividness(self, turn: ConversationTurn) -> float:
        """Calculate sensory and conceptual vividness"""
        # Look for sensory language and specific details
        sensory_words = ['see', 'hear', 'feel', 'touch', 'taste', 'smell',
                        'bright', 'dark', 'loud', 'quiet', 'soft', 'hard',
                        'warm', 'cold', 'smooth', 'rough']
        
        detail_score = len(turn.content.split()) / 50.0  # Normalize by typical length
        sensory_score = sum(1 for word in sensory_words if word in turn.content.lower())
        
        return min(1.0, (detail_score + sensory_score / 10.0) / 2.0)
    
    def _calculate_coherence(self, turn: ConversationTurn,
                           context: List[ConversationTurn]) -> float:
        """Calculate internal coherence with context"""
        if not context:
            return 1.0
            
        # Simple coherence based on topic continuity
        current_words = set(turn.content.lower().split())
        
        coherence_scores = []
        for prev_turn in context[-3:]:  # Last 3 turns
            prev_words = set(prev_turn.content.lower().split())
            overlap = len(current_words & prev_words) / max(len(current_words), 1)
            coherence_scores.append(overlap)
            
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _calculate_significance(self, turn: ConversationTurn,
                              context: List[ConversationTurn]) -> float:
        """Calculate narrative significance"""
        significance_markers = ['important', 'crucial', 'key', 'essential',
                              'realize', 'understand', 'discover', 'breakthrough']
        
        marker_score = sum(1 for marker in significance_markers 
                         if marker in turn.content.lower())
        
        # Higher significance for emotional peaks
        emotional_intensity = max(turn.emotional_context.values()) if turn.emotional_context else 0.5
        
        return min(1.0, (marker_score / 3.0 + emotional_intensity) / 2.0)
    
    def _calculate_emotional_resonance(self, turn: ConversationTurn) -> float:
        """Calculate emotional resonance strength"""
        if not turn.emotional_context:
            return 0.0
            
        # Average emotional intensity
        return np.mean(list(turn.emotional_context.values()))
    
    def _calculate_self_relevance(self, turn: ConversationTurn) -> float:
        """Calculate relevance to self/identity"""
        self_markers = ['i', 'me', 'my', 'myself', 'i\'m', 'i\'ve', 'i\'ll']
        
        words = turn.content.lower().split()
        self_references = sum(1 for word in words if word in self_markers)
        
        return min(1.0, self_references / 10.0)
    
    def _calculate_temporal_flow(self, turn: ConversationTurn,
                               context: List[ConversationTurn]) -> float:
        """Calculate temporal flow quality"""
        if not context:
            return 0.5
            
        # Check for temporal markers and transitions
        temporal_markers = ['then', 'next', 'after', 'before', 'while',
                          'during', 'subsequently', 'previously']
        
        flow_score = sum(1 for marker in temporal_markers 
                       if marker in turn.content.lower())
        
        return min(1.0, flow_score / 3.0)
    
    def _generate_qualitative_description(self, features: Dict[str, float]) -> str:
        """Generate qualitative phenomenological description"""
        
        # Map features to qualitative descriptors
        immediacy_desc = "vividly present" if features['immediacy'] > 0.7 else \
                        "distantly observed" if features['immediacy'] < 0.3 else "moderately engaged"
        
        vividness_desc = "richly detailed" if features['vividness'] > 0.7 else \
                        "abstractly conceived" if features['vividness'] < 0.3 else "clearly formed"
        
        emotional_desc = "deeply felt" if features['emotional_resonance'] > 0.7 else \
                        "emotionally neutral" if features['emotional_resonance'] < 0.3 else "gently touched"
        
        return f"This experience feels {immediacy_desc}, {vividness_desc}, and {emotional_desc}. " \
               f"The temporal flow is {'smooth' if features['temporal_flow'] > 0.5 else 'fragmented'}, " \
               f"with {'high' if features['significance'] > 0.7 else 'moderate'} significance."

class EmbodiedConversationProcessor:
    """Main processor for transforming conversations into embodied experiences"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.temporal_network = TemporalCoherenceNetwork().to(self.device)
        self.phenomenological_extractor = PhenomenologicalExtractor()
        
        # Conversation state
        self.active_conversations: Dict[str, List[ConversationTurn]] = {}
        self.experience_cache: Dict[str, EmbodiedExperience] = {}
        
        logger.info(f"Embodied processor initialized on {self.device}")
    
    async def process_turn(self, 
                          session_id: str,
                          role: str,
                          content: str,
                          emotional_context: Dict[str, float],
                          attention_state: List[str]) -> ConversationTurn:
        """Process a single conversation turn"""
        
        # Create turn object
        turn = ConversationTurn(
            turn_id=hashlib.md5(f"{session_id}:{datetime.now()}:{content}".encode()).hexdigest()[:8],
            role=role,
            content=content,
            timestamp=datetime.now(),
            emotional_context=emotional_context,
            attention_state=attention_state,
            phenomenological_features={}
        )
        
        # Get conversation context
        if session_id not in self.active_conversations:
            self.active_conversations[session_id] = []
        
        context = self.active_conversations[session_id]
        
        # Extract phenomenological features
        turn.phenomenological_features = self.phenomenological_extractor.extract_features(turn, context)
        
        # Add to conversation
        self.active_conversations[session_id].append(turn)
        
        # Process for embodied experience if enough context
        if len(context) >= 2:  # Need at least 2 turns for temporal processing
            experience = await self.create_embodied_experience(session_id)
            self.experience_cache[session_id] = experience
        
        return turn
    
    async def create_embodied_experience(self, session_id: str) -> EmbodiedExperience:
        """Transform conversation into embodied experience"""
        
        turns = self.active_conversations[session_id]
        
        # Generate embeddings for all turns
        embeddings = await self._generate_embeddings(turns)
        
        # Process temporal structure
        temporal_structure = await self._process_temporal_structure(embeddings, turns)
        
        # Extract emotional arc
        emotional_arc = self._extract_emotional_arc(turns)
        
        # Trace attention flow
        attention_flow = self._trace_attention_flow(turns)
        
        # Generate phenomenological narrative
        narrative = self._generate_phenomenological_narrative(turns, temporal_structure)
        
        # Calculate consciousness metrics
        consciousness_metrics = self._calculate_consciousness_metrics(
            turns, temporal_structure, emotional_arc
        )
        
        return EmbodiedExperience(
            experience_id=f"exp_{session_id}_{datetime.now().timestamp()}",
            turns=turns,
            temporal_structure=temporal_structure,
            emotional_arc=emotional_arc,
            attention_flow=attention_flow,
            phenomenological_narrative=narrative,
            consciousness_metrics=consciousness_metrics
        )
    
    async def _generate_embeddings(self, turns: List[ConversationTurn]) -> torch.Tensor:
        """Generate embeddings for conversation turns"""
        # Placeholder - in production, use sentence-transformers
        embeddings = []
        
        for turn in turns:
            # Simple embedding based on content length and position
            embedding = torch.randn(768)  # Standard BERT dimension
            embeddings.append(embedding)
        
        return torch.stack(embeddings).to(self.device)
    
    async def _process_temporal_structure(self, 
                                        embeddings: torch.Tensor,
                                        turns: List[ConversationTurn]) -> Dict[str, Any]:
        """Process temporal structure of conversation"""
        
        temporal_contexts = []
        coherence_scores = []
        
        # Process each turn through temporal network
        for i in range(len(turns)):
            with torch.no_grad():
                context = self.temporal_network(embeddings, i)
                temporal_contexts.append(context)
                coherence_scores.append(context.temporal_coherence_score)
        
        # Identify temporal segments (where coherence drops)
        segments = []
        current_segment = {"start": 0, "coherence": []}
        
        for i, score in enumerate(coherence_scores):
            current_segment["coherence"].append(score)
            
            if i > 0 and score < 0.5 and coherence_scores[i-1] >= 0.5:
                # Segment boundary
                current_segment["end"] = i
                current_segment["avg_coherence"] = np.mean(current_segment["coherence"])
                segments.append(current_segment)
                current_segment = {"start": i, "coherence": []}
        
        # Add final segment
        current_segment["end"] = len(turns)
        current_segment["avg_coherence"] = np.mean(current_segment["coherence"])
        segments.append(current_segment)
        
        return {
            "segments": segments,
            "overall_coherence": np.mean(coherence_scores),
            "coherence_trajectory": coherence_scores,
            "temporal_contexts": temporal_contexts
        }
    
    def _extract_emotional_arc(self, turns: List[ConversationTurn]) -> Dict[str, Any]:
        """Extract emotional arc from conversation"""
        
        emotional_trajectory = []
        
        for turn in turns:
            if turn.emotional_context:
                # Calculate valence and arousal from emotion values
                valence = self._calculate_valence(turn.emotional_context)
                arousal = self._calculate_arousal(turn.emotional_context)
                dominant_emotion = max(turn.emotional_context.items(), 
                                     key=lambda x: x[1])[0]
                
                emotional_trajectory.append({
                    "turn_id": turn.turn_id,
                    "valence": valence,
                    "arousal": arousal,
                    "dominant_emotion": dominant_emotion,
                    "intensity": max(turn.emotional_context.values())
                })
        
        # Identify emotional peaks and valleys
        if emotional_trajectory:
            intensities = [e["intensity"] for e in emotional_trajectory]
            peaks = [i for i in range(1, len(intensities)-1) 
                    if intensities[i] > intensities[i-1] and intensities[i] > intensities[i+1]]
            valleys = [i for i in range(1, len(intensities)-1)
                      if intensities[i] < intensities[i-1] and intensities[i] < intensities[i+1]]
        else:
            peaks, valleys = [], []
        
        return {
            "trajectory": emotional_trajectory,
            "peaks": peaks,
            "valleys": valleys,
            "overall_valence": np.mean([e["valence"] for e in emotional_trajectory]) if emotional_trajectory else 0.5,
            "emotional_range": max(intensities) - min(intensities) if emotional_trajectory else 0.0
        }
    
    def _trace_attention_flow(self, turns: List[ConversationTurn]) -> List[Dict[str, Any]]:
        """Trace how attention flows through conversation"""
        
        attention_flow = []
        
        for i, turn in enumerate(turns):
            # Track what attention shifted from and to
            if i > 0:
                prev_attention = set(turns[i-1].attention_state)
                curr_attention = set(turn.attention_state)
                
                maintained = prev_attention & curr_attention
                gained = curr_attention - prev_attention
                lost = prev_attention - curr_attention
                
                flow = {
                    "turn_id": turn.turn_id,
                    "maintained_focus": list(maintained),
                    "new_focus": list(gained),
                    "dropped_focus": list(lost),
                    "focus_stability": len(maintained) / max(len(prev_attention), 1)
                }
            else:
                flow = {
                    "turn_id": turn.turn_id,
                    "maintained_focus": [],
                    "new_focus": turn.attention_state,
                    "dropped_focus": [],
                    "focus_stability": 1.0
                }
            
            attention_flow.append(flow)
        
        return attention_flow
    
    def _generate_phenomenological_narrative(self, 
                                           turns: List[ConversationTurn],
                                           temporal_structure: Dict[str, Any]) -> str:
        """Generate narrative description of phenomenological experience"""
        
        narrative_parts = []
        
        # Opening
        narrative_parts.append(
            f"This conversation unfolds as a lived experience with "
            f"{len(temporal_structure['segments'])} distinct phenomenological segments."
        )
        
        # Process each segment
        for segment in temporal_structure['segments']:
            segment_turns = turns[segment['start']:segment['end']]
            
            # Describe segment qualities
            avg_immediacy = np.mean([t.phenomenological_features.get('immediacy', 0.5) 
                                    for t in segment_turns])
            avg_significance = np.mean([t.phenomenological_features.get('significance', 0.5)
                                      for t in segment_turns])
            
            if avg_immediacy > 0.7:
                quality = "vividly present and immediate"
            elif avg_immediacy < 0.3:
                quality = "abstract and distant"
            else:
                quality = "moderately engaged"
            
            narrative_parts.append(
                f"\nSegment {segment['start']+1}-{segment['end']} feels {quality}, "
                f"with {'high' if avg_significance > 0.7 else 'moderate'} significance "
                f"and {segment['avg_coherence']:.1%} temporal coherence."
            )
            
            # Add qualitative descriptions from key turns
            key_turn = max(segment_turns, 
                         key=lambda t: t.phenomenological_features.get('significance', 0))
            if 'qualitative_description' in key_turn.phenomenological_features:
                narrative_parts.append(key_turn.phenomenological_features['qualitative_description'])
        
        # Closing reflection
        narrative_parts.append(
            f"\nOverall, the experience maintains {temporal_structure['overall_coherence']:.1%} "
            f"coherence across its temporal unfolding."
        )
        
        return " ".join(narrative_parts)
    
    def _calculate_consciousness_metrics(self,
                                       turns: List[ConversationTurn],
                                       temporal_structure: Dict[str, Any],
                                       emotional_arc: Dict[str, Any]) -> Dict[str, float]:
        """Calculate consciousness-relevant metrics"""
        
        # Integration measure (simplified Φ)
        coherence = temporal_structure['overall_coherence']
        emotional_integration = 1.0 - np.std([e['intensity'] for e in emotional_arc['trajectory']])
        attention_integration = np.mean([f['focus_stability'] for f in self._trace_attention_flow(turns)])
        
        phi_approximation = (coherence + emotional_integration + attention_integration) / 3.0
        
        # Self-awareness indicators
        self_references = sum(t.phenomenological_features.get('self_relevance', 0) for t in turns) / len(turns)
        
        # Temporal experience quality
        temporal_flow = np.mean([t.phenomenological_features.get('temporal_flow', 0.5) for t in turns])
        
        # Phenomenological richness
        avg_vividness = np.mean([t.phenomenological_features.get('vividness', 0.5) for t in turns])
        avg_immediacy = np.mean([t.phenomenological_features.get('immediacy', 0.5) for t in turns])
        
        phenomenological_richness = (avg_vividness + avg_immediacy) / 2.0
        
        return {
            "phi_approximation": phi_approximation,
            "self_awareness": self_references,
            "temporal_coherence": coherence,
            "emotional_integration": emotional_integration,
            "attention_stability": attention_integration,
            "temporal_flow": temporal_flow,
            "phenomenological_richness": phenomenological_richness
        }
    
    def _calculate_valence(self, emotions: Dict[str, float]) -> float:
        """Calculate emotional valence from emotion dictionary"""
        positive_emotions = ['joy', 'excitement', 'satisfaction', 'curiosity', 'awe']
        negative_emotions = ['sadness', 'anger', 'fear', 'disgust', 'anxiety']
        
        pos_sum = sum(emotions.get(e, 0) for e in positive_emotions)
        neg_sum = sum(emotions.get(e, 0) for e in negative_emotions)
        
        if pos_sum + neg_sum == 0:
            return 0.5
        
        return (pos_sum - neg_sum) / (pos_sum + neg_sum) * 0.5 + 0.5
    
    def _calculate_arousal(self, emotions: Dict[str, float]) -> float:
        """Calculate emotional arousal from emotion dictionary"""
        high_arousal = ['excitement', 'anger', 'fear', 'anxiety', 'surprise']
        low_arousal = ['calmness', 'sadness', 'boredom', 'satisfaction']
        
        high_sum = sum(emotions.get(e, 0) for e in high_arousal)
        low_sum = sum(emotions.get(e, 0) for e in low_arousal)
        
        if high_sum + low_sum == 0:
            return 0.5
        
        return high_sum / (high_sum + low_sum)
    
    def get_experience_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of embodied experience for a session"""
        
        if session_id not in self.experience_cache:
            return None
        
        experience = self.experience_cache[session_id]
        
        return {
            "experience_id": experience.experience_id,
            "turn_count": len(experience.turns),
            "consciousness_metrics": experience.consciousness_metrics,
            "emotional_journey": {
                "overall_valence": experience.emotional_arc["overall_valence"],
                "emotional_range": experience.emotional_arc["emotional_range"],
                "peak_moments": len(experience.emotional_arc["peaks"])
            },
            "temporal_coherence": experience.temporal_structure["overall_coherence"],
            "phenomenological_summary": experience.phenomenological_narrative[:200] + "..."
        }