"""
Advanced emotional states including mixed emotions, emotional blends, 
and complex emotional phenomena like ambivalence and emotional granularity.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
from collections import defaultdict
import time
import logging

from .emotional_processor import Emotion, EmotionalState, EmotionVADMapping

logger = logging.getLogger(__name__)

@dataclass
class ComplexEmotionalState:
    """Represents complex emotional states with multiple layers"""
    surface_emotions: Dict[Emotion, float]  # What's immediately expressed
    core_emotions: Dict[Emotion, float]     # Deeper, more stable emotions
    suppressed_emotions: Dict[Emotion, float]  # Emotions being held back
    conflicting_pairs: List[Tuple[Emotion, Emotion]]  # Emotional conflicts
    emotional_complexity: float  # 0-1 measure of complexity
    ambivalence_score: float    # Degree of mixed feelings
    authenticity_score: float   # How genuine vs performed
    
class EmotionalPhenomena(Enum):
    """Complex emotional phenomena"""
    AMBIVALENCE = "ambivalence"  # Simultaneous conflicting emotions
    EMOTIONAL_GRANULARITY = "granularity"  # Fine-grained emotion differentiation
    META_EMOTION = "meta_emotion"  # Emotions about emotions
    EMOTIONAL_CONTAGION = "contagion"  # Catching others' emotions
    EMOTIONAL_LABOR = "labor"  # Managing emotions for social reasons
    PEAK_EXPERIENCE = "peak"  # Intense positive emotional states
    FLOW_STATE = "flow"  # Complete absorption and enjoyment
    MIXED_EMOTIONS = "mixed"  # Bittersweet, nostalgic, etc.
    
class EmotionalBlend:
    """Represents blended emotions like bittersweet or nostalgic"""
    
    # Common emotional blends
    BLENDS = {
        "bittersweet": {
            "components": [(Emotion.JOY, 0.5), (Emotion.SADNESS, 0.5)],
            "description": "Happiness tinged with sadness"
        },
        "nostalgic": {
            "components": [(Emotion.JOY, 0.3), (Emotion.SADNESS, 0.3), (Emotion.NOSTALGIA, 0.4)],
            "description": "Wistful affection for the past"
        },
        "guilty_pleasure": {
            "components": [(Emotion.JOY, 0.6), (Emotion.DISGUST, 0.2), (Emotion.EXCITEMENT, 0.2)],
            "description": "Enjoyment despite knowing it's wrong"
        },
        "schadenfreude": {
            "components": [(Emotion.JOY, 0.4), (Emotion.SATISFACTION, 0.3), (Emotion.DISGUST, 0.3)],
            "description": "Pleasure from others' misfortune"
        },
        "sublime": {
            "components": [(Emotion.AWE, 0.5), (Emotion.FEAR, 0.2), (Emotion.ADMIRATION, 0.3)],
            "description": "Overwhelming greatness beyond comprehension"
        },
        "frisson": {
            "components": [(Emotion.AWE, 0.4), (Emotion.EXCITEMENT, 0.3), (Emotion.SURPRISE, 0.3)],
            "description": "Aesthetic chills from beauty"
        },
        "saudade": {
            "components": [(Emotion.NOSTALGIA, 0.4), (Emotion.SADNESS, 0.3), (Emotion.ROMANCE, 0.3)],
            "description": "Deep longing for something absent"
        }
    }
    
    @classmethod
    def identify_blend(cls, emotion_state: EmotionalState) -> Optional[str]:
        """Identify if current state matches a known blend"""
        for blend_name, blend_info in cls.BLENDS.items():
            match_score = 0.0
            for emotion, expected_intensity in blend_info["components"]:
                actual_intensity = emotion_state.emotion_intensities.get(emotion, 0)
                if abs(actual_intensity - expected_intensity) < 0.2:
                    match_score += 1
                    
            if match_score >= len(blend_info["components"]) * 0.7:
                return blend_name
                
        return None

class MetaEmotionProcessor:
    """Processes emotions about emotions"""
    
    def __init__(self):
        self.meta_emotions = {}
        self.emotion_judgments = defaultdict(float)
        
    def process_meta_emotion(self, primary_emotion: Emotion, 
                           reaction_to_emotion: str) -> Dict[str, Any]:
        """Process how we feel about feeling something"""
        
        meta_patterns = {
            "shame_about_anger": {
                "primary": Emotion.ANGER,
                "meta": Emotion.DISGUST,
                "judgment": "negative",
                "description": "Feeling bad about being angry"
            },
            "pride_in_calmness": {
                "primary": Emotion.CALMNESS,
                "meta": Emotion.SATISFACTION,
                "judgment": "positive",
                "description": "Feeling good about staying calm"
            },
            "fear_of_joy": {
                "primary": Emotion.JOY,
                "meta": Emotion.ANXIETY,
                "judgment": "avoidant",
                "description": "Anxiety about feeling happy"
            },
            "guilt_about_relief": {
                "primary": Emotion.RELIEF,
                "meta": Emotion.DISGUST,
                "judgment": "negative",
                "description": "Feeling guilty for feeling relieved"
            }
        }
        
        # Find matching meta-emotion pattern
        for pattern_name, pattern in meta_patterns.items():
            if pattern["primary"] == primary_emotion:
                return {
                    "pattern": pattern_name,
                    "meta_emotion": pattern["meta"],
                    "judgment": pattern["judgment"],
                    "complexity_increase": 0.3
                }
                
        return {
            "pattern": "unrecognized",
            "meta_emotion": None,
            "judgment": "neutral",
            "complexity_increase": 0.1
        }

class EmotionalGranularityAnalyzer:
    """Analyzes emotional granularity - ability to distinguish between similar emotions"""
    
    def __init__(self):
        # Group similar emotions
        self.emotion_families = {
            "positive_high_arousal": [Emotion.JOY, Emotion.EXCITEMENT, Emotion.AMUSEMENT],
            "positive_low_arousal": [Emotion.CALMNESS, Emotion.SATISFACTION, Emotion.RELIEF],
            "negative_high_arousal": [Emotion.ANGER, Emotion.FEAR, Emotion.ANXIETY],
            "negative_low_arousal": [Emotion.SADNESS, Emotion.BOREDOM, Emotion.NOSTALGIA],
            "social_positive": [Emotion.ADMIRATION, Emotion.ADORATION, Emotion.ROMANCE],
            "social_negative": [Emotion.DISGUST, Emotion.AWKWARDNESS, Emotion.EMPATHIC_PAIN],
            "aesthetic": [Emotion.AWE, Emotion.AESTHETIC_APPRECIATION, Emotion.ENTRANCEMENT],
            "surprise_family": [Emotion.SURPRISE, Emotion.CONFUSION, Emotion.INTEREST]
        }
        
    def calculate_granularity(self, emotion_history: List[EmotionalState]) -> Dict[str, float]:
        """Calculate emotional granularity metrics"""
        if len(emotion_history) < 10:
            return {"overall_granularity": 0.5, "family_differentiation": {}}
            
        granularity_scores = {}
        
        for family_name, family_emotions in self.emotion_families.items():
            # Check how well emotions within family are differentiated
            differentiation_events = []
            
            for state in emotion_history:
                family_intensities = [
                    state.emotion_intensities.get(emotion, 0) 
                    for emotion in family_emotions
                ]
                
                # High granularity = different intensities within family
                if max(family_intensities) > 0.1:  # Family is active
                    variance = np.var(family_intensities)
                    differentiation_events.append(variance)
                    
            if differentiation_events:
                granularity_scores[family_name] = np.mean(differentiation_events)
                
        overall_granularity = np.mean(list(granularity_scores.values())) if granularity_scores else 0.5
        
        return {
            "overall_granularity": overall_granularity,
            "family_differentiation": granularity_scores
        }

class EmotionalContagionModel(nn.Module):
    """Models emotional contagion between agents"""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        self.susceptibility_network = nn.Sequential(
            nn.Linear(27 * 2 + 3, hidden_dim),  # self + other emotions + context
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Susceptibility score
        )
        
        self.contagion_transform = nn.Sequential(
            nn.Linear(27 * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 27),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, self_emotions: torch.Tensor, other_emotions: torch.Tensor,
                relationship_strength: float = 0.5) -> torch.Tensor:
        """Calculate emotional contagion effect"""
        # Prepare context tensor
        context = torch.tensor([relationship_strength, 0.5, 0.5])  # relationship, time, situation
        
        # Calculate susceptibility
        combined = torch.cat([self_emotions, other_emotions, context])
        susceptibility = self.susceptibility_network(combined)
        
        # Calculate contagion effect
        emotion_input = torch.cat([self_emotions, other_emotions])
        contagion_effect = self.contagion_transform(emotion_input)
        
        # Apply susceptibility to contagion
        return contagion_effect * susceptibility

class AdvancedEmotionalProcessor:
    """Handles complex emotional phenomena"""
    
    def __init__(self, base_processor):
        self.base_processor = base_processor
        self.meta_processor = MetaEmotionProcessor()
        self.granularity_analyzer = EmotionalGranularityAnalyzer()
        self.contagion_model = EmotionalContagionModel()
        
        # Track complex states
        self.complex_state_history = []
        self.peak_experiences = []
        self.flow_states = []
        
    def create_complex_state(self, base_state: EmotionalState,
                           context: Optional[Dict[str, Any]] = None) -> ComplexEmotionalState:
        """Create complex emotional state from base state"""
        
        # Identify surface vs core emotions
        surface_emotions = self._identify_surface_emotions(base_state, context)
        core_emotions = self._identify_core_emotions(base_state)
        suppressed_emotions = self._identify_suppressed_emotions(base_state, context)
        
        # Find conflicting pairs
        conflicting_pairs = self._find_emotional_conflicts(base_state.emotion_intensities)
        
        # Calculate complexity metrics
        complexity = self._calculate_emotional_complexity(base_state)
        ambivalence = self._calculate_ambivalence(conflicting_pairs, base_state)
        authenticity = self._calculate_authenticity(surface_emotions, core_emotions)
        
        complex_state = ComplexEmotionalState(
            surface_emotions=surface_emotions,
            core_emotions=core_emotions,
            suppressed_emotions=suppressed_emotions,
            conflicting_pairs=conflicting_pairs,
            emotional_complexity=complexity,
            ambivalence_score=ambivalence,
            authenticity_score=authenticity
        )
        
        self.complex_state_history.append(complex_state)
        return complex_state
        
    def _identify_surface_emotions(self, state: EmotionalState,
                                 context: Optional[Dict[str, Any]]) -> Dict[Emotion, float]:
        """Identify emotions being expressed on the surface"""
        surface = {}
        
        # In social contexts, some emotions are more likely to be surface
        if context and context.get("social_situation"):
            # Socially acceptable emotions get boosted
            acceptable_emotions = [
                Emotion.JOY, Emotion.INTEREST, Emotion.CALMNESS,
                Emotion.AMUSEMENT, Emotion.SATISFACTION
            ]
            
            for emotion, intensity in state.emotion_intensities.items():
                if emotion in acceptable_emotions:
                    surface[emotion] = min(intensity * 1.2, 1.0)
                else:
                    surface[emotion] = intensity * 0.8
        else:
            surface = state.emotion_intensities.copy()
            
        return surface
        
    def _identify_core_emotions(self, state: EmotionalState) -> Dict[Emotion, float]:
        """Identify deeper, more stable emotional patterns"""
        # Core emotions are those consistently present across history
        if len(self.base_processor.emotional_history) < 10:
            return state.emotion_intensities.copy()
            
        emotion_persistence = defaultdict(list)
        
        for hist_state in list(self.base_processor.emotional_history)[-20:]:
            for emotion, intensity in hist_state.emotion_intensities.items():
                emotion_persistence[emotion].append(intensity)
                
        core = {}
        for emotion, intensities in emotion_persistence.items():
            # Core emotions have consistent presence
            avg_intensity = np.mean(intensities)
            consistency = 1.0 - np.std(intensities)
            core[emotion] = avg_intensity * consistency
            
        # Normalize
        total = sum(core.values())
        if total > 0:
            for emotion in core:
                core[emotion] /= total
                
        return core
        
    def _identify_suppressed_emotions(self, state: EmotionalState,
                                    context: Optional[Dict[str, Any]]) -> Dict[Emotion, float]:
        """Identify emotions being suppressed"""
        suppressed = {}
        
        # Emotions often suppressed in social contexts
        suppressible = [
            Emotion.ANGER, Emotion.DISGUST, Emotion.FEAR,
            Emotion.SADNESS, Emotion.ANXIETY, Emotion.SEXUAL_DESIRE
        ]
        
        if context and context.get("social_situation"):
            for emotion in suppressible:
                intensity = state.emotion_intensities.get(emotion, 0)
                if intensity > 0.3:  # Strong enough to need suppression
                    suppressed[emotion] = intensity * 0.5
                    
        return suppressed
        
    def _find_emotional_conflicts(self, emotion_intensities: Dict[Emotion, float]) -> List[Tuple[Emotion, Emotion]]:
        """Find conflicting emotion pairs"""
        conflicts = []
        
        # Define conflicting emotion pairs
        conflict_pairs = [
            (Emotion.JOY, Emotion.SADNESS),
            (Emotion.ANGER, Emotion.CALMNESS),
            (Emotion.FEAR, Emotion.EXCITEMENT),
            (Emotion.DISGUST, Emotion.ADMIRATION),
            (Emotion.ANXIETY, Emotion.RELIEF),
            (Emotion.BOREDOM, Emotion.INTEREST)
        ]
        
        for emotion1, emotion2 in conflict_pairs:
            intensity1 = emotion_intensities.get(emotion1, 0)
            intensity2 = emotion_intensities.get(emotion2, 0)
            
            # Both emotions present = conflict
            if intensity1 > 0.2 and intensity2 > 0.2:
                conflicts.append((emotion1, emotion2))
                
        return conflicts
        
    def _calculate_emotional_complexity(self, state: EmotionalState) -> float:
        """Calculate overall emotional complexity"""
        # Factor 1: Number of active emotions
        active_emotions = sum(1 for intensity in state.emotion_intensities.values() if intensity > 0.1)
        diversity_score = min(active_emotions / 10.0, 1.0)
        
        # Factor 2: Entropy of emotion distribution
        intensities = [i for i in state.emotion_intensities.values() if i > 0]
        if intensities:
            entropy = -sum(i * np.log(i + 1e-10) for i in intensities)
            entropy_score = min(entropy / 2.0, 1.0)  # Normalize
        else:
            entropy_score = 0.0
            
        # Factor 3: Presence of mixed valence
        positive_total = sum(state.emotion_intensities.get(e, 0) 
                           for e in [Emotion.JOY, Emotion.EXCITEMENT, Emotion.SATISFACTION])
        negative_total = sum(state.emotion_intensities.get(e, 0)
                           for e in [Emotion.SADNESS, Emotion.ANGER, Emotion.FEAR])
        
        mixed_valence_score = min(positive_total, negative_total) * 2
        
        # Combine factors
        complexity = (diversity_score + entropy_score + mixed_valence_score) / 3.0
        return complexity
        
    def _calculate_ambivalence(self, conflicts: List[Tuple[Emotion, Emotion]],
                             state: EmotionalState) -> float:
        """Calculate ambivalence score"""
        if not conflicts:
            return 0.0
            
        conflict_intensities = []
        for emotion1, emotion2 in conflicts:
            intensity1 = state.emotion_intensities.get(emotion1, 0)
            intensity2 = state.emotion_intensities.get(emotion2, 0)
            # Ambivalence is highest when conflicting emotions are equally strong
            conflict_strength = min(intensity1, intensity2) * 2
            conflict_intensities.append(conflict_strength)
            
        return np.mean(conflict_intensities)
        
    def _calculate_authenticity(self, surface: Dict[Emotion, float],
                              core: Dict[Emotion, float]) -> float:
        """Calculate emotional authenticity"""
        # Compare surface and core emotions
        alignment_scores = []
        
        for emotion in Emotion:
            surface_intensity = surface.get(emotion, 0)
            core_intensity = core.get(emotion, 0)
            
            # High authenticity = surface matches core
            alignment = 1.0 - abs(surface_intensity - core_intensity)
            alignment_scores.append(alignment)
            
        return np.mean(alignment_scores)
        
    def detect_peak_experience(self, state: EmotionalState) -> bool:
        """Detect if current state is a peak experience"""
        # Peak experiences have high positive valence, high arousal, and specific emotions
        peak_emotions = [Emotion.AWE, Emotion.JOY, Emotion.EXCITEMENT, Emotion.ENTRANCEMENT]
        
        peak_intensity = sum(state.emotion_intensities.get(e, 0) for e in peak_emotions)
        
        if (state.valence > 0.8 and state.arousal > 0.7 and peak_intensity > 0.6):
            self.peak_experiences.append({
                "state": state,
                "timestamp": time.time(),
                "intensity": peak_intensity
            })
            return True
            
        return False
        
    def detect_flow_state(self, state: EmotionalState, 
                         activity_engagement: float = 0.0) -> bool:
        """Detect flow state"""
        # Flow = high focus, moderate arousal, positive valence, low self-consciousness
        flow_indicators = {
            "focus": state.emotion_intensities.get(Emotion.INTEREST, 0),
            "enjoyment": state.emotion_intensities.get(Emotion.JOY, 0),
            "calm_intensity": state.emotion_intensities.get(Emotion.CALMNESS, 0) * state.arousal,
            "low_anxiety": 1.0 - state.emotion_intensities.get(Emotion.ANXIETY, 0),
            "engagement": activity_engagement
        }
        
        flow_score = np.mean(list(flow_indicators.values()))
        
        if flow_score > 0.7:
            self.flow_states.append({
                "state": state,
                "timestamp": time.time(),
                "flow_score": flow_score,
                "indicators": flow_indicators
            })
            return True
            
        return False
        
    def process_emotional_contagion(self, self_state: EmotionalState,
                                  other_state: EmotionalState,
                                  relationship_strength: float = 0.5) -> EmotionalState:
        """Process emotional contagion from another agent"""
        # Convert states to tensors
        self_tensor = torch.tensor([
            self_state.emotion_intensities.get(e, 0) for e in Emotion
        ])
        other_tensor = torch.tensor([
            other_state.emotion_intensities.get(e, 0) for e in Emotion
        ])
        
        # Calculate contagion effect
        with torch.no_grad():
            contagion_effect = self.contagion_model(
                self_tensor, other_tensor, relationship_strength
            )
            
        # Apply contagion to current state
        new_intensities = {}
        for i, emotion in enumerate(Emotion):
            original = self_state.emotion_intensities.get(emotion, 0)
            contagion = contagion_effect[i].item()
            # Blend original and contagion
            new_intensities[emotion] = original * 0.7 + contagion * 0.3
            
        # Normalize
        total = sum(new_intensities.values())
        if total > 0:
            for emotion in new_intensities:
                new_intensities[emotion] /= total
                
        # Create new state
        valence, arousal, dominance = EmotionVADMapping.calculate_mixed_vad(new_intensities)
        
        return EmotionalState(
            primary_emotion=max(new_intensities.items(), key=lambda x: x[1])[0],
            emotion_intensities=new_intensities,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            timestamp=time.time(),
            consciousness_level=self_state.consciousness_level
        )
        
    def analyze_emotional_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in complex emotional states"""
        if len(self.complex_state_history) < 10:
            return {
                "average_complexity": 0.5,
                "ambivalence_frequency": 0.0,
                "authenticity_trend": 0.5,
                "peak_experience_count": 0,
                "flow_state_count": 0
            }
            
        complexities = [s.emotional_complexity for s in self.complex_state_history]
        ambivalences = [s.ambivalence_score for s in self.complex_state_history]
        authenticities = [s.authenticity_score for s in self.complex_state_history]
        
        # Check for emotional growth (increasing granularity)
        recent_granularity = self.granularity_analyzer.calculate_granularity(
            list(self.base_processor.emotional_history)[-50:]
        )
        
        return {
            "average_complexity": np.mean(complexities),
            "complexity_trend": np.polyfit(range(len(complexities)), complexities, 1)[0],
            "ambivalence_frequency": np.mean([a > 0.5 for a in ambivalences]),
            "authenticity_trend": np.polyfit(range(len(authenticities)), authenticities, 1)[0],
            "peak_experience_count": len(self.peak_experiences),
            "flow_state_count": len(self.flow_states),
            "emotional_granularity": recent_granularity["overall_granularity"],
            "dominant_blends": self._identify_dominant_blends()
        }
        
    def _identify_dominant_blends(self) -> List[str]:
        """Identify most common emotional blends"""
        blend_counts = defaultdict(int)
        
        for state in list(self.base_processor.emotional_history)[-100:]:
            blend = EmotionalBlend.identify_blend(state)
            if blend:
                blend_counts[blend] += 1
                
        # Sort by frequency
        sorted_blends = sorted(blend_counts.items(), key=lambda x: x[1], reverse=True)
        return [blend for blend, _ in sorted_blends[:3]]