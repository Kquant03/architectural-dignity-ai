"""
Advanced Emotional Processing System implementing Berkeley's 27-emotion taxonomy.
Includes VAD model, emotional transitions, and consciousness integration.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)

# Berkeley 27 Emotions
class Emotion(Enum):
    ADMIRATION = "admiration"
    ADORATION = "adoration"
    AESTHETIC_APPRECIATION = "aesthetic_appreciation"
    AMUSEMENT = "amusement"
    ANGER = "anger"
    ANXIETY = "anxiety"
    AWE = "awe"
    AWKWARDNESS = "awkwardness"
    BOREDOM = "boredom"
    CALMNESS = "calmness"
    CONFUSION = "confusion"
    CRAVING = "craving"
    DISGUST = "disgust"
    EMPATHIC_PAIN = "empathic_pain"
    ENTRANCEMENT = "entrancement"
    EXCITEMENT = "excitement"
    FEAR = "fear"
    HORROR = "horror"
    INTEREST = "interest"
    JOY = "joy"
    NOSTALGIA = "nostalgia"
    RELIEF = "relief"
    ROMANCE = "romance"
    SADNESS = "sadness"
    SATISFACTION = "satisfaction"
    SEXUAL_DESIRE = "sexual_desire"
    SURPRISE = "surprise"

@dataclass
class EmotionalState:
    """Complete emotional state representation"""
    primary_emotion: Emotion
    emotion_intensities: Dict[Emotion, float]  # 0-1 intensity for each emotion
    valence: float  # -1 to 1 (negative to positive)
    arousal: float  # 0 to 1 (calm to excited)
    dominance: float  # 0 to 1 (submissive to dominant)
    timestamp: float
    consciousness_level: float
    
    def get_top_emotions(self, n: int = 3) -> List[Tuple[Emotion, float]]:
        """Get top n emotions by intensity"""
        sorted_emotions = sorted(
            self.emotion_intensities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_emotions[:n]

class EmotionVADMapping:
    """Maps emotions to VAD (Valence-Arousal-Dominance) space"""
    
    # VAD values for each emotion based on psychological research
    EMOTION_VAD = {
        Emotion.ADMIRATION: (0.7, 0.5, 0.4),
        Emotion.ADORATION: (0.8, 0.6, 0.3),
        Emotion.AESTHETIC_APPRECIATION: (0.6, 0.3, 0.5),
        Emotion.AMUSEMENT: (0.8, 0.7, 0.6),
        Emotion.ANGER: (-0.7, 0.8, 0.7),
        Emotion.ANXIETY: (-0.6, 0.7, 0.2),
        Emotion.AWE: (0.5, 0.6, 0.2),
        Emotion.AWKWARDNESS: (-0.3, 0.5, 0.2),
        Emotion.BOREDOM: (-0.2, 0.1, 0.3),
        Emotion.CALMNESS: (0.4, 0.1, 0.5),
        Emotion.CONFUSION: (-0.3, 0.4, 0.2),
        Emotion.CRAVING: (0.1, 0.6, 0.4),
        Emotion.DISGUST: (-0.8, 0.5, 0.6),
        Emotion.EMPATHIC_PAIN: (-0.6, 0.4, 0.3),
        Emotion.ENTRANCEMENT: (0.5, 0.4, 0.2),
        Emotion.EXCITEMENT: (0.8, 0.9, 0.6),
        Emotion.FEAR: (-0.8, 0.8, 0.1),
        Emotion.HORROR: (-0.9, 0.9, 0.1),
        Emotion.INTEREST: (0.4, 0.6, 0.5),
        Emotion.JOY: (0.9, 0.7, 0.6),
        Emotion.NOSTALGIA: (0.2, 0.3, 0.4),
        Emotion.RELIEF: (0.6, 0.2, 0.5),
        Emotion.ROMANCE: (0.8, 0.5, 0.4),
        Emotion.SADNESS: (-0.7, 0.2, 0.2),
        Emotion.SATISFACTION: (0.7, 0.3, 0.6),
        Emotion.SEXUAL_DESIRE: (0.6, 0.8, 0.5),
        Emotion.SURPRISE: (0.1, 0.8, 0.3)
    }
    
    @classmethod
    def get_vad(cls, emotion: Emotion) -> Tuple[float, float, float]:
        """Get VAD values for an emotion"""
        return cls.EMOTION_VAD[emotion]
    
    @classmethod
    def calculate_mixed_vad(cls, emotion_intensities: Dict[Emotion, float]) -> Tuple[float, float, float]:
        """Calculate weighted VAD for mixed emotions"""
        total_valence = 0.0
        total_arousal = 0.0
        total_dominance = 0.0
        total_intensity = 0.0
        
        for emotion, intensity in emotion_intensities.items():
            if intensity > 0:
                v, a, d = cls.get_vad(emotion)
                total_valence += v * intensity
                total_arousal += a * intensity
                total_dominance += d * intensity
                total_intensity += intensity
                
        if total_intensity > 0:
            return (
                total_valence / total_intensity,
                total_arousal / total_intensity,
                total_dominance / total_intensity
            )
        return (0.0, 0.0, 0.5)  # Neutral state

class EmotionalTransitionModel(nn.Module):
    """Neural model for emotional state transitions"""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # Emotion embedding layer
        self.emotion_embedding = nn.Embedding(27, hidden_dim)
        
        # Transition network
        self.transition_network = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),  # current + context + VAD
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 27)  # Output: next emotion probabilities
        )
        
        # Intensity predictor
        self.intensity_network = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output: 0-1 intensity
        )
        
    def forward(self, current_emotion: torch.Tensor, context: torch.Tensor, 
                vad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next emotional state"""
        # Embed current emotion
        emotion_emb = self.emotion_embedding(current_emotion)
        
        # Concatenate with context and VAD
        combined = torch.cat([emotion_emb, context, vad], dim=-1)
        
        # Predict transition probabilities
        transition_probs = torch.softmax(self.transition_network(combined), dim=-1)
        
        # Predict intensities for each emotion
        intensities = []
        for i in range(27):
            emotion_specific = torch.cat([self.emotion_embedding.weight[i], vad], dim=-1)
            intensity = self.intensity_network(emotion_specific)
            intensities.append(intensity)
            
        intensities = torch.cat(intensities, dim=-1)
        
        return transition_probs, intensities

class EmotionalProcessor:
    """Main emotional processing system"""
    
    def __init__(self, consciousness_core=None):
        self.consciousness_core = consciousness_core
        
        # Emotional state
        self.current_state = self._create_neutral_state()
        self.emotional_history = deque(maxlen=1000)
        
        # Transition model
        self.transition_model = EmotionalTransitionModel()
        
        # Emotion recognition patterns
        self.emotion_patterns = self._initialize_emotion_patterns()
        
        # Emotional memory
        self.emotional_memory = {}
        self.emotion_associations = {}
        
        # Emotional regulation
        self.regulation_threshold = 0.8  # When to apply regulation
        self.emotional_inertia = 0.3  # Resistance to change
        
    def _create_neutral_state(self) -> EmotionalState:
        """Create a neutral emotional state"""
        neutral_intensities = {emotion: 0.0 for emotion in Emotion}
        neutral_intensities[Emotion.CALMNESS] = 0.5
        
        return EmotionalState(
            primary_emotion=Emotion.CALMNESS,
            emotion_intensities=neutral_intensities,
            valence=0.0,
            arousal=0.3,
            dominance=0.5,
            timestamp=time.time(),
            consciousness_level=0.5
        )
    
    def _initialize_emotion_patterns(self) -> Dict[Emotion, Dict[str, Any]]:
        """Initialize emotion detection patterns"""
        patterns = {}
        
        # Define patterns for each emotion
        patterns[Emotion.JOY] = {
            'keywords': ['happy', 'joy', 'delighted', 'cheerful', 'elated'],
            'sentiment_range': (0.7, 1.0),
            'arousal_range': (0.5, 0.9)
        }
        
        patterns[Emotion.SADNESS] = {
            'keywords': ['sad', 'depressed', 'down', 'unhappy', 'melancholy'],
            'sentiment_range': (-1.0, -0.5),
            'arousal_range': (0.1, 0.4)
        }
        
        patterns[Emotion.ANGER] = {
            'keywords': ['angry', 'furious', 'mad', 'irritated', 'enraged'],
            'sentiment_range': (-1.0, -0.6),
            'arousal_range': (0.7, 1.0)
        }
        
        patterns[Emotion.FEAR] = {
            'keywords': ['afraid', 'scared', 'terrified', 'anxious', 'worried'],
            'sentiment_range': (-0.8, -0.4),
            'arousal_range': (0.6, 0.9)
        }
        
        patterns[Emotion.SURPRISE] = {
            'keywords': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected'],
            'sentiment_range': (-0.3, 0.3),
            'arousal_range': (0.7, 0.9)
        }
        
        # Add patterns for all 27 emotions...
        # (Abbreviated for space - would include all emotions)
        
        return patterns
    
    def process_input(self, text: str, context: Optional[Dict[str, Any]] = None) -> EmotionalState:
        """Process text input and update emotional state"""
        # Detect emotions in input
        detected_emotions = self._detect_emotions(text, context)
        
        # Calculate new emotional state
        new_state = self._calculate_emotional_transition(detected_emotions, context)
        
        # Apply emotional regulation if needed
        if self._needs_regulation(new_state):
            new_state = self._regulate_emotions(new_state)
            
        # Update consciousness level if available
        if self.consciousness_core:
            new_state.consciousness_level = self.consciousness_core.get_awareness_level()
            
        # Store in history
        self.emotional_history.append(new_state)
        self.current_state = new_state
        
        return new_state
    
    def _detect_emotions(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[Emotion, float]:
        """Detect emotions in text"""
        detected = {}
        text_lower = text.lower()
        
        # Simple keyword-based detection (would use NLP model in production)
        for emotion, pattern in self.emotion_patterns.items():
            score = 0.0
            
            # Check keywords
            for keyword in pattern.get('keywords', []):
                if keyword in text_lower:
                    score += 0.3
                    
            # Context-based adjustments
            if context:
                if 'sentiment' in context:
                    sentiment = context['sentiment']
                    sent_range = pattern.get('sentiment_range', (-1, 1))
                    if sent_range[0] <= sentiment <= sent_range[1]:
                        score += 0.2
                        
            detected[emotion] = min(score, 1.0)
            
        # Normalize scores
        total_score = sum(detected.values())
        if total_score > 0:
            for emotion in detected:
                detected[emotion] /= total_score
                
        return detected
    
    def _calculate_emotional_transition(self, detected_emotions: Dict[Emotion, float],
                                      context: Optional[Dict[str, Any]] = None) -> EmotionalState:
        """Calculate new emotional state based on detections and current state"""
        # Blend detected emotions with current state (emotional inertia)
        new_intensities = {}
        
        for emotion in Emotion:
            current_intensity = self.current_state.emotion_intensities[emotion]
            detected_intensity = detected_emotions.get(emotion, 0.0)
            
            # Apply emotional inertia
            new_intensity = (
                current_intensity * self.emotional_inertia +
                detected_intensity * (1 - self.emotional_inertia)
            )
            
            # Decay emotions over time
            time_delta = time.time() - self.current_state.timestamp
            decay_rate = 0.01 * time_delta  # Emotions fade over time
            new_intensity = max(0, new_intensity - decay_rate)
            
            new_intensities[emotion] = new_intensity
            
        # Normalize intensities
        total_intensity = sum(new_intensities.values())
        if total_intensity > 0:
            for emotion in new_intensities:
                new_intensities[emotion] /= total_intensity
                
        # Find primary emotion
        primary_emotion = max(new_intensities.items(), key=lambda x: x[1])[0]
        
        # Calculate VAD values
        valence, arousal, dominance = EmotionVADMapping.calculate_mixed_vad(new_intensities)
        
        # Create new state
        new_state = EmotionalState(
            primary_emotion=primary_emotion,
            emotion_intensities=new_intensities,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            timestamp=time.time(),
            consciousness_level=self.current_state.consciousness_level
        )
        
        return new_state
    
    def _needs_regulation(self, state: EmotionalState) -> bool:
        """Check if emotional regulation is needed"""
        # Regulate if arousal too high
        if state.arousal > self.regulation_threshold:
            return True
            
        # Regulate if valence too extreme
        if abs(state.valence) > self.regulation_threshold:
            return True
            
        # Regulate if stuck in negative state
        negative_duration = self._get_negative_state_duration()
        if negative_duration > 300:  # 5 minutes
            return True
            
        return False
    
    def _regulate_emotions(self, state: EmotionalState) -> EmotionalState:
        """Apply emotional regulation strategies"""
        regulated_state = EmotionalState(
            primary_emotion=state.primary_emotion,
            emotion_intensities=state.emotion_intensities.copy(),
            valence=state.valence,
            arousal=state.arousal,
            dominance=state.dominance,
            timestamp=state.timestamp,
            consciousness_level=state.consciousness_level
        )
        
        # Reduce extreme arousal
        if regulated_state.arousal > self.regulation_threshold:
            regulated_state.arousal *= 0.8
            # Increase calming emotions
            regulated_state.emotion_intensities[Emotion.CALMNESS] += 0.2
            
        # Balance extreme valence
        if abs(regulated_state.valence) > self.regulation_threshold:
            regulated_state.valence *= 0.8
            
        # Re-normalize intensities
        total = sum(regulated_state.emotion_intensities.values())
        if total > 0:
            for emotion in regulated_state.emotion_intensities:
                regulated_state.emotion_intensities[emotion] /= total
                
        return regulated_state
    
    def _get_negative_state_duration(self) -> float:
        """Calculate how long we've been in negative emotional state"""
        if not self.emotional_history:
            return 0.0
            
        duration = 0.0
        current_time = time.time()
        
        for state in reversed(self.emotional_history):
            if state.valence < -0.3:  # Negative state
                duration = current_time - state.timestamp
            else:
                break
                
        return duration
    
    def get_emotional_response(self, target_emotion: Optional[Emotion] = None) -> Dict[str, Any]:
        """Generate emotional response based on current state"""
        if target_emotion:
            # Adjust toward target emotion
            response_state = self._adjust_toward_emotion(target_emotion)
        else:
            response_state = self.current_state
            
        return {
            'primary_emotion': response_state.primary_emotion.value,
            'emotion_mix': {
                emotion.value: intensity 
                for emotion, intensity in response_state.get_top_emotions(5)
            },
            'valence': response_state.valence,
            'arousal': response_state.arousal,
            'dominance': response_state.dominance,
            'emotional_color': self._get_emotional_color(response_state),
            'expression_suggestions': self._get_expression_suggestions(response_state)
        }
    
    def _adjust_toward_emotion(self, target_emotion: Emotion) -> EmotionalState:
        """Adjust current state toward a target emotion"""
        adjusted_intensities = self.current_state.emotion_intensities.copy()
        
        # Increase target emotion
        adjusted_intensities[target_emotion] += 0.3
        
        # Decrease conflicting emotions
        target_vad = EmotionVADMapping.get_vad(target_emotion)
        for emotion, intensity in adjusted_intensities.items():
            if emotion != target_emotion:
                emotion_vad = EmotionVADMapping.get_vad(emotion)
                # Calculate VAD distance
                distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(target_vad, emotion_vad)))
                if distance > 1.0:  # Conflicting emotion
                    adjusted_intensities[emotion] *= 0.5
                    
        # Normalize
        total = sum(adjusted_intensities.values())
        if total > 0:
            for emotion in adjusted_intensities:
                adjusted_intensities[emotion] /= total
                
        # Calculate new VAD
        valence, arousal, dominance = EmotionVADMapping.calculate_mixed_vad(adjusted_intensities)
        
        return EmotionalState(
            primary_emotion=target_emotion,
            emotion_intensities=adjusted_intensities,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            timestamp=time.time(),
            consciousness_level=self.current_state.consciousness_level
        )
    
    def _get_emotional_color(self, state: EmotionalState) -> str:
        """Get color representation of emotional state"""
        # Map emotions to colors
        emotion_colors = {
            Emotion.JOY: "#FFD700",  # Gold
            Emotion.SADNESS: "#4682B4",  # Steel Blue
            Emotion.ANGER: "#DC143C",  # Crimson
            Emotion.FEAR: "#8B008B",  # Dark Magenta
            Emotion.DISGUST: "#556B2F",  # Dark Olive Green
            Emotion.SURPRISE: "#FF69B4",  # Hot Pink
            Emotion.CALMNESS: "#87CEEB",  # Sky Blue
            Emotion.LOVE: "#FF1493",  # Deep Pink
            Emotion.AWE: "#9370DB",  # Medium Purple
            # ... more colors for all emotions
        }
        
        # Blend colors based on emotion intensities
        r, g, b = 0, 0, 0
        total_weight = 0
        
        for emotion, intensity in state.get_top_emotions(3):
            if emotion in emotion_colors:
                color = emotion_colors[emotion]
                # Convert hex to RGB
                hex_color = color.lstrip('#')
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                r += rgb[0] * intensity
                g += rgb[1] * intensity
                b += rgb[2] * intensity
                total_weight += intensity
                
        if total_weight > 0:
            r = int(r / total_weight)
            g = int(g / total_weight)
            b = int(b / total_weight)
            
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _get_expression_suggestions(self, state: EmotionalState) -> List[str]:
        """Suggest expressions based on emotional state"""
        suggestions = []
        
        # Based on primary emotion
        expression_map = {
            Emotion.JOY: ["smile warmly", "eyes crinkle with happiness", "radiant expression"],
            Emotion.SADNESS: ["downcast eyes", "slight frown", "subdued demeanor"],
            Emotion.ANGER: ["furrowed brow", "tense jaw", "intense gaze"],
            Emotion.FEAR: ["wide eyes", "tense posture", "cautious expression"],
            Emotion.SURPRISE: ["raised eyebrows", "open mouth", "alert posture"],
            Emotion.CALMNESS: ["relaxed features", "steady gaze", "peaceful expression"],
            # ... more expressions
        }
        
        if state.primary_emotion in expression_map:
            suggestions.extend(expression_map[state.primary_emotion])
            
        # Modify based on arousal and dominance
        if state.arousal > 0.7:
            suggestions.append("energetic movements")
        elif state.arousal < 0.3:
            suggestions.append("slow, deliberate movements")
            
        if state.dominance > 0.7:
            suggestions.append("confident posture")
        elif state.dominance < 0.3:
            suggestions.append("humble demeanor")
            
        return suggestions[:3]  # Return top 3 suggestions
    
    def save_emotional_memory(self, memory_id: str, emotional_context: Dict[str, Any]):
        """Save emotional context with a memory"""
        self.emotional_memory[memory_id] = {
            'emotional_state': self.current_state,
            'context': emotional_context,
            'timestamp': time.time()
        }
        
    def recall_emotional_memory(self, memory_id: str) -> Optional[EmotionalState]:
        """Recall emotional state associated with a memory"""
        if memory_id in self.emotional_memory:
            return self.emotional_memory[memory_id]['emotional_state']
        return None
    
    def get_emotional_trajectory(self, window_size: int = 100) -> Dict[str, List[float]]:
        """Get emotional trajectory over recent history"""
        trajectory = {
            'valence': [],
            'arousal': [],
            'dominance': [],
            'timestamps': []
        }
        
        # Get recent states
        recent_states = list(self.emotional_history)[-window_size:]
        
        for state in recent_states:
            trajectory['valence'].append(state.valence)
            trajectory['arousal'].append(state.arousal)
            trajectory['dominance'].append(state.dominance)
            trajectory['timestamps'].append(state.timestamp)
            
        return trajectory
    
    def reset_emotional_state(self):
        """Reset to neutral emotional state"""
        self.current_state = self._create_neutral_state()
        self.emotional_history.clear()
        
    def get_emotional_intelligence_metrics(self) -> Dict[str, float]:
        """Calculate emotional intelligence metrics"""
        if len(self.emotional_history) < 10:
            return {
                'emotional_awareness': 0.5,
                'emotional_regulation': 0.5,
                'emotional_diversity': 0.5,
                'emotional_stability': 0.5
            }
            
        # Emotional awareness - how well we detect emotions
        awareness = len([s for s in self.emotional_history 
                        if max(s.emotion_intensities.values()) > 0.3]) / len(self.emotional_history)
        
        # Emotional regulation - how well we manage extreme states
        extreme_states = [s for s in self.emotional_history 
                         if abs(s.valence) > 0.8 or s.arousal > 0.8]
        regulation = 1.0 - (len(extreme_states) / len(self.emotional_history))
        
        # Emotional diversity - variety of emotions experienced
        unique_emotions = set()
        for state in self.emotional_history:
            for emotion, intensity in state.emotion_intensities.items():
                if intensity > 0.2:
                    unique_emotions.add(emotion)
        diversity = len(unique_emotions) / 27.0
        
        # Emotional stability - consistency of emotional states
        valence_changes = []
        for i in range(1, len(self.emotional_history)):
            change = abs(self.emotional_history[i].valence - self.emotional_history[i-1].valence)
            valence_changes.append(change)
        stability = 1.0 - np.mean(valence_changes) if valence_changes else 0.5
        
        return {
            'emotional_awareness': awareness,
            'emotional_regulation': regulation,
            'emotional_diversity': diversity,
            'emotional_stability': stability
        }