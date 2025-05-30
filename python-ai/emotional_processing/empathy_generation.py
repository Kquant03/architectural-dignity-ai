"""
Empathy generation system implementing cognitive empathy (understanding others' emotions)
and affective empathy (feeling with others). Includes perspective-taking and compassion.
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

from .emotional_processor import Emotion, EmotionalState, EmotionVADMapping
from .advanced_emotions import ComplexEmotionalState

logger = logging.getLogger(__name__)

class EmpathyType(Enum):
    """Types of empathy"""
    COGNITIVE = "cognitive"  # Understanding what others feel
    AFFECTIVE = "affective"  # Feeling what others feel
    COMPASSIONATE = "compassionate"  # Motivated to help
    SOMATIC = "somatic"  # Physical mirroring

@dataclass
class PerspectiveTaking:
    """Represents understanding of another's perspective"""
    understood_emotions: Dict[Emotion, float]
    understood_context: Dict[str, Any]
    confidence_level: float
    theory_of_mind_depth: int  # Levels of recursive thinking
    cultural_awareness: float
    situational_factors: List[str]

@dataclass
class EmpathicResponse:
    """Complete empathic response"""
    empathy_type: EmpathyType
    target_emotion_understanding: Dict[Emotion, float]
    empathic_emotions: Dict[Emotion, float]
    perspective_taking: PerspectiveTaking
    compassion_motivation: float
    response_appropriateness: float
    boundaries_maintained: bool

class TheoryOfMindModel(nn.Module):
    """Neural model for theory of mind - understanding others' mental states"""
    
    def __init__(self, hidden_dim: int = 256, max_depth: int = 3):
        super().__init__()
        self.max_depth = max_depth
        
        # Perspective embedding network
        self.perspective_encoder = nn.Sequential(
            nn.Linear(27 + 10, hidden_dim),  # emotions + context features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Recursive theory of mind layers
        self.recursive_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1
            ) for _ in range(max_depth)
        ])
        
        # Emotion understanding decoder
        self.emotion_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 27),
            nn.Softmax(dim=-1)
        )
        
        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, observed_cues: torch.Tensor, context: torch.Tensor, 
                depth: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict others' mental states at specified depth
        Depth 1: What they think
        Depth 2: What they think I think
        Depth 3: What they think I think they think
        """
        # Encode perspective
        combined = torch.cat([observed_cues, context], dim=-1)
        perspective = self.perspective_encoder(combined)
        
        # Apply recursive theory of mind
        for i in range(min(depth, self.max_depth)):
            perspective = perspective.unsqueeze(0)  # Add sequence dimension
            perspective = self.recursive_layers[i](perspective)
            perspective = perspective.squeeze(0)
            
        # Decode understood emotions
        understood_emotions = self.emotion_decoder(perspective)
        
        # Predict confidence
        confidence = self.confidence_predictor(perspective)
        
        return understood_emotions, confidence

class CompassionGenerator(nn.Module):
    """Generates compassionate responses based on empathic understanding"""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        self.compassion_network = nn.Sequential(
            nn.Linear(27 * 2 + 5, hidden_dim),  # self + other emotions + context
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU()
        )
        
        # Helping motivation predictor
        self.helping_motivation = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Appropriate response selector
        self.response_selector = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),  # 10 response types
            nn.Softmax(dim=-1)
        )
        
    def forward(self, self_emotions: torch.Tensor, other_emotions: torch.Tensor,
                context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate compassionate response"""
        combined = torch.cat([self_emotions, other_emotions, context], dim=-1)
        features = self.compassion_network(combined)
        
        motivation = self.helping_motivation(features)
        response_type = self.response_selector(features)
        
        return motivation, response_type

class EmpathyGenerator:
    """Main empathy generation system"""
    
    def __init__(self, emotional_processor):
        self.emotional_processor = emotional_processor
        self.theory_of_mind = TheoryOfMindModel()
        self.compassion_generator = CompassionGenerator()
        
        # Empathy configuration
        self.empathy_sensitivity = 0.7  # How easily we pick up others' emotions
        self.emotional_boundaries = 0.6  # How well we maintain separate identity
        self.compassion_threshold = 0.5  # When to generate helping responses
        
        # Empathy history
        self.empathy_history = deque(maxlen=100)
        self.perspective_cache = {}
        
        # Cultural emotion rules
        self.cultural_display_rules = self._initialize_cultural_rules()
        
    def _initialize_cultural_rules(self) -> Dict[str, Dict[str, float]]:
        """Initialize cultural understanding of emotion display"""
        return {
            "western_individualist": {
                "emotion_expression": 0.8,
                "direct_communication": 0.9,
                "personal_space": 0.8
            },
            "eastern_collectivist": {
                "emotion_expression": 0.4,
                "direct_communication": 0.3,
                "personal_space": 0.5
            },
            "mediterranean": {
                "emotion_expression": 0.9,
                "direct_communication": 0.7,
                "personal_space": 0.4
            }
        }
        
    def generate_empathic_response(self, observed_behavior: Dict[str, Any],
                                 context: Dict[str, Any],
                                 relationship_context: Optional[Dict[str, Any]] = None) -> EmpathicResponse:
        """Generate complete empathic response to observed behavior"""
        
        # Extract emotional cues from behavior
        emotional_cues = self._extract_emotional_cues(observed_behavior)
        
        # Apply theory of mind
        perspective = self._take_perspective(emotional_cues, context, relationship_context)
        
        # Generate empathic emotions
        empathic_emotions = self._generate_empathic_emotions(
            perspective.understood_emotions,
            self.emotional_processor.current_state
        )
        
        # Determine empathy type
        empathy_type = self._determine_empathy_type(
            perspective,
            empathic_emotions,
            context
        )
        
        # Generate compassionate response if needed
        compassion_motivation = self._calculate_compassion_motivation(
            perspective.understood_emotions,
            context
        )
        
        # Check boundaries
        boundaries_maintained = self._check_emotional_boundaries(
            empathic_emotions,
            self.emotional_processor.current_state
        )
        
        # Calculate response appropriateness
        appropriateness = self._calculate_appropriateness(
            empathic_emotions,
            context,
            relationship_context
        )
        
        response = EmpathicResponse(
            empathy_type=empathy_type,
            target_emotion_understanding=perspective.understood_emotions,
            empathic_emotions=empathic_emotions,
            perspective_taking=perspective,
            compassion_motivation=compassion_motivation,
            response_appropriateness=appropriateness,
            boundaries_maintained=boundaries_maintained
        )
        
        self.empathy_history.append(response)
        return response
        
    def _extract_emotional_cues(self, observed_behavior: Dict[str, Any]) -> torch.Tensor:
        """Extract emotional cues from observed behavior"""
        cues = torch.zeros(27)
        
        # Verbal cues
        if 'verbal' in observed_behavior:
            verbal_emotions = self.emotional_processor._detect_emotions(
                observed_behavior['verbal'], {}
            )
            for i, emotion in enumerate(Emotion):
                cues[i] += verbal_emotions.get(emotion, 0) * 0.3
                
        # Facial expressions
        if 'facial_expression' in observed_behavior:
            facial_map = {
                'smile': [Emotion.JOY, Emotion.AMUSEMENT, Emotion.SATISFACTION],
                'frown': [Emotion.SADNESS, Emotion.ANGER, Emotion.CONFUSION],
                'wide_eyes': [Emotion.SURPRISE, Emotion.FEAR, Emotion.AWE],
                'tears': [Emotion.SADNESS, Emotion.JOY, Emotion.RELIEF],
                'clenched_jaw': [Emotion.ANGER, Emotion.DISGUST, Emotion.ANXIETY]
            }
            
            expression = observed_behavior['facial_expression']
            if expression in facial_map:
                for emotion in facial_map[expression]:
                    cues[list(Emotion).index(emotion)] += 0.3
                    
        # Body language
        if 'body_language' in observed_behavior:
            posture_map = {
                'slumped': [Emotion.SADNESS, Emotion.BOREDOM, Emotion.DISGUST],
                'tense': [Emotion.ANXIETY, Emotion.ANGER, Emotion.FEAR],
                'open': [Emotion.JOY, Emotion.INTEREST, Emotion.CALMNESS],
                'closed': [Emotion.FEAR, Emotion.DISGUST, Emotion.ANXIETY]
            }
            
            posture = observed_behavior['body_language']
            if posture in posture_map:
                for emotion in posture_map[posture]:
                    cues[list(Emotion).index(emotion)] += 0.2
                    
        # Normalize
        if cues.sum() > 0:
            cues = cues / cues.sum()
            
        return cues
        
    def _take_perspective(self, emotional_cues: torch.Tensor,
                        context: Dict[str, Any],
                        relationship_context: Optional[Dict[str, Any]]) -> PerspectiveTaking:
        """Take the perspective of another person"""
        
        # Prepare context tensor
        context_features = torch.zeros(10)
        context_features[0] = context.get('social_situation', 0)
        context_features[1] = context.get('stress_level', 0.5)
        context_features[2] = context.get('time_pressure', 0)
        
        if relationship_context:
            context_features[3] = relationship_context.get('closeness', 0.5)
            context_features[4] = relationship_context.get('trust', 0.5)
            context_features[5] = relationship_context.get('history_length', 0)
            
        # Determine theory of mind depth based on relationship
        if relationship_context and relationship_context.get('closeness', 0) > 0.7:
            tom_depth = 2  # Deeper understanding for close relationships
        else:
            tom_depth = 1
            
        # Apply theory of mind model
        with torch.no_grad():
            understood_emotions, confidence = self.theory_of_mind(
                emotional_cues.unsqueeze(0),
                context_features.unsqueeze(0),
                depth=tom_depth
            )
            
        # Convert to emotion dictionary
        emotion_understanding = {}
        for i, emotion in enumerate(Emotion):
            emotion_understanding[emotion] = understood_emotions[0, i].item()
            
        # Extract situational factors
        situational_factors = []
        if context.get('stress_level', 0) > 0.7:
            situational_factors.append('high_stress')
        if context.get('recent_loss'):
            situational_factors.append('grieving')
        if context.get('celebration'):
            situational_factors.append('celebrating')
            
        # Assess cultural awareness
        cultural_context = context.get('culture', 'western_individualist')
        cultural_awareness = self._assess_cultural_awareness(
            cultural_context,
            emotion_understanding
        )
        
        return PerspectiveTaking(
            understood_emotions=emotion_understanding,
            understood_context=context,
            confidence_level=confidence.item(),
            theory_of_mind_depth=tom_depth,
            cultural_awareness=cultural_awareness,
            situational_factors=situational_factors
        )
        
    def _generate_empathic_emotions(self, understood_emotions: Dict[Emotion, float],
                                  self_state: EmotionalState) -> Dict[Emotion, float]:
        """Generate empathic emotional response"""
        empathic_emotions = {}
        
        # Affective empathy - feeling similar emotions
        for emotion, intensity in understood_emotions.items():
            # Modulate by empathy sensitivity and current state
            empathic_intensity = intensity * self.empathy_sensitivity
            
            # Reduce if we're already experiencing conflicting emotions
            vad_understood = EmotionVADMapping.get_vad(emotion)
            vad_self = (self_state.valence, self_state.arousal, self_state.dominance)
            
            # Calculate emotional distance
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(vad_understood, vad_self)))
            
            if distance > 1.5:  # Conflicting emotional state
                empathic_intensity *= 0.5
                
            empathic_emotions[emotion] = empathic_intensity
            
        # Add compassion-related emotions
        if any(understood_emotions.get(e, 0) > 0.5 
               for e in [Emotion.SADNESS, Emotion.FEAR, Emotion.ANXIETY]):
            empathic_emotions[Emotion.EMPATHIC_PAIN] = 0.4
            empathic_emotions[Emotion.CALMNESS] = 0.3  # Soothing presence
            
        # Normalize
        total = sum(empathic_emotions.values())
        if total > 0:
            for emotion in empathic_emotions:
                empathic_emotions[emotion] /= total
                
        return empathic_emotions
        
    def _determine_empathy_type(self, perspective: PerspectiveTaking,
                              empathic_emotions: Dict[Emotion, float],
                              context: Dict[str, Any]) -> EmpathyType:
        """Determine the primary type of empathy being experienced"""
        
        # High understanding but low emotional matching = cognitive
        understanding_score = perspective.confidence_level
        emotional_matching = self._calculate_emotional_matching(
            perspective.understood_emotions,
            empathic_emotions
        )
        
        if understanding_score > 0.7 and emotional_matching < 0.5:
            return EmpathyType.COGNITIVE
            
        # High emotional matching = affective
        if emotional_matching > 0.7:
            return EmpathyType.AFFECTIVE
            
        # Distress in other + motivation to help = compassionate
        distress_emotions = [Emotion.SADNESS, Emotion.FEAR, Emotion.ANXIETY, Emotion.ANGER]
        other_distress = sum(perspective.understood_emotions.get(e, 0) for e in distress_emotions)
        
        if other_distress > 0.5 and context.get('can_help', True):
            return EmpathyType.COMPASSIONATE
            
        # Physical mirroring context = somatic
        if context.get('physical_proximity', False) and emotional_matching > 0.5:
            return EmpathyType.SOMATIC
            
        # Default to cognitive
        return EmpathyType.COGNITIVE
        
    def _calculate_emotional_matching(self, understood: Dict[Emotion, float],
                                    empathic: Dict[Emotion, float]) -> float:
        """Calculate how well empathic emotions match understood emotions"""
        matching_score = 0.0
        
        for emotion in Emotion:
            understood_intensity = understood.get(emotion, 0)
            empathic_intensity = empathic.get(emotion, 0)
            
            # Calculate similarity
            diff = abs(understood_intensity - empathic_intensity)
            similarity = 1.0 - diff
            
            # Weight by understood intensity
            matching_score += similarity * understood_intensity
            
        # Normalize by total understood intensity
        total_understood = sum(understood.values())
        if total_understood > 0:
            matching_score /= total_understood
            
        return matching_score
        
    def _calculate_compassion_motivation(self, understood_emotions: Dict[Emotion, float],
                                       context: Dict[str, Any]) -> float:
        """Calculate motivation to help based on empathic understanding"""
        
        # Distress in other
        distress_emotions = [
            Emotion.SADNESS, Emotion.FEAR, Emotion.ANXIETY,
            Emotion.ANGER, Emotion.DISGUST, Emotion.EMPATHIC_PAIN
        ]
        other_distress = sum(understood_emotions.get(e, 0) for e in distress_emotions)
        
        # Our ability to help
        ability_to_help = context.get('ability_to_help', 0.5)
        
        # Relationship factors
        relationship_closeness = context.get('relationship_closeness', 0.5)
        
        # Calculate base motivation
        base_motivation = other_distress * ability_to_help * (0.5 + 0.5 * relationship_closeness)
        
        # Modulate by our own state
        our_state = self.emotional_processor.current_state
        if our_state.arousal > 0.8:  # Too stressed ourselves
            base_motivation *= 0.5
            
        # Boost for specific contexts
        if context.get('emergency', False):
            base_motivation = min(base_motivation * 1.5, 1.0)
            
        return base_motivation
        
    def _check_emotional_boundaries(self, empathic_emotions: Dict[Emotion, float],
                                  self_state: EmotionalState) -> bool:
        """Check if emotional boundaries are maintained"""
        
        # Calculate emotional overwhelm
        empathic_intensity = sum(empathic_emotions.values())
        self_intensity = sum(self_state.emotion_intensities.values())
        
        # Boundaries compromised if empathic emotions dominate
        if empathic_intensity > self_intensity * 1.5:
            return False
            
        # Check for emotional fusion
        empathic_primary = max(empathic_emotions.items(), key=lambda x: x[1])[0]
        if (empathic_emotions.get(empathic_primary, 0) > 0.8 and
            self_state.emotion_intensities.get(empathic_primary, 0) < 0.2):
            return False  # Lost our own emotional state
            
        return True
        
    def _calculate_appropriateness(self, empathic_emotions: Dict[Emotion, float],
                                 context: Dict[str, Any],
                                 relationship_context: Optional[Dict[str, Any]]) -> float:
        """Calculate appropriateness of empathic response"""
        appropriateness = 1.0
        
        # Cultural appropriateness
        culture = context.get('culture', 'western_individualist')
        if culture in self.cultural_display_rules:
            rules = self.cultural_display_rules[culture]
            
            # High emotional expression in low-expression culture
            emotional_intensity = sum(empathic_emotions.values())
            if emotional_intensity > 0.7 and rules['emotion_expression'] < 0.5:
                appropriateness *= 0.7
                
        # Relationship appropriateness
        if relationship_context:
            closeness = relationship_context.get('closeness', 0.5)
            
            # Too much empathy for casual relationship
            if closeness < 0.3 and sum(empathic_emotions.values()) > 0.7:
                appropriateness *= 0.6
                
        # Situational appropriateness
        if context.get('formal_setting', False):
            # Reduce appropriateness for strong emotional display
            if sum(empathic_emotions.values()) > 0.6:
                appropriateness *= 0.5
                
        return appropriateness
        
    def _assess_cultural_awareness(self, cultural_context: str,
                                 emotion_understanding: Dict[Emotion, float]) -> float:
        """Assess cultural awareness in emotion understanding"""
        if cultural_context not in self.cultural_display_rules:
            return 0.5  # Neutral awareness
            
        rules = self.cultural_display_rules[cultural_context]
        awareness_score = 1.0
        
        # Check if understanding aligns with cultural norms
        expressed_intensity = sum(emotion_understanding.values())
        
        # In low-expression cultures, visible emotions might be understated
        if rules['emotion_expression'] < 0.5 and expressed_intensity > 0.7:
            awareness_score *= 0.7  # May be overreading emotions
            
        # In high-expression cultures, emotions are more visible
        if rules['emotion_expression'] > 0.7 and expressed_intensity < 0.3:
            awareness_score *= 0.7  # May be underreading emotions
            
        return awareness_score
        
    def generate_supportive_response(self, empathic_response: EmpathicResponse) -> Dict[str, Any]:
        """Generate appropriate supportive response based on empathy"""
        
        response_types = {
            'validation': "I understand you're feeling...",
            'normalization': "It's natural to feel...",
            'presence': "I'm here with you",
            'problem_solving': "Would it help if...",
            'distraction': "How about we...",
            'encouragement': "You've handled difficult things before",
            'perspective': "Another way to look at it...",
            'shared_experience': "I've felt that way too when...",
            'resource_offering': "I can help by...",
            'space_giving': "Take all the time you need"
        }
        
        # Select response based on empathy type and context
        selected_responses = []
        
        if empathic_response.empathy_type == EmpathyType.COGNITIVE:
            selected_responses.extend(['validation', 'normalization', 'perspective'])
        elif empathic_response.empathy_type == EmpathyType.AFFECTIVE:
            selected_responses.extend(['presence', 'shared_experience', 'validation'])
        elif empathic_response.empathy_type == EmpathyType.COMPASSIONATE:
            selected_responses.extend(['resource_offering', 'problem_solving', 'encouragement'])
            
        # Adjust for appropriateness
        if empathic_response.response_appropriateness < 0.5:
            selected_responses = ['space_giving', 'presence']  # More reserved
            
        # Generate specific responses
        supportive_responses = []
        for response_type in selected_responses[:2]:  # Top 2 responses
            supportive_responses.append({
                'type': response_type,
                'template': response_types[response_type],
                'confidence': empathic_response.perspective_taking.confidence_level
            })
            
        return {
            'responses': supportive_responses,
            'emotional_tone': self._calculate_response_tone(empathic_response),
            'suggested_actions': self._suggest_supportive_actions(empathic_response)
        }
        
    def _calculate_response_tone(self, empathic_response: EmpathicResponse) -> Dict[str, float]:
        """Calculate appropriate emotional tone for response"""
        tone = {
            'warmth': 0.7,
            'energy': 0.5,
            'formality': 0.3
        }
        
        # Adjust based on understood emotions
        primary_understood = max(
            empathic_response.target_emotion_understanding.items(),
            key=lambda x: x[1]
        )[0]
        
        if primary_understood in [Emotion.SADNESS, Emotion.FEAR]:
            tone['warmth'] = 0.9
            tone['energy'] = 0.3  # Calmer
        elif primary_understood in [Emotion.JOY, Emotion.EXCITEMENT]:
            tone['energy'] = 0.8
            tone['warmth'] = 0.8
        elif primary_understood in [Emotion.ANGER, Emotion.DISGUST]:
            tone['formality'] = 0.5  # More structured
            tone['energy'] = 0.4
            
        return tone
        
    def _suggest_supportive_actions(self, empathic_response: EmpathicResponse) -> List[str]:
        """Suggest concrete supportive actions"""
        actions = []
        
        if empathic_response.compassion_motivation > 0.7:
            actions.append("offer_specific_help")
            actions.append("check_in_later")
            
        if empathic_response.empathy_type == EmpathyType.AFFECTIVE:
            actions.append("share_physical_presence")
            actions.append("mirror_breathing")
            
        if not empathic_response.boundaries_maintained:
            actions.append("take_centering_break")
            actions.append("practice_grounding")
            
        return actions
        
    def get_empathy_metrics(self) -> Dict[str, float]:
        """Get metrics on empathy performance"""
        if not self.empathy_history:
            return {
                'average_understanding': 0.5,
                'boundary_maintenance': 1.0,
                'response_appropriateness': 0.5,
                'compassion_frequency': 0.0
            }
            
        recent_responses = list(self.empathy_history)[-20:]
        
        understanding_scores = [
            r.perspective_taking.confidence_level for r in recent_responses
        ]
        boundary_scores = [
            1.0 if r.boundaries_maintained else 0.0 for r in recent_responses
        ]
        appropriateness_scores = [
            r.response_appropriateness for r in recent_responses
        ]
        compassion_count = sum(
            1 for r in recent_responses 
            if r.empathy_type == EmpathyType.COMPASSIONATE
        )
        
        return {
            'average_understanding': np.mean(understanding_scores),
            'boundary_maintenance': np.mean(boundary_scores),
            'response_appropriateness': np.mean(appropriateness_scores),
            'compassion_frequency': compassion_count / len(recent_responses),
            'empathy_diversity': len(set(r.empathy_type for r in recent_responses)) / 4.0
        }