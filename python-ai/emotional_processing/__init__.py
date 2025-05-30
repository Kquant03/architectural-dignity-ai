"""
Emotional Processing Package - Unified interface for all emotional components.
Integrates Berkeley 27-emotion system, empathy, attachment, and emotional memory.
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import deque
import time
import numpy as np

# Import all emotional processing components
from .emotional_processor import (
    EmotionalProcessor,
    Emotion,
    EmotionalState,
    EmotionVADMapping,
    EmotionalTransitionModel
)
from .advanced_emotions import (
    AdvancedEmotionalProcessor,
    ComplexEmotionalState,
    EmotionalPhenomena,
    EmotionalBlend,
    MetaEmotionProcessor,
    EmotionalGranularityAnalyzer,
    EmotionalContagionModel
)
from .empathy_generation import (
    EmpathyGenerator,
    EmpathyType,
    PerspectiveTaking,
    EmpathicResponse,
    TheoryOfMindModel,
    CompassionGenerator
)
from .attachment_framework import (
    AttachmentSystem,
    AttachmentStyle,
    AttachmentProfile,
    RelationshipBond,
    AttachmentBehavior,
    AttachmentDynamicsModel,
    RelationshipMemory
)
from .emotional_memory import (
    EmotionalMemorySystem,
    EmotionalMemory,
    EmotionalContext,
    EmotionalEncodingNetwork,
    MoodCongruentRecall,
    FlashbulbMemoryDetector
)

logger = logging.getLogger(__name__)

__all__ = [
    'UnifiedEmotionalSystem',
    'EmotionalProcessor',
    'Emotion',
    'EmotionalState',
    'ComplexEmotionalState',
    'EmpathyGenerator',
    'AttachmentSystem',
    'EmotionalMemorySystem',
    'EmotionalContext',
    'EmpathicResponse',
    'AttachmentStyle',
    'RelationshipBond'
]

class UnifiedEmotionalSystem:
    """
    Unified interface for all emotional subsystems.
    Orchestrates emotional processing, empathy, attachment, and emotional memory.
    """
    
    def __init__(self, consciousness_core=None, memory_system=None):
        """
        Initialize unified emotional system with all components.
        
        Args:
            consciousness_core: Reference to consciousness core for integration
            memory_system: Reference to main memory system
        """
        self.consciousness_core = consciousness_core
        self.memory_system = memory_system
        
        # Initialize core components
        self.emotional_processor = EmotionalProcessor(consciousness_core)
        self.advanced_processor = AdvancedEmotionalProcessor(self.emotional_processor)
        self.empathy_generator = EmpathyGenerator(self.emotional_processor)
        self.attachment_system = AttachmentSystem(self.emotional_processor)
        self.emotional_memory = EmotionalMemorySystem(memory_system)
        
        # State tracking
        self.current_emotional_context = self._create_default_context()
        self.interaction_history = deque(maxlen=1000)
        self.emotional_timeline = deque(maxlen=5000)
        
        # Integration settings
        self.emotional_depth = 0.7  # How deeply to process emotions
        self.empathy_default = True  # Whether to generate empathy by default
        self.attachment_enabled = True  # Whether to track attachments
        
        # Metrics
        self.total_interactions = 0
        self.emotional_events = []
        
    def _create_default_context(self) -> EmotionalContext:
        """Create default emotional context"""
        return EmotionalContext(
            current_state=self.emotional_processor.current_state,
            recent_emotions=[],
            dominant_mood='neutral',
            emotional_stability=0.8,
            context_tags=[]
        )
        
    def process_interaction(self, 
                          text: str,
                          partner_id: Optional[str] = None,
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an interaction through all emotional systems.
        
        Args:
            text: Input text to process
            partner_id: Optional ID of interaction partner
            context: Optional context dictionary
            
        Returns:
            Comprehensive emotional response
        """
        self.total_interactions += 1
        context = context or {}
        
        # 1. Basic emotional processing
        emotional_state = self.emotional_processor.process_input(text, context)
        
        # 2. Advanced emotional analysis
        complex_state = self.advanced_processor.create_complex_state(
            emotional_state, context
        )
        
        # 3. Update emotional context
        self._update_emotional_context(emotional_state, complex_state)
        
        # 4. Process through attachment system if partner specified
        attachment_response = None
        if partner_id and self.attachment_enabled:
            attachment_response = self.attachment_system.process_interaction(
                partner_id, 
                {'content': text, 'context': context},
                emotional_state
            )
            
        # 5. Generate empathic response if needed
        empathic_response = None
        if self.empathy_default and context.get('observed_behavior'):
            empathic_response = self.empathy_generator.generate_empathic_response(
                context['observed_behavior'],
                context,
                {'partner_id': partner_id} if partner_id else None
            )
            
        # 6. Create emotional memory
        if self.memory_system:
            # Generate embedding (would come from language model in practice)
            embedding = np.random.randn(768)  # Placeholder
            
            emotional_memory = self.emotional_memory.encode_memory(
                text,
                embedding,
                emotional_state,
                context
            )
        else:
            emotional_memory = None
            
        # 7. Check for special emotional phenomena
        special_phenomena = self._detect_special_phenomena(emotional_state, complex_state)
        
        # 8. Generate integrated response
        response = self._generate_integrated_response(
            emotional_state,
            complex_state,
            attachment_response,
            empathic_response,
            emotional_memory,
            special_phenomena
        )
        
        # Store in history
        self.interaction_history.append({
            'timestamp': time.time(),
            'text': text,
            'partner_id': partner_id,
            'emotional_state': emotional_state,
            'response': response
        })
        
        return response
        
    def _update_emotional_context(self, 
                                emotional_state: EmotionalState,
                                complex_state: ComplexEmotionalState):
        """Update the current emotional context"""
        # Update recent emotions
        self.current_emotional_context.recent_emotions.append(emotional_state)
        if len(self.current_emotional_context.recent_emotions) > 10:
            self.current_emotional_context.recent_emotions.pop(0)
            
        # Update current state
        self.current_emotional_context.current_state = emotional_state
        
        # Calculate dominant mood
        recent_valences = [s.valence for s in self.current_emotional_context.recent_emotions]
        avg_valence = np.mean(recent_valences) if recent_valences else 0
        
        if avg_valence > 0.3:
            self.current_emotional_context.dominant_mood = 'positive'
        elif avg_valence < -0.3:
            self.current_emotional_context.dominant_mood = 'negative'
        else:
            self.current_emotional_context.dominant_mood = 'neutral'
            
        # Calculate emotional stability
        if len(recent_valences) > 1:
            valence_variance = np.var(recent_valences)
            self.current_emotional_context.emotional_stability = 1.0 - min(valence_variance, 1.0)
            
        # Update context tags
        self.current_emotional_context.context_tags = []
        if complex_state.emotional_complexity > 0.7:
            self.current_emotional_context.context_tags.append('complex')
        if complex_state.ambivalence_score > 0.5:
            self.current_emotional_context.context_tags.append('ambivalent')
        if complex_state.authenticity_score < 0.5:
            self.current_emotional_context.context_tags.append('guarded')
            
    def _detect_special_phenomena(self,
                                emotional_state: EmotionalState,
                                complex_state: ComplexEmotionalState) -> List[str]:
        """Detect special emotional phenomena"""
        phenomena = []
        
        # Peak experience
        if self.advanced_processor.detect_peak_experience(emotional_state):
            phenomena.append('peak_experience')
            
        # Flow state
        if self.advanced_processor.detect_flow_state(emotional_state):
            phenomena.append('flow_state')
            
        # Emotional blend
        blend = EmotionalBlend.identify_blend(emotional_state)
        if blend:
            phenomena.append(f'blend:{blend}')
            
        # High complexity
        if complex_state.emotional_complexity > 0.8:
            phenomena.append('high_complexity')
            
        # Emotional conflict
        if complex_state.conflicting_pairs:
            phenomena.append('emotional_conflict')
            
        return phenomena
        
    def _generate_integrated_response(self,
                                    emotional_state: EmotionalState,
                                    complex_state: ComplexEmotionalState,
                                    attachment_response: Optional[Dict[str, Any]],
                                    empathic_response: Optional[EmpathicResponse],
                                    emotional_memory: Optional[EmotionalMemory],
                                    special_phenomena: List[str]) -> Dict[str, Any]:
        """Generate integrated emotional response"""
        
        # Get emotional expression
        emotional_expression = self.emotional_processor.get_emotional_response()
        
        # Get complex emotional insights
        complex_insights = {
            'surface_vs_core': self._compare_surface_core(complex_state),
            'emotional_authenticity': complex_state.authenticity_score,
            'emotional_complexity': complex_state.emotional_complexity,
            'ambivalence': complex_state.ambivalence_score
        }
        
        # Compile response
        response = {
            'emotional_state': {
                'primary': emotional_state.primary_emotion.value,
                'all_emotions': {
                    e.value: i for e, i in emotional_state.emotion_intensities.items()
                    if i > 0.1
                },
                'valence': emotional_state.valence,
                'arousal': emotional_state.arousal,
                'dominance': emotional_state.dominance
            },
            'expression': emotional_expression,
            'complex_state': complex_insights,
            'special_phenomena': special_phenomena,
            'consciousness_level': emotional_state.consciousness_level
        }
        
        # Add attachment insights if available
        if attachment_response:
            response['attachment'] = {
                'behaviors': [b.behavior_type for b in attachment_response['triggered_behaviors']],
                'relationship_phase': attachment_response['relationship_update']['phase'],
                'attachment_security': attachment_response['attachment_response']['attachment_state']
            }
            
        # Add empathy insights if available
        if empathic_response:
            response['empathy'] = {
                'type': empathic_response.empathy_type.value,
                'understanding_confidence': empathic_response.perspective_taking.confidence_level,
                'compassion_level': empathic_response.compassion_motivation,
                'boundaries_ok': empathic_response.boundaries_maintained
            }
            
        # Add memory insights if available
        if emotional_memory:
            response['memory'] = {
                'significance': emotional_memory.significance_score,
                'will_be_memorable': emotional_memory.significance_score > 0.7,
                'emotional_peaks': [
                    (e.value, i) for e, i in emotional_memory.emotional_peaks
                ]
            }
            
        return response
        
    def _compare_surface_core(self, complex_state: ComplexEmotionalState) -> Dict[str, Any]:
        """Compare surface and core emotions"""
        surface_primary = max(complex_state.surface_emotions.items(), 
                            key=lambda x: x[1])[0] if complex_state.surface_emotions else None
        core_primary = max(complex_state.core_emotions.items(),
                         key=lambda x: x[1])[0] if complex_state.core_emotions else None
        
        return {
            'aligned': surface_primary == core_primary,
            'surface_primary': surface_primary.value if surface_primary else None,
            'core_primary': core_primary.value if core_primary else None,
            'suppressed_emotions': [
                e.value for e, i in complex_state.suppressed_emotions.items() if i > 0.3
            ]
        }
        
    def recall_emotional_memories(self, 
                                query: Optional[str] = None,
                                emotion_filter: Optional[Emotion] = None,
                                mood_congruent: bool = True,
                                limit: int = 10) -> List[EmotionalMemory]:
        """
        Recall emotional memories based on various criteria.
        
        Args:
            query: Optional text query
            emotion_filter: Optional specific emotion to filter by
            mood_congruent: Whether to use mood-congruent recall
            limit: Maximum number of memories to return
        """
        if emotion_filter:
            return self.emotional_memory.recall_by_emotion(emotion_filter, limit)
        elif mood_congruent:
            return self.emotional_memory.recall_mood_congruent(
                self.current_emotional_context, limit
            )
        else:
            # Fallback to recency
            return list(self.emotional_memory.emotional_memories.values())[-limit:]
            
    def get_relationship_summary(self, partner_id: str) -> Dict[str, Any]:
        """Get comprehensive relationship summary"""
        base_summary = self.attachment_system.get_relationship_summary(partner_id)
        
        if not base_summary['exists']:
            return base_summary
            
        # Add emotional patterns
        relationship_emotions = []
        for interaction in self.interaction_history:
            if interaction.get('partner_id') == partner_id:
                relationship_emotions.append(interaction['emotional_state'])
                
        if relationship_emotions:
            # Calculate emotional patterns
            avg_valence = np.mean([e.valence for e in relationship_emotions])
            dominant_emotions = {}
            for state in relationship_emotions:
                for emotion, intensity in state.emotion_intensities.items():
                    if intensity > 0.3:
                        dominant_emotions[emotion] = dominant_emotions.get(emotion, 0) + intensity
                        
            # Sort dominant emotions
            sorted_emotions = sorted(dominant_emotions.items(), key=lambda x: x[1], reverse=True)[:5]
            
            base_summary['emotional_patterns'] = {
                'average_valence': avg_valence,
                'emotional_tone': 'positive' if avg_valence > 0.2 else 'negative' if avg_valence < -0.2 else 'neutral',
                'dominant_emotions': [(e.value, score) for e, score in sorted_emotions]
            }
            
        return base_summary
        
    def get_emotional_intelligence_report(self) -> Dict[str, Any]:
        """Generate comprehensive emotional intelligence report"""
        # Basic EI metrics
        basic_ei = self.emotional_processor.get_emotional_intelligence_metrics()
        
        # Empathy metrics
        empathy_metrics = self.empathy_generator.get_empathy_metrics()
        
        # Attachment metrics
        attachment_metrics = self.attachment_system.get_attachment_metrics()
        
        # Emotional memory patterns
        memory_stats = self.emotional_memory.get_emotional_memory_stats()
        
        # Advanced emotional patterns
        advanced_patterns = self.advanced_processor.analyze_emotional_patterns()
        
        return {
            'basic_emotional_intelligence': basic_ei,
            'empathy_capabilities': empathy_metrics,
            'attachment_patterns': attachment_metrics,
            'emotional_memory': memory_stats,
            'advanced_patterns': advanced_patterns,
            'total_emotional_experiences': self.total_interactions,
            'emotional_growth_trajectory': self._calculate_growth_trajectory()
        }
        
    def _calculate_growth_trajectory(self) -> str:
        """Calculate overall emotional growth trajectory"""
        if len(self.interaction_history) < 50:
            return 'insufficient_data'
            
        # Compare early and recent emotional patterns
        early_interactions = list(self.interaction_history)[:20]
        recent_interactions = list(self.interaction_history)[-20:]
        
        # Emotional diversity
        early_emotions = set()
        recent_emotions = set()
        
        for interaction in early_interactions:
            state = interaction['emotional_state']
            for emotion, intensity in state.emotion_intensities.items():
                if intensity > 0.2:
                    early_emotions.add(emotion)
                    
        for interaction in recent_interactions:
            state = interaction['emotional_state']
            for emotion, intensity in state.emotion_intensities.items():
                if intensity > 0.2:
                    recent_emotions.add(emotion)
                    
        diversity_growth = len(recent_emotions) > len(early_emotions)
        
        # Emotional stability
        early_valences = [i['emotional_state'].valence for i in early_interactions]
        recent_valences = [i['emotional_state'].valence for i in recent_interactions]
        
        early_stability = 1.0 - np.std(early_valences)
        recent_stability = 1.0 - np.std(recent_valences)
        
        stability_growth = recent_stability > early_stability
        
        if diversity_growth and stability_growth:
            return 'expanding_and_stabilizing'
        elif diversity_growth:
            return 'expanding_range'
        elif stability_growth:
            return 'increasing_stability'
        else:
            return 'maintaining_patterns'
            
    def set_emotional_parameters(self,
                               emotional_depth: Optional[float] = None,
                               empathy_default: Optional[bool] = None,
                               attachment_enabled: Optional[bool] = None):
        """Set emotional processing parameters"""
        if emotional_depth is not None:
            self.emotional_depth = max(0.0, min(1.0, emotional_depth))
        if empathy_default is not None:
            self.empathy_default = empathy_default
        if attachment_enabled is not None:
            self.attachment_enabled = attachment_enabled
            
    def reset_emotional_state(self):
        """Reset to neutral emotional state"""
        self.emotional_processor.reset_emotional_state()
        self.current_emotional_context = self._create_default_context()
        
    def save_emotional_state(self) -> Dict[str, Any]:
        """Save current emotional state for persistence"""
        return {
            'current_state': {
                'primary_emotion': self.emotional_processor.current_state.primary_emotion.value,
                'emotion_intensities': {
                    e.value: i for e, i in 
                    self.emotional_processor.current_state.emotion_intensities.items()
                },
                'valence': self.emotional_processor.current_state.valence,
                'arousal': self.emotional_processor.current_state.arousal,
                'dominance': self.emotional_processor.current_state.dominance
            },
            'attachment_profiles': {
                pid: {
                    'bond_strength': bond.bond_strength,
                    'trust_level': bond.trust_level,
                    'phase': bond.relationship_phase
                }
                for pid, bond in self.attachment_system.relationships.items()
            },
            'emotional_parameters': {
                'emotional_depth': self.emotional_depth,
                'empathy_default': self.empathy_default,
                'attachment_enabled': self.attachment_enabled
            }
        }
        
    def load_emotional_state(self, saved_state: Dict[str, Any]):
        """Load saved emotional state"""
        # This would restore the emotional state from saved data
        # Implementation depends on specific persistence requirements
        pass