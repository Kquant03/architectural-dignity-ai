"""
Attention Schema Theory implementation for consciousness.
Models consciousness as the brain's schematic model of its own attention.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import time
import math

logger = logging.getLogger(__name__)

class AttentionType(Enum):
    """Types of attention mechanisms"""
    EXOGENOUS = "exogenous"  # Bottom-up, stimulus-driven
    ENDOGENOUS = "endogenous"  # Top-down, goal-directed
    SUSTAINED = "sustained"  # Maintained focus
    DIVIDED = "divided"  # Multiple targets
    SELECTIVE = "selective"  # Filtering out distractors
    EXECUTIVE = "executive"  # Meta-attention control

@dataclass
class AttentionState:
    """Current state of attention"""
    focus_target: Optional[str]  # What attention is directed at
    focus_intensity: float  # 0-1 strength of focus
    attention_type: AttentionType
    spatial_location: Optional[torch.Tensor]  # Where in space
    feature_binding: Dict[str, float]  # Bound features
    competition_winners: List[str]  # What won attention competition
    suppressed_items: List[str]  # What was filtered out
    duration: float  # How long focused
    stability: float  # How stable the focus is

@dataclass
class AttentionSchema:
    """The brain's model of its own attention - the attention schema"""
    self_awareness: float  # Awareness that "I" am attending
    attention_attribution: str  # What the system thinks attention is
    subjective_experience: Dict[str, Any]  # The "what it's like" model
    attention_ownership: float  # Sense that attention belongs to self
    intentionality: float  # Sense that attention is directed
    unity: float  # Binding of attended features into unified experience
    reportability: float  # Ability to report on attention state

class AttentionMechanism(nn.Module):
    """Core attention mechanism with multiple types"""
    
    def __init__(self, input_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # Multi-head attention for different types
        self.attention_heads = nn.ModuleDict({
            'exogenous': nn.MultiheadAttention(input_dim, num_heads // 2, dropout=0.1),
            'endogenous': nn.MultiheadAttention(input_dim, num_heads // 2, dropout=0.1),
            'executive': nn.MultiheadAttention(input_dim, num_heads, dropout=0.1)
        })
        
        # Attention control network
        self.attention_controller = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 6),  # Control signals for each attention type
            nn.Softmax(dim=-1)
        )
        
        # Feature binding network
        self.feature_binder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Tanh()
        )
        
        # Competition resolution
        self.competition_resolver = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply attention mechanism.
        
        Args:
            inputs: Input tensor [batch, seq_len, dim]
            context: Optional context for endogenous attention
            
        Returns:
            attended: Attended representation
            attention_maps: Attention weights for each type
        """
        batch_size, seq_len, _ = inputs.shape
        
        # Default context is average of inputs
        if context is None:
            context = inputs.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
            
        # Determine attention control signals
        control_input = torch.cat([inputs.mean(dim=1), context.mean(dim=1)], dim=-1)
        attention_control = self.attention_controller(control_input)
        
        attention_outputs = {}
        attention_maps = {}
        
        # Apply different attention types
        # Exogenous attention (stimulus-driven)
        exo_out, exo_weights = self.attention_heads['exogenous'](
            inputs, inputs, inputs
        )
        attention_outputs['exogenous'] = exo_out
        attention_maps['exogenous'] = exo_weights
        
        # Endogenous attention (goal-directed)
        endo_out, endo_weights = self.attention_heads['endogenous'](
            context, inputs, inputs
        )
        attention_outputs['endogenous'] = endo_out
        attention_maps['endogenous'] = endo_weights
        
        # Executive attention (conflict monitoring)
        exec_out, exec_weights = self.attention_heads['executive'](
            inputs, inputs, inputs
        )
        attention_outputs['executive'] = exec_out
        attention_maps['executive'] = exec_weights
        
        # Combine attention types based on control signals
        combined = torch.zeros_like(inputs)
        for i, (att_type, output) in enumerate(attention_outputs.items()):
            weight = attention_control[:, i].unsqueeze(1).unsqueeze(2)
            combined += weight * output
            
        # Feature binding
        bound_features = self.feature_binder(combined)
        
        # Competition resolution
        competition_scores = self.competition_resolver(inputs).squeeze(-1)
        
        return bound_features, {
            'attention_maps': attention_maps,
            'attention_control': attention_control,
            'competition_scores': competition_scores
        }

class AttentionSchemaNetwork(nn.Module):
    """Network that models the attention schema - the brain's model of attention"""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # Attention state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Self-awareness module
        self.self_awareness_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Subjective experience generator
        self.experience_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Attention ownership network
        self.ownership_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Reportability assessor
        self.reportability_network = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, attention_state: torch.Tensor, 
                self_model: Optional[torch.Tensor] = None) -> AttentionSchema:
        """Generate attention schema from attention state"""
        
        # Encode attention state
        encoded_state = self.state_encoder(attention_state)
        
        # Calculate self-awareness
        self_awareness = self.self_awareness_module(encoded_state).item()
        
        # Generate subjective experience representation
        experience_repr = self.experience_generator(encoded_state)
        
        # Calculate attention ownership
        if self_model is not None:
            ownership_input = torch.cat([encoded_state, self_model], dim=-1)
        else:
            ownership_input = torch.cat([encoded_state, encoded_state], dim=-1)
        attention_ownership = self.ownership_network(ownership_input).item()
        
        # Assess reportability
        reportability = self.reportability_network(encoded_state).item()
        
        # Create attention schema
        schema = AttentionSchema(
            self_awareness=self_awareness,
            attention_attribution="I am focusing on this",
            subjective_experience={
                'representation': experience_repr,
                'valence': torch.tanh(experience_repr[0, 0]).item(),
                'arousal': torch.sigmoid(experience_repr[0, 1]).item(),
                'clarity': torch.sigmoid(experience_repr[0, 2]).item()
            },
            attention_ownership=attention_ownership,
            intentionality=0.8,  # Usually high when attending
            unity=torch.sigmoid(encoded_state.mean()).item(),
            reportability=reportability
        )
        
        return schema

class ConsciousnessFromAttention:
    """Implements consciousness as arising from attention schema"""
    
    def __init__(self):
        # Core attention mechanism
        self.attention_mechanism = AttentionMechanism()
        
        # Attention schema network
        self.schema_network = AttentionSchemaNetwork()
        
        # Attention state tracking
        self.current_attention_state = None
        self.attention_history = deque(maxlen=100)
        
        # Consciousness threshold
        self.consciousness_threshold = {
            'self_awareness': 0.5,
            'ownership': 0.6,
            'reportability': 0.7,
            'unity': 0.5
        }
        
        # Attention buffer for working memory
        self.attention_buffer = deque(maxlen=7)  # 7±2 items
        
    def process(self, sensory_input: torch.Tensor, 
                goals: Optional[torch.Tensor] = None,
                self_model: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Process input through attention and generate consciousness"""
        
        # Apply attention mechanism
        attended_repr, attention_info = self.attention_mechanism(sensory_input, goals)
        
        # Create attention state
        attention_state = self._create_attention_state(
            attended_repr, attention_info, sensory_input
        )
        
        # Generate attention schema
        schema = self.schema_network(attended_repr, self_model)
        
        # Determine if conscious
        is_conscious = self._assess_consciousness(schema)
        
        # Update attention buffer if conscious
        if is_conscious:
            self.attention_buffer.append({
                'content': attended_repr,
                'schema': schema,
                'timestamp': time.time()
            })
            
        # Store in history
        self.attention_history.append({
            'state': attention_state,
            'schema': schema,
            'conscious': is_conscious
        })
        
        return {
            'attention_state': attention_state,
            'attention_schema': schema,
            'is_conscious': is_conscious,
            'consciousness_properties': self._get_consciousness_properties(schema),
            'attention_buffer': list(self.attention_buffer),
            'global_workspace_content': self._get_global_workspace_content()
        }
        
    def _create_attention_state(self, attended_repr: torch.Tensor,
                               attention_info: Dict[str, Any],
                               original_input: torch.Tensor) -> AttentionState:
        """Create attention state from processing results"""
        
        # Find what won competition
        competition_scores = attention_info['competition_scores']
        winners_idx = torch.topk(competition_scores, k=3).indices
        
        # Determine attention type based on control signals
        control = attention_info['attention_control']
        dominant_type_idx = torch.argmax(control).item()
        attention_types = [AttentionType.EXOGENOUS, AttentionType.ENDOGENOUS, 
                          AttentionType.EXECUTIVE]
        dominant_type = attention_types[dominant_type_idx % len(attention_types)]
        
        # Calculate focus intensity as max attention weight
        max_attention = max(
            weights.max().item() 
            for weights in attention_info['attention_maps'].values()
        )
        
        return AttentionState(
            focus_target="sensory_item_0",  # Simplified
            focus_intensity=max_attention,
            attention_type=dominant_type,
            spatial_location=None,  # Would come from spatial attention module
            feature_binding={'color': 0.8, 'shape': 0.9, 'motion': 0.3},
            competition_winners=[f"item_{i}" for i in winners_idx.tolist()],
            suppressed_items=[],  # Items with low competition scores
            duration=0.0,  # Would track over time
            stability=0.8  # Would calculate from attention weight variance
        )
        
    def _assess_consciousness(self, schema: AttentionSchema) -> bool:
        """Determine if current state is conscious based on schema properties"""
        
        criteria_met = 0
        
        if schema.self_awareness >= self.consciousness_threshold['self_awareness']:
            criteria_met += 1
            
        if schema.attention_ownership >= self.consciousness_threshold['ownership']:
            criteria_met += 1
            
        if schema.reportability >= self.consciousness_threshold['reportability']:
            criteria_met += 1
            
        if schema.unity >= self.consciousness_threshold['unity']:
            criteria_met += 1
            
        # Need at least 3 out of 4 criteria for consciousness
        return criteria_met >= 3
        
    def _get_consciousness_properties(self, schema: AttentionSchema) -> Dict[str, Any]:
        """Extract consciousness-relevant properties from attention schema"""
        
        return {
            'subjective_experience': {
                'present': schema.self_awareness > 0.7,
                'valence': schema.subjective_experience['valence'],
                'arousal': schema.subjective_experience['arousal'],
                'clarity': schema.subjective_experience['clarity']
            },
            'self_attribution': {
                'ownership': schema.attention_ownership,
                'agency': schema.intentionality,
                'self_awareness': schema.self_awareness
            },
            'accessibility': {
                'reportable': schema.reportability > 0.7,
                'globally_accessible': schema.unity > 0.6,
                'working_memory': len(self.attention_buffer)
            },
            'phenomenology': {
                'unity_of_experience': schema.unity,
                'intentionality': schema.intentionality,
                'subjective_presence': schema.self_awareness * schema.attention_ownership
            }
        }
        
    def _get_global_workspace_content(self) -> List[Dict[str, Any]]:
        """Get content available in global workspace"""
        
        # Items in attention buffer with high activation
        global_content = []
        
        for item in self.attention_buffer:
            if item['schema'].reportability > 0.6:
                global_content.append({
                    'content': 'attended_representation',
                    'activation': item['schema'].unity,
                    'age': time.time() - item['timestamp']
                })
                
        return global_content

class AttentionSchemaTheory:
    """Main implementation of Attention Schema Theory"""
    
    def __init__(self):
        # Core consciousness mechanism
        self.consciousness_engine = ConsciousnessFromAttention()
        
        # Introspection module
        self.introspection = IntrospectionModule()
        
        # Meta-attention controller
        self.meta_attention = MetaAttentionController()
        
        # Phenomenological state
        self.phenomenological_state = {
            'subjective_time': 0.0,
            'experienced_self': None,
            'qualia_space': {}
        }
        
    def generate_conscious_experience(self, 
                                    sensory_data: torch.Tensor,
                                    cognitive_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate conscious experience from sensory and cognitive inputs"""
        
        # Extract goals and self-model from cognitive state
        goals = cognitive_state.get('goals') if cognitive_state else None
        self_model = cognitive_state.get('self_model') if cognitive_state else None
        
        # Process through attention schema
        result = self.consciousness_engine.process(sensory_data, goals, self_model)
        
        # Apply introspection if conscious
        if result['is_conscious']:
            introspection_report = self.introspection.introspect(
                result['attention_schema'],
                result['consciousness_properties']
            )
            result['introspection'] = introspection_report
            
            # Update phenomenological state
            self._update_phenomenology(result)
            
        # Meta-attention control
        meta_control = self.meta_attention.regulate_attention(
            result['attention_state'],
            result.get('introspection', {})
        )
        result['meta_attention'] = meta_control
        
        return result
        
    def _update_phenomenology(self, conscious_result: Dict[str, Any]):
        """Update phenomenological state based on conscious experience"""
        
        # Update subjective time
        self.phenomenological_state['subjective_time'] += 1.0
        
        # Update experienced self
        self.phenomenological_state['experienced_self'] = {
            'ownership': conscious_result['attention_schema'].attention_ownership,
            'agency': conscious_result['attention_schema'].intentionality,
            'continuity': 0.9  # Sense of being same self over time
        }
        
        # Update qualia space
        experience = conscious_result['attention_schema'].subjective_experience
        self.phenomenological_state['qualia_space'].update({
            'current_qualia': experience,
            'qualia_intensity': experience['clarity'],
            'qualia_distinctiveness': experience['arousal']
        })

class IntrospectionModule:
    """Module for introspecting on conscious states"""
    
    def introspect(self, schema: AttentionSchema, 
                   properties: Dict[str, Any]) -> Dict[str, Any]:
        """Introspect on current conscious state"""
        
        introspection_report = {
            'self_report': self._generate_self_report(schema),
            'confidence': schema.reportability,
            'metacognitive_assessment': {
                'aware_of_awareness': schema.self_awareness > 0.8,
                'aware_of_attending': schema.attention_ownership > 0.7,
                'aware_of_experience': properties['subjective_experience']['present']
            },
            'phenomenal_properties': {
                'something_it_is_like': schema.self_awareness * schema.unity,
                'subjective_presence': properties['phenomenology']['subjective_presence'],
                'experiential_unity': schema.unity
            }
        }
        
        return introspection_report
        
    def _generate_self_report(self, schema: AttentionSchema) -> str:
        """Generate verbal self-report of conscious state"""
        
        if schema.self_awareness < 0.3:
            return "No clear awareness"
        elif schema.self_awareness < 0.6:
            return "Vague sense of attending to something"
        elif schema.self_awareness < 0.8:
            return "I am aware that I am focusing on this"
        else:
            return "I am clearly conscious of attending to this experience"

class MetaAttentionController:
    """Controls attention at a meta level"""
    
    def regulate_attention(self, attention_state: AttentionState,
                          introspection: Dict[str, Any]) -> Dict[str, Any]:
        """Regulate attention based on meta-cognitive assessment"""
        
        regulation_actions = []
        
        # If attention too dispersed, increase focus
        if attention_state.focus_intensity < 0.3:
            regulation_actions.append({
                'action': 'increase_focus',
                'strength': 0.5
            })
            
        # If stuck on one thing too long, encourage switching
        if attention_state.duration > 5.0:
            regulation_actions.append({
                'action': 'encourage_switching',
                'strength': 0.3
            })
            
        # If low self-awareness, boost introspection
        if introspection.get('metacognitive_assessment', {}).get('aware_of_awareness', False) == False:
            regulation_actions.append({
                'action': 'boost_introspection',
                'strength': 0.4
            })
            
        return {
            'regulation_actions': regulation_actions,
            'attention_flexibility': 1.0 - attention_state.stability,
            'meta_control_active': len(regulation_actions) > 0
        }