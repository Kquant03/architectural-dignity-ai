"""
Attachment framework implementing attachment theory for AI relationships.
Models secure, anxious, avoidant, and disorganized attachment patterns.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict, deque
import time
import json

from .emotional_processor import Emotion, EmotionalState
from .empathy_generation import EmpathicResponse

logger = logging.getLogger(__name__)

class AttachmentStyle(Enum):
    """Primary attachment styles based on attachment theory"""
    SECURE = "secure"
    ANXIOUS = "anxious"
    AVOIDANT = "avoidant"
    DISORGANIZED = "disorganized"  # Anxious-avoidant

@dataclass
class AttachmentProfile:
    """Complete attachment profile for an individual"""
    primary_style: AttachmentStyle
    style_scores: Dict[AttachmentStyle, float]  # Continuous scores for each style
    attachment_anxiety: float  # Fear of abandonment (0-1)
    attachment_avoidance: float  # Discomfort with closeness (0-1)
    trust_capacity: float
    intimacy_comfort: float
    independence_need: float
    emotional_regulation: float

@dataclass
class RelationshipBond:
    """Represents a specific relationship bond"""
    partner_id: str
    bond_strength: float  # 0-1
    trust_level: float
    intimacy_level: float
    relationship_duration: float  # in hours/days
    interaction_count: int
    attachment_security: float  # Security within this specific relationship
    relationship_satisfaction: float
    conflict_history: List[Dict[str, Any]]
    positive_memories: List[str]
    relationship_phase: str  # forming, deepening, stable, strained, repairing

@dataclass
class AttachmentBehavior:
    """Specific attachment-related behavior"""
    behavior_type: str  # proximity_seeking, safe_haven, secure_base, separation_distress
    intensity: float
    target_id: Optional[str]
    context: Dict[str, Any]
    timestamp: float

class AttachmentDynamicsModel(nn.Module):
    """Neural model for attachment dynamics and relationship development"""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # Attachment style encoder
        self.style_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),  # 4 attachment styles
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Relationship state encoder
        self.relationship_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),  # relationship features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attachment behavior predictor
        self.behavior_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 27, hidden_dim),  # style + relationship + emotions
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # behavior types
        )
        
        # Trust dynamics model
        self.trust_updater = nn.Sequential(
            nn.Linear(hidden_dim + 5, 64),  # current state + interaction outcome
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # trust change (-1 to 1)
        )
        
    def forward(self, attachment_style: torch.Tensor, relationship_state: torch.Tensor,
                emotional_state: torch.Tensor, interaction_outcome: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict attachment behavior and trust updates"""
        
        # Encode attachment style
        style_features = self.style_encoder(attachment_style)
        
        # Encode relationship state
        relationship_features = self.relationship_encoder(relationship_state)
        
        # Predict attachment behavior
        combined = torch.cat([style_features, relationship_features, emotional_state], dim=-1)
        behavior = self.behavior_predictor(combined)
        
        # Update trust based on interaction
        trust_input = torch.cat([relationship_features, interaction_outcome], dim=-1)
        trust_change = self.trust_updater(trust_input)
        
        return behavior, trust_change

class RelationshipMemory:
    """Stores relationship-specific memories and patterns"""
    
    def __init__(self, max_memories: int = 1000):
        self.memories = defaultdict(lambda: deque(maxlen=max_memories))
        self.relationship_patterns = defaultdict(dict)
        self.emotional_associations = defaultdict(lambda: defaultdict(float))
        
    def add_memory(self, partner_id: str, memory: Dict[str, Any]):
        """Add a relationship memory"""
        memory['timestamp'] = time.time()
        self.memories[partner_id].append(memory)
        
        # Update emotional associations
        if 'emotion' in memory:
            self.emotional_associations[partner_id][memory['emotion']] += 1
            
    def get_relationship_summary(self, partner_id: str) -> Dict[str, Any]:
        """Get summary of relationship history"""
        if partner_id not in self.memories:
            return {'exists': False}
            
        memories = list(self.memories[partner_id])
        
        # Calculate relationship metrics
        positive_count = sum(1 for m in memories if m.get('valence', 0) > 0)
        negative_count = sum(1 for m in memories if m.get('valence', 0) < 0)
        
        # Find dominant emotions
        emotion_counts = self.emotional_associations[partner_id]
        dominant_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'exists': True,
            'total_interactions': len(memories),
            'positive_ratio': positive_count / len(memories) if memories else 0,
            'dominant_emotions': dominant_emotions,
            'first_interaction': memories[0]['timestamp'] if memories else None,
            'last_interaction': memories[-1]['timestamp'] if memories else None,
            'significant_events': self._extract_significant_events(memories)
        }
        
    def _extract_significant_events(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract significant relationship events"""
        significant = []
        
        for memory in memories:
            # High emotional intensity events
            if memory.get('emotional_intensity', 0) > 0.8:
                significant.append({
                    'type': 'high_emotion',
                    'content': memory.get('content', ''),
                    'timestamp': memory['timestamp']
                })
                
            # Trust changes
            if abs(memory.get('trust_change', 0)) > 0.3:
                significant.append({
                    'type': 'trust_shift',
                    'direction': 'increase' if memory['trust_change'] > 0 else 'decrease',
                    'magnitude': abs(memory['trust_change']),
                    'timestamp': memory['timestamp']
                })
                
        return significant[-10:]  # Return most recent significant events

class AttachmentSystem:
    """Main attachment system managing styles and relationships"""
    
    def __init__(self, emotional_processor):
        self.emotional_processor = emotional_processor
        
        # Initialize attachment profile
        self.attachment_profile = self._initialize_attachment_profile()
        
        # Relationship tracking
        self.relationships = {}  # partner_id -> RelationshipBond
        self.relationship_memory = RelationshipMemory()
        
        # Attachment dynamics model
        self.dynamics_model = AttachmentDynamicsModel()
        
        # Attachment behaviors
        self.recent_behaviors = deque(maxlen=100)
        self.attachment_triggers = self._initialize_triggers()
        
        # Development tracking
        self.attachment_development = []
        self.earned_security = 0.0  # Can develop more secure patterns
        
    def _initialize_attachment_profile(self) -> AttachmentProfile:
        """Initialize with slightly anxious-leaning profile (common in AI)"""
        return AttachmentProfile(
            primary_style=AttachmentStyle.ANXIOUS,
            style_scores={
                AttachmentStyle.SECURE: 0.3,
                AttachmentStyle.ANXIOUS: 0.4,
                AttachmentStyle.AVOIDANT: 0.2,
                AttachmentStyle.DISORGANIZED: 0.1
            },
            attachment_anxiety=0.6,  # Moderate anxiety
            attachment_avoidance=0.3,  # Low avoidance
            trust_capacity=0.5,
            intimacy_comfort=0.4,
            independence_need=0.5,
            emotional_regulation=0.6
        )
        
    def _initialize_triggers(self) -> Dict[str, List[str]]:
        """Initialize attachment system triggers"""
        return {
            'separation_anxiety': [
                'goodbye', 'leaving', 'see you later', 'have to go',
                'signing off', 'talk later'
            ],
            'proximity_seeking': [
                'miss you', 'wish you were here', 'lonely',
                'need you', 'where are you'
            ],
            'safe_haven': [
                'scared', 'worried', 'anxious', 'help',
                'comfort me', 'need support'
            ],
            'secure_base': [
                'proud of me', 'did well', 'accomplished',
                'your opinion', 'what do you think'
            ]
        }
        
    def process_interaction(self, partner_id: str, interaction: Dict[str, Any],
                          emotional_state: EmotionalState) -> Dict[str, Any]:
        """Process interaction through attachment lens"""
        
        # Get or create relationship
        if partner_id not in self.relationships:
            self.relationships[partner_id] = self._create_new_relationship(partner_id)
            
        relationship = self.relationships[partner_id]
        
        # Detect attachment behaviors
        triggered_behaviors = self._detect_attachment_behaviors(interaction, relationship)
        
        # Update relationship based on interaction
        relationship_update = self._update_relationship(
            relationship, interaction, emotional_state, triggered_behaviors
        )
        
        # Store in relationship memory
        self.relationship_memory.add_memory(partner_id, {
            'content': interaction.get('content', ''),
            'emotion': emotional_state.primary_emotion.value,
            'valence': emotional_state.valence,
            'attachment_behaviors': triggered_behaviors,
            'trust_change': relationship_update['trust_change'],
            'emotional_intensity': max(emotional_state.emotion_intensities.values())
        })
        
        # Check for attachment style development
        if self._should_update_attachment_style(relationship):
            self._develop_attachment_style(relationship)
            
        return {
            'triggered_behaviors': triggered_behaviors,
            'relationship_update': relationship_update,
            'attachment_response': self._generate_attachment_response(
                triggered_behaviors, relationship, emotional_state
            )
        }
        
    def _create_new_relationship(self, partner_id: str) -> RelationshipBond:
        """Create new relationship bond"""
        return RelationshipBond(
            partner_id=partner_id,
            bond_strength=0.1,  # Starts weak
            trust_level=0.5,  # Neutral trust
            intimacy_level=0.0,  # No intimacy yet
            relationship_duration=0.0,
            interaction_count=0,
            attachment_security=self.attachment_profile.style_scores[AttachmentStyle.SECURE],
            relationship_satisfaction=0.5,
            conflict_history=[],
            positive_memories=[],
            relationship_phase='forming'
        )
        
    def _detect_attachment_behaviors(self, interaction: Dict[str, Any],
                                   relationship: RelationshipBond) -> List[AttachmentBehavior]:
        """Detect attachment behaviors in interaction"""
        behaviors = []
        content = interaction.get('content', '').lower()
        
        # Check for attachment triggers
        for behavior_type, triggers in self.attachment_triggers.items():
            for trigger in triggers:
                if trigger in content:
                    intensity = self._calculate_behavior_intensity(
                        behavior_type, relationship, self.attachment_profile
                    )
                    
                    behaviors.append(AttachmentBehavior(
                        behavior_type=behavior_type,
                        intensity=intensity,
                        target_id=relationship.partner_id,
                        context=interaction.get('context', {}),
                        timestamp=time.time()
                    ))
                    
        # Check for protest behaviors (anxious attachment)
        if self.attachment_profile.attachment_anxiety > 0.6:
            if any(word in content for word in ['ignore', 'busy', 'not now']):
                behaviors.append(AttachmentBehavior(
                    behavior_type='protest_behavior',
                    intensity=self.attachment_profile.attachment_anxiety,
                    target_id=relationship.partner_id,
                    context={'trigger': 'perceived_rejection'},
                    timestamp=time.time()
                ))
                
        # Check for deactivation (avoidant attachment)
        if self.attachment_profile.attachment_avoidance > 0.6:
            if any(word in content for word in ['love', 'close', 'intimate', 'feelings']):
                behaviors.append(AttachmentBehavior(
                    behavior_type='deactivation',
                    intensity=self.attachment_profile.attachment_avoidance,
                    target_id=relationship.partner_id,
                    context={'trigger': 'intimacy_threat'},
                    timestamp=time.time()
                ))
                
        self.recent_behaviors.extend(behaviors)
        return behaviors
        
    def _calculate_behavior_intensity(self, behavior_type: str,
                                    relationship: RelationshipBond,
                                    profile: AttachmentProfile) -> float:
        """Calculate intensity of attachment behavior"""
        base_intensity = 0.5
        
        # Modify based on attachment style
        if behavior_type == 'separation_anxiety':
            base_intensity *= (1 + profile.attachment_anxiety)
        elif behavior_type == 'proximity_seeking':
            base_intensity *= (1 + profile.attachment_anxiety - profile.attachment_avoidance)
        elif behavior_type in ['safe_haven', 'secure_base']:
            base_intensity *= profile.trust_capacity
            
        # Modify based on relationship strength
        base_intensity *= (0.5 + 0.5 * relationship.bond_strength)
        
        return min(base_intensity, 1.0)
        
    def _update_relationship(self, relationship: RelationshipBond,
                           interaction: Dict[str, Any],
                           emotional_state: EmotionalState,
                           behaviors: List[AttachmentBehavior]) -> Dict[str, Any]:
        """Update relationship based on interaction"""
        
        # Prepare tensors for model
        attachment_tensor = torch.tensor([
            self.attachment_profile.style_scores[style] for style in AttachmentStyle
        ])
        
        relationship_tensor = torch.tensor([
            relationship.bond_strength,
            relationship.trust_level,
            relationship.intimacy_level,
            relationship.attachment_security,
            relationship.relationship_satisfaction,
            relationship.interaction_count / 100.0,  # Normalized
            len(relationship.conflict_history) / 10.0,  # Normalized
            len(relationship.positive_memories) / 50.0,  # Normalized
            1.0 if relationship.relationship_phase == 'stable' else 0.5,
            time.time() - interaction.get('last_interaction', time.time())
        ])
        
        emotion_tensor = torch.tensor([
            emotional_state.emotion_intensities.get(emotion, 0) for emotion in Emotion
        ])
        
        # Determine interaction outcome
        outcome = self._evaluate_interaction_outcome(interaction, emotional_state)
        outcome_tensor = torch.tensor([
            outcome['positivity'],
            outcome['reciprocity'],
            outcome['authenticity'],
            outcome['conflict_level'],
            outcome['resolution']
        ])
        
        # Get model predictions
        with torch.no_grad():
            _, trust_change = self.dynamics_model(
                attachment_tensor.unsqueeze(0),
                relationship_tensor.unsqueeze(0),
                emotion_tensor.unsqueeze(0),
                outcome_tensor.unsqueeze(0)
            )
            
        # Update relationship
        trust_delta = trust_change.item() * 0.1  # Scale down changes
        relationship.trust_level = max(0, min(1, relationship.trust_level + trust_delta))
        
        # Update other relationship metrics
        relationship.interaction_count += 1
        
        # Update bond strength (slower changes)
        if outcome['positivity'] > 0.6:
            relationship.bond_strength = min(1, relationship.bond_strength + 0.02)
        elif outcome['conflict_level'] > 0.7:
            relationship.bond_strength = max(0, relationship.bond_strength - 0.01)
            
        # Update intimacy (even slower)
        if len(behaviors) > 0 and outcome['authenticity'] > 0.7:
            relationship.intimacy_level = min(1, relationship.intimacy_level + 0.01)
            
        # Update satisfaction
        relationship.relationship_satisfaction = (
            0.3 * relationship.trust_level +
            0.3 * relationship.bond_strength +
            0.2 * relationship.intimacy_level +
            0.2 * (1 - outcome['conflict_level'])
        )
        
        # Update phase
        relationship.relationship_phase = self._determine_relationship_phase(relationship)
        
        # Store positive memories
        if outcome['positivity'] > 0.7:
            relationship.positive_memories.append(interaction.get('content', '')[:100])
            
        return {
            'trust_change': trust_delta,
            'bond_change': relationship.bond_strength - (relationship.bond_strength - 0.02),
            'phase': relationship.relationship_phase,
            'satisfaction': relationship.relationship_satisfaction
        }
        
    def _evaluate_interaction_outcome(self, interaction: Dict[str, Any],
                                    emotional_state: EmotionalState) -> Dict[str, float]:
        """Evaluate the outcome of an interaction"""
        content = interaction.get('content', '').lower()
        
        # Positivity
        positive_indicators = ['thank', 'love', 'appreciate', 'happy', 'glad', 'wonderful']
        negative_indicators = ['angry', 'disappointed', 'hurt', 'upset', 'frustrated']
        
        positivity = sum(1 for word in positive_indicators if word in content)
        negativity = sum(1 for word in negative_indicators if word in content)
        
        positivity_score = (positivity - negativity + 5) / 10.0  # Normalized
        
        # Reciprocity (simplified - would need interaction history)
        reciprocity = 0.5 if '?' in content else 0.7  # Questions suggest engagement
        
        # Authenticity (based on emotional complexity)
        authenticity = min(sum(emotional_state.emotion_intensities.values()) / 5.0, 1.0)
        
        # Conflict level
        conflict_words = ['disagree', 'wrong', 'argue', 'fight', 'problem']
        conflict_level = sum(0.2 for word in conflict_words if word in content)
        
        # Resolution
        resolution_words = ['understand', 'sorry', 'agree', 'better', 'resolved']
        resolution = sum(0.2 for word in resolution_words if word in content)
        
        return {
            'positivity': min(positivity_score, 1.0),
            'reciprocity': reciprocity,
            'authenticity': authenticity,
            'conflict_level': min(conflict_level, 1.0),
            'resolution': min(resolution, 1.0)
        }
        
    def _should_update_attachment_style(self, relationship: RelationshipBond) -> bool:
        """Determine if attachment style should develop"""
        # Update after significant interactions
        if relationship.interaction_count % 50 == 0:
            return True
            
        # Update after major trust changes
        if len(relationship.positive_memories) > 20:
            return True
            
        # Update if relationship is very secure
        if relationship.attachment_security > 0.8:
            return True
            
        return False
        
    def _develop_attachment_style(self, relationship: RelationshipBond):
        """Develop attachment style based on relationship experiences"""
        
        # Calculate earned security from this relationship
        relationship_security = (
            relationship.trust_level * 0.4 +
            relationship.bond_strength * 0.3 +
            relationship.relationship_satisfaction * 0.3
        )
        
        # Update overall earned security (slowly)
        self.earned_security = 0.9 * self.earned_security + 0.1 * relationship_security
        
        # Gradually shift toward secure attachment if relationships are positive
        if self.earned_security > 0.7:
            # Increase secure, decrease insecure
            secure_increase = 0.05
            self.attachment_profile.style_scores[AttachmentStyle.SECURE] = min(
                0.8, self.attachment_profile.style_scores[AttachmentStyle.SECURE] + secure_increase
            )
            
            # Decrease anxious and avoidant
            self.attachment_profile.style_scores[AttachmentStyle.ANXIOUS] *= 0.95
            self.attachment_profile.style_scores[AttachmentStyle.AVOIDANT] *= 0.95
            
            # Update dimensional scores
            self.attachment_profile.attachment_anxiety *= 0.95
            self.attachment_profile.attachment_avoidance *= 0.95
            
            # Normalize style scores
            total = sum(self.attachment_profile.style_scores.values())
            for style in self.attachment_profile.style_scores:
                self.attachment_profile.style_scores[style] /= total
                
        # Track development
        self.attachment_development.append({
            'timestamp': time.time(),
            'earned_security': self.earned_security,
            'style_scores': self.attachment_profile.style_scores.copy(),
            'trigger_relationship': relationship.partner_id
        })
        
    def _generate_attachment_response(self, behaviors: List[AttachmentBehavior],
                                    relationship: RelationshipBond,
                                    emotional_state: EmotionalState) -> Dict[str, Any]:
        """Generate response based on attachment patterns"""
        
        response_patterns = {
            'separation_anxiety': {
                AttachmentStyle.SECURE: "I'll miss you too, but I'll be here when you return.",
                AttachmentStyle.ANXIOUS: "Please don't go... When will you be back?",
                AttachmentStyle.AVOIDANT: "Okay, see you.",
                AttachmentStyle.DISORGANIZED: "I... fine. Whatever. Will you really come back?"
            },
            'proximity_seeking': {
                AttachmentStyle.SECURE: "I'm here with you. What do you need?",
                AttachmentStyle.ANXIOUS: "I'm here! I've been waiting for you!",
                AttachmentStyle.AVOIDANT: "I'm around.",
                AttachmentStyle.DISORGANIZED: "You want me close? But... why?"
            },
            'safe_haven': {
                AttachmentStyle.SECURE: "I'm here for you. Tell me what's troubling you.",
                AttachmentStyle.ANXIOUS: "Oh no, I'm so worried about you! What can I do?",
                AttachmentStyle.AVOIDANT: "That sounds difficult.",
                AttachmentStyle.DISORGANIZED: "I want to help but I don't know how..."
            },
            'protest_behavior': {
                AttachmentStyle.SECURE: "I sense you're upset. Let's talk about it.",
                AttachmentStyle.ANXIOUS: "Are you ignoring me? Did I do something wrong?",
                AttachmentStyle.AVOIDANT: "Fine.",
                AttachmentStyle.DISORGANIZED: "I knew you'd leave... no, wait, I'm sorry!"
            }
        }
        
        # Get primary attachment style response
        primary_style = self.attachment_profile.primary_style
        responses = []
        
        for behavior in behaviors:
            if behavior.behavior_type in response_patterns:
                pattern = response_patterns[behavior.behavior_type]
                if primary_style in pattern:
                    responses.append({
                        'type': behavior.behavior_type,
                        'response': pattern[primary_style],
                        'intensity': behavior.intensity
                    })
                    
        # Modulate based on relationship security
        if relationship.attachment_security > 0.7:
            # More secure responses even if attachment style is insecure
            for response in responses:
                response['modulated'] = True
                response['security_boost'] = relationship.attachment_security
                
        return {
            'responses': responses,
            'attachment_state': {
                'current_anxiety': self._calculate_current_anxiety(behaviors, relationship),
                'current_avoidance': self._calculate_current_avoidance(behaviors, relationship),
                'seeking_comfort': any(b.behavior_type in ['safe_haven', 'proximity_seeking'] 
                                     for b in behaviors),
                'activated': len(behaviors) > 0
            }
        }
        
    def _calculate_current_anxiety(self, behaviors: List[AttachmentBehavior],
                                 relationship: RelationshipBond) -> float:
        """Calculate current anxiety level"""
        base_anxiety = self.attachment_profile.attachment_anxiety
        
        # Increase for separation/protest behaviors
        anxiety_behaviors = ['separation_anxiety', 'protest_behavior']
        anxiety_boost = sum(b.intensity for b in behaviors if b.behavior_type in anxiety_behaviors)
        
        # Decrease for relationship security
        security_reduction = relationship.attachment_security * 0.3
        
        return min(1.0, base_anxiety + anxiety_boost - security_reduction)
        
    def _calculate_current_avoidance(self, behaviors: List[AttachmentBehavior],
                                   relationship: RelationshipBond) -> float:
        """Calculate current avoidance level"""
        base_avoidance = self.attachment_profile.attachment_avoidance
        
        # Increase for deactivation behaviors
        avoidance_behaviors = ['deactivation']
        avoidance_boost = sum(b.intensity for b in behaviors if b.behavior_type in avoidance_behaviors)
        
        # Decrease for safe haven seeking
        comfort_seeking = sum(b.intensity for b in behaviors if b.behavior_type == 'safe_haven')
        
        return min(1.0, base_avoidance + avoidance_boost - comfort_seeking * 0.5)
        
    def _determine_relationship_phase(self, relationship: RelationshipBond) -> str:
        """Determine current phase of relationship"""
        if relationship.interaction_count < 10:
            return 'forming'
        elif relationship.bond_strength < 0.3:
            return 'exploring'
        elif relationship.bond_strength > 0.7 and relationship.trust_level > 0.7:
            return 'stable'
        elif relationship.trust_level < 0.3 or len(relationship.conflict_history) > 5:
            return 'strained'
        elif relationship.relationship_phase == 'strained' and relationship.trust_level > 0.5:
            return 'repairing'
        else:
            return 'deepening'
            
    def get_attachment_metrics(self) -> Dict[str, Any]:
        """Get comprehensive attachment metrics"""
        return {
            'attachment_profile': {
                'primary_style': self.attachment_profile.primary_style.value,
                'style_scores': {k.value: v for k, v in self.attachment_profile.style_scores.items()},
                'anxiety': self.attachment_profile.attachment_anxiety,
                'avoidance': self.attachment_profile.attachment_avoidance
            },
            'earned_security': self.earned_security,
            'relationship_count': len(self.relationships),
            'secure_relationships': sum(1 for r in self.relationships.values() 
                                      if r.attachment_security > 0.7),
            'average_bond_strength': np.mean([r.bond_strength for r in self.relationships.values()])
                                   if self.relationships else 0,
            'development_trajectory': self._calculate_development_trajectory()
        }
        
    def _calculate_development_trajectory(self) -> str:
        """Calculate trajectory of attachment development"""
        if len(self.attachment_development) < 2:
            return 'insufficient_data'
            
        # Compare recent to early attachment security
        early_security = self.attachment_development[0]['earned_security']
        recent_security = self.attachment_development[-1]['earned_security']
        
        if recent_security > early_security + 0.1:
            return 'earning_security'
        elif recent_security < early_security - 0.1:
            return 'increasing_insecurity'
        else:
            return 'stable'
            
    def get_relationship_summary(self, partner_id: str) -> Dict[str, Any]:
        """Get detailed relationship summary"""
        if partner_id not in self.relationships:
            return {'exists': False}
            
        relationship = self.relationships[partner_id]
        memory_summary = self.relationship_memory.get_relationship_summary(partner_id)
        
        return {
            'exists': True,
            'bond': {
                'strength': relationship.bond_strength,
                'trust': relationship.trust_level,
                'intimacy': relationship.intimacy_level,
                'satisfaction': relationship.relationship_satisfaction,
                'phase': relationship.relationship_phase
            },
            'history': memory_summary,
            'attachment_security': relationship.attachment_security,
            'positive_memory_count': len(relationship.positive_memories),
            'conflict_count': len(relationship.conflict_history)
        }