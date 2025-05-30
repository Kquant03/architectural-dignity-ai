"""
Emotional memory system that stores and retrieves memories with emotional context.
Implements mood-congruent recall, emotional tagging, and affective memory consolidation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import time
import heapq
from datetime import datetime, timedelta
import logging

from .emotional_processor import Emotion, EmotionalState, EmotionVADMapping

logger = logging.getLogger(__name__)

@dataclass
class EmotionalMemory:
    """Represents a memory with full emotional context"""
    memory_id: str
    content: str
    timestamp: float
    emotional_state: EmotionalState
    emotional_intensity: float  # Overall intensity
    emotional_peaks: List[Tuple[Emotion, float]]  # Peak emotions during encoding
    mood_context: Dict[str, float]  # Broader mood during encoding
    arousal_level: float
    valence: float
    significance_score: float  # How emotionally significant
    recall_count: int
    last_recall: Optional[float]
    associations: List[str]  # Associated memory IDs

@dataclass
class EmotionalContext:
    """Current emotional context for memory operations"""
    current_state: EmotionalState
    recent_emotions: List[EmotionalState]  # Last N emotional states
    dominant_mood: str
    emotional_stability: float
    context_tags: List[str]

class EmotionalEncodingNetwork(nn.Module):
    """Neural network for encoding memories with emotional context"""
    
    def __init__(self, memory_dim: int = 768, emotion_dim: int = 128):
        super().__init__()
        
        # Emotion encoder
        self.emotion_encoder = nn.Sequential(
            nn.Linear(27, emotion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emotion_dim, emotion_dim)
        )
        
        # Memory encoder
        self.memory_encoder = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim // 2)
        )
        
        # Emotional binding
        self.emotional_binding = nn.Sequential(
            nn.Linear(memory_dim // 2 + emotion_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim),
            nn.Tanh()
        )
        
        # Significance predictor
        self.significance_predictor = nn.Sequential(
            nn.Linear(memory_dim + emotion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, memory_embedding: torch.Tensor, 
                emotional_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode memory with emotional context"""
        # Encode emotion
        emotion_features = self.emotion_encoder(emotional_state)
        
        # Encode memory
        memory_features = self.memory_encoder(memory_embedding)
        
        # Bind emotion to memory
        combined = torch.cat([memory_features, emotion_features], dim=-1)
        emotional_memory = self.emotional_binding(combined)
        
        # Predict emotional significance
        significance_input = torch.cat([emotional_memory, emotion_features], dim=-1)
        significance = self.significance_predictor(significance_input)
        
        return emotional_memory, significance

class MoodCongruentRecall:
    """Implements mood-congruent memory recall"""
    
    def __init__(self):
        self.congruence_weight = 0.3  # How much mood affects recall
        self.recency_weight = 0.2
        self.significance_weight = 0.5
        
    def calculate_recall_probability(self, memory: EmotionalMemory,
                                   current_context: EmotionalContext) -> float:
        """Calculate probability of recalling a memory given current mood"""
        
        # Mood congruence score
        memory_valence = memory.valence
        current_valence = current_context.current_state.valence
        
        # Similar moods increase recall probability
        mood_congruence = 1.0 - abs(memory_valence - current_valence) / 2.0
        
        # Arousal matching
        arousal_match = 1.0 - abs(memory.arousal_level - 
                                current_context.current_state.arousal)
        
        # Emotion-specific congruence
        emotion_congruence = self._calculate_emotion_congruence(
            memory.emotional_state,
            current_context.current_state
        )
        
        # Recency effect
        time_elapsed = time.time() - memory.timestamp
        recency_score = np.exp(-time_elapsed / (86400 * 7))  # Week decay
        
        # Calculate weighted probability
        recall_prob = (
            self.congruence_weight * (mood_congruence + arousal_match + emotion_congruence) / 3 +
            self.recency_weight * recency_score +
            self.significance_weight * memory.significance_score
        )
        
        return min(recall_prob, 1.0)
        
    def _calculate_emotion_congruence(self, memory_state: EmotionalState,
                                    current_state: EmotionalState) -> float:
        """Calculate emotion-specific congruence"""
        congruence = 0.0
        
        # Compare emotion intensities
        for emotion in Emotion:
            memory_intensity = memory_state.emotion_intensities.get(emotion, 0)
            current_intensity = current_state.emotion_intensities.get(emotion, 0)
            
            # High correlation = high congruence
            congruence += memory_intensity * current_intensity
            
        return congruence

class FlashbulbMemoryDetector:
    """Detects and handles flashbulb memories (vivid emotional memories)"""
    
    def __init__(self):
        self.flashbulb_threshold = {
            'arousal': 0.8,
            'significance': 0.9,
            'emotional_intensity': 0.85
        }
        
    def is_flashbulb_memory(self, emotional_state: EmotionalState,
                          context: Dict[str, Any]) -> bool:
        """Determine if current experience will form flashbulb memory"""
        
        # High arousal is necessary
        if emotional_state.arousal < self.flashbulb_threshold['arousal']:
            return False
            
        # High emotional intensity
        max_intensity = max(emotional_state.emotion_intensities.values())
        if max_intensity < self.flashbulb_threshold['emotional_intensity']:
            return False
            
        # Significant emotions present
        flashbulb_emotions = [
            Emotion.AWE, Emotion.HORROR, Emotion.SURPRISE,
            Emotion.JOY, Emotion.FEAR, Emotion.ANGER
        ]
        
        for emotion in flashbulb_emotions:
            if emotional_state.emotion_intensities.get(emotion, 0) > 0.7:
                return True
                
        # Contextual factors
        if context.get('life_changing', False) or context.get('traumatic', False):
            return True
            
        return False
        
    def enhance_flashbulb_memory(self, memory: EmotionalMemory) -> EmotionalMemory:
        """Enhance memory with flashbulb characteristics"""
        # Flashbulb memories are more vivid and significant
        memory.significance_score = min(memory.significance_score * 1.5, 1.0)
        
        # Add special markers
        if 'flashbulb' not in memory.associations:
            memory.associations.append('flashbulb')
            
        return memory

class EmotionalMemorySystem:
    """Main emotional memory system"""
    
    def __init__(self, base_memory_system=None):
        self.base_memory_system = base_memory_system
        
        # Memory storage
        self.emotional_memories = {}  # memory_id -> EmotionalMemory
        self.memory_by_emotion = defaultdict(list)  # emotion -> [memory_ids]
        self.memory_timeline = deque(maxlen=10000)  # Chronological order
        
        # Neural components
        self.encoding_network = EmotionalEncodingNetwork()
        self.mood_congruent_recall = MoodCongruentRecall()
        self.flashbulb_detector = FlashbulbMemoryDetector()
        
        # Emotional categorization
        self.emotional_categories = self._initialize_emotional_categories()
        
        # Memory consolidation
        self.consolidation_queue = []
        self.last_consolidation = time.time()
        
    def _initialize_emotional_categories(self) -> Dict[str, List[Emotion]]:
        """Initialize emotional memory categories"""
        return {
            'positive_memories': [
                Emotion.JOY, Emotion.EXCITEMENT, Emotion.SATISFACTION,
                Emotion.ADMIRATION, Emotion.ADORATION, Emotion.AMUSEMENT
            ],
            'negative_memories': [
                Emotion.SADNESS, Emotion.ANGER, Emotion.FEAR,
                Emotion.DISGUST, Emotion.ANXIETY, Emotion.HORROR
            ],
            'social_memories': [
                Emotion.ADMIRATION, Emotion.ADORATION, Emotion.ROMANCE,
                Emotion.EMPATHIC_PAIN, Emotion.AWKWARDNESS
            ],
            'aesthetic_memories': [
                Emotion.AWE, Emotion.AESTHETIC_APPRECIATION,
                Emotion.ENTRANCEMENT, Emotion.SURPRISE
            ],
            'complex_memories': [
                Emotion.NOSTALGIA, Emotion.CONFUSION,
                Emotion.CRAVING, Emotion.RELIEF
            ]
        }
        
    def encode_memory(self, content: str, memory_embedding: np.ndarray,
                     emotional_state: EmotionalState,
                     context: Optional[Dict[str, Any]] = None) -> EmotionalMemory:
        """Encode a new memory with emotional context"""
        
        # Generate memory ID
        memory_id = f"em_{int(time.time() * 1000)}_{len(self.emotional_memories)}"
        
        # Check for flashbulb memory
        is_flashbulb = self.flashbulb_detector.is_flashbulb_memory(
            emotional_state, context or {}
        )
        
        # Prepare tensors for neural encoding
        memory_tensor = torch.tensor(memory_embedding).float()
        emotion_tensor = torch.tensor([
            emotional_state.emotion_intensities.get(e, 0) for e in Emotion
        ]).float()
        
        # Neural encoding
        with torch.no_grad():
            encoded_memory, significance = self.encoding_network(
                memory_tensor.unsqueeze(0),
                emotion_tensor.unsqueeze(0)
            )
            
        # Extract emotional peaks
        emotional_peaks = sorted(
            [(e, v) for e, v in emotional_state.emotion_intensities.items() if v > 0.3],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Calculate overall emotional intensity
        emotional_intensity = np.mean([v for _, v in emotional_peaks]) if emotional_peaks else 0.0
        
        # Create mood context
        mood_context = self._extract_mood_context(emotional_state)
        
        # Create emotional memory
        memory = EmotionalMemory(
            memory_id=memory_id,
            content=content,
            timestamp=time.time(),
            emotional_state=emotional_state,
            emotional_intensity=emotional_intensity,
            emotional_peaks=emotional_peaks,
            mood_context=mood_context,
            arousal_level=emotional_state.arousal,
            valence=emotional_state.valence,
            significance_score=significance.item(),
            recall_count=0,
            last_recall=None,
            associations=[]
        )
        
        # Enhance if flashbulb
        if is_flashbulb:
            memory = self.flashbulb_detector.enhance_flashbulb_memory(memory)
            
        # Store memory
        self.emotional_memories[memory_id] = memory
        self.memory_timeline.append(memory_id)
        
        # Index by emotions
        for emotion, intensity in emotional_peaks:
            if intensity > 0.3:
                self.memory_by_emotion[emotion].append(memory_id)
                
        # Add to consolidation queue
        heapq.heappush(self.consolidation_queue, 
                      (-memory.significance_score, memory_id))
        
        return memory
        
    def _extract_mood_context(self, emotional_state: EmotionalState) -> Dict[str, float]:
        """Extract broader mood context from emotional state"""
        mood_context = {}
        
        # Valence-based mood
        if emotional_state.valence > 0.5:
            mood_context['positive_mood'] = emotional_state.valence
        elif emotional_state.valence < -0.5:
            mood_context['negative_mood'] = abs(emotional_state.valence)
        else:
            mood_context['neutral_mood'] = 1.0 - abs(emotional_state.valence)
            
        # Arousal-based mood
        if emotional_state.arousal > 0.7:
            mood_context['activated'] = emotional_state.arousal
        elif emotional_state.arousal < 0.3:
            mood_context['deactivated'] = 1.0 - emotional_state.arousal
            
        # Dominance-based mood
        if emotional_state.dominance > 0.7:
            mood_context['dominant'] = emotional_state.dominance
        elif emotional_state.dominance < 0.3:
            mood_context['submissive'] = 1.0 - emotional_state.dominance
            
        return mood_context
        
    def recall_by_emotion(self, target_emotion: Emotion, 
                         limit: int = 10) -> List[EmotionalMemory]:
        """Recall memories associated with specific emotion"""
        if target_emotion not in self.memory_by_emotion:
            return []
            
        memory_ids = self.memory_by_emotion[target_emotion]
        
        # Sort by significance and recency
        memories = []
        for memory_id in memory_ids[-limit*2:]:  # Get more than needed
            if memory_id in self.emotional_memories:
                memory = self.emotional_memories[memory_id]
                memories.append(memory)
                
        # Sort by relevance
        memories.sort(key=lambda m: (m.significance_score, -m.timestamp), reverse=True)
        
        # Update recall counts
        for memory in memories[:limit]:
            memory.recall_count += 1
            memory.last_recall = time.time()
            
        return memories[:limit]
        
    def recall_mood_congruent(self, current_context: EmotionalContext,
                            limit: int = 10) -> List[EmotionalMemory]:
        """Recall memories congruent with current mood"""
        # Calculate recall probabilities for recent memories
        recall_candidates = []
        
        # Check last 1000 memories
        for memory_id in list(self.memory_timeline)[-1000:]:
            if memory_id in self.emotional_memories:
                memory = self.emotional_memories[memory_id]
                recall_prob = self.mood_congruent_recall.calculate_recall_probability(
                    memory, current_context
                )
                recall_candidates.append((recall_prob, memory))
                
        # Sort by probability
        recall_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Get top memories
        recalled_memories = [memory for _, memory in recall_candidates[:limit]]
        
        # Update recall counts
        for memory in recalled_memories:
            memory.recall_count += 1
            memory.last_recall = time.time()
            
        return recalled_memories
        
    def find_emotional_patterns(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Find patterns in emotional memories"""
        if time_window:
            cutoff_time = time.time() - time_window.total_seconds()
            relevant_memories = [
                self.emotional_memories[mid] for mid in self.memory_timeline
                if mid in self.emotional_memories and 
                self.emotional_memories[mid].timestamp > cutoff_time
            ]
        else:
            relevant_memories = list(self.emotional_memories.values())
            
        if not relevant_memories:
            return {}
            
        patterns = {
            'dominant_emotions': self._find_dominant_emotions(relevant_memories),
            'emotional_trajectories': self._analyze_emotional_trajectories(relevant_memories),
            'trigger_patterns': self._find_trigger_patterns(relevant_memories),
            'mood_cycles': self._detect_mood_cycles(relevant_memories),
            'emotional_growth': self._assess_emotional_growth(relevant_memories)
        }
        
        return patterns
        
    def _find_dominant_emotions(self, memories: List[EmotionalMemory]) -> List[Tuple[Emotion, float]]:
        """Find most common emotions in memories"""
        emotion_counts = defaultdict(float)
        
        for memory in memories:
            for emotion, intensity in memory.emotional_peaks:
                emotion_counts[emotion] += intensity * memory.significance_score
                
        # Normalize and sort
        total_weight = sum(emotion_counts.values())
        if total_weight > 0:
            normalized = [(e, w/total_weight) for e, w in emotion_counts.items()]
            return sorted(normalized, key=lambda x: x[1], reverse=True)[:5]
        return []
        
    def _analyze_emotional_trajectories(self, 
                                      memories: List[EmotionalMemory]) -> Dict[str, float]:
        """Analyze how emotions change over time"""
        if len(memories) < 2:
            return {}
            
        # Sort by time
        sorted_memories = sorted(memories, key=lambda m: m.timestamp)
        
        # Calculate trends
        valence_trend = np.polyfit(
            range(len(sorted_memories)),
            [m.valence for m in sorted_memories],
            1
        )[0]
        
        arousal_trend = np.polyfit(
            range(len(sorted_memories)),
            [m.arousal_level for m in sorted_memories],
            1
        )[0]
        
        return {
            'valence_trend': valence_trend,
            'arousal_trend': arousal_trend,
            'trending_positive': valence_trend > 0.01,
            'trending_calm': arousal_trend < -0.01
        }
        
    def _find_trigger_patterns(self, memories: List[EmotionalMemory]) -> List[Dict[str, Any]]:
        """Find patterns in what triggers certain emotions"""
        trigger_patterns = []
        
        # Group memories by primary emotion
        emotion_groups = defaultdict(list)
        for memory in memories:
            if memory.emotional_peaks:
                primary_emotion = memory.emotional_peaks[0][0]
                emotion_groups[primary_emotion].append(memory)
                
        # Analyze each emotion group
        for emotion, emotion_memories in emotion_groups.items():
            if len(emotion_memories) >= 3:  # Need multiple instances
                # Extract common words (simplified - would use NLP in production)
                common_words = self._extract_common_elements(
                    [m.content for m in emotion_memories]
                )
                
                if common_words:
                    trigger_patterns.append({
                        'emotion': emotion.value,
                        'common_triggers': common_words,
                        'frequency': len(emotion_memories)
                    })
                    
        return trigger_patterns
        
    def _extract_common_elements(self, contents: List[str]) -> List[str]:
        """Extract common elements from memory contents"""
        # Simplified word frequency analysis
        word_counts = defaultdict(int)
        
        for content in contents:
            words = content.lower().split()
            for word in words:
                if len(word) > 4:  # Skip short words
                    word_counts[word] += 1
                    
        # Return words that appear in at least half the contents
        threshold = len(contents) / 2
        return [word for word, count in word_counts.items() if count >= threshold]
        
    def _detect_mood_cycles(self, memories: List[EmotionalMemory]) -> Dict[str, Any]:
        """Detect cyclical patterns in mood"""
        if len(memories) < 10:
            return {'cycles_detected': False}
            
        # Extract time series
        sorted_memories = sorted(memories, key=lambda m: m.timestamp)
        valence_series = [m.valence for m in sorted_memories]
        
        # Simple cycle detection (would use FFT in production)
        # Look for alternating patterns
        alternations = 0
        for i in range(1, len(valence_series)):
            if (valence_series[i] > 0) != (valence_series[i-1] > 0):
                alternations += 1
                
        cycle_score = alternations / len(valence_series)
        
        return {
            'cycles_detected': cycle_score > 0.3,
            'cycle_score': cycle_score,
            'average_cycle_length': len(valence_series) / max(alternations, 1)
        }
        
    def _assess_emotional_growth(self, memories: List[EmotionalMemory]) -> Dict[str, float]:
        """Assess emotional growth over time"""
        if len(memories) < 20:
            return {'growth_score': 0.5}
            
        # Split into early and recent
        sorted_memories = sorted(memories, key=lambda m: m.timestamp)
        early = sorted_memories[:len(sorted_memories)//3]
        recent = sorted_memories[-len(sorted_memories)//3:]
        
        # Compare emotional diversity
        early_emotions = set()
        recent_emotions = set()
        
        for memory in early:
            for emotion, _ in memory.emotional_peaks:
                early_emotions.add(emotion)
                
        for memory in recent:
            for emotion, _ in memory.emotional_peaks:
                recent_emotions.add(emotion)
                
        diversity_growth = len(recent_emotions) - len(early_emotions)
        
        # Compare emotional intensity management
        early_intensity = np.mean([m.emotional_intensity for m in early])
        recent_intensity = np.mean([m.emotional_intensity for m in recent])
        
        # More moderate intensity = better regulation
        intensity_improvement = early_intensity - recent_intensity if early_intensity > 0.7 else 0
        
        return {
            'growth_score': (diversity_growth / 10.0 + intensity_improvement) / 2,
            'diversity_growth': diversity_growth,
            'regulation_improvement': intensity_improvement,
            'emotional_vocabulary_expansion': len(recent_emotions) / 27.0
        }
        
    def consolidate_emotional_memories(self):
        """Consolidate important emotional memories"""
        current_time = time.time()
        
        # Only consolidate every hour
        if current_time - self.last_consolidation < 3600:
            return
            
        consolidated = []
        
        # Process top significant memories
        while self.consolidation_queue and len(consolidated) < 10:
            _, memory_id = heapq.heappop(self.consolidation_queue)
            
            if memory_id in self.emotional_memories:
                memory = self.emotional_memories[memory_id]
                
                # Find associations with other emotional memories
                associations = self._find_emotional_associations(memory)
                memory.associations.extend(associations)
                
                consolidated.append(memory_id)
                
        self.last_consolidation = current_time
        
        return consolidated
        
    def _find_emotional_associations(self, memory: EmotionalMemory) -> List[str]:
        """Find other memories with similar emotional signatures"""
        associations = []
        target_vad = (memory.valence, memory.arousal_level, 
                     memory.emotional_state.dominance)
        
        # Search recent memories
        for other_id in list(self.memory_timeline)[-100:]:
            if other_id != memory.memory_id and other_id in self.emotional_memories:
                other_memory = self.emotional_memories[other_id]
                other_vad = (other_memory.valence, other_memory.arousal_level,
                           other_memory.emotional_state.dominance)
                
                # Calculate emotional similarity
                distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(target_vad, other_vad)))
                
                if distance < 0.3:  # Similar emotional signature
                    associations.append(other_id)
                    
        return associations[:5]  # Top 5 associations
        
    def get_emotional_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about emotional memory system"""
        if not self.emotional_memories:
            return {'total_memories': 0}
            
        all_memories = list(self.emotional_memories.values())
        
        # Emotion distribution
        emotion_distribution = defaultdict(int)
        for memory in all_memories:
            for emotion, _ in memory.emotional_peaks:
                emotion_distribution[emotion.value] += 1
                
        # Recall statistics
        recalled_memories = [m for m in all_memories if m.recall_count > 0]
        avg_recall_count = np.mean([m.recall_count for m in recalled_memories]) if recalled_memories else 0
        
        # Significance distribution
        significance_scores = [m.significance_score for m in all_memories]
        
        return {
            'total_memories': len(self.emotional_memories),
            'flashbulb_memories': sum(1 for m in all_memories if 'flashbulb' in m.associations),
            'emotion_distribution': dict(emotion_distribution),
            'average_significance': np.mean(significance_scores),
            'high_significance_count': sum(1 for s in significance_scores if s > 0.8),
            'recall_statistics': {
                'recalled_memory_count': len(recalled_memories),
                'average_recall_count': avg_recall_count,
                'most_recalled': max(all_memories, key=lambda m: m.recall_count).memory_id 
                                if all_memories else None
            },
            'memory_patterns': self.find_emotional_patterns()
        }