"""
Real-time consciousness monitoring system.
Tracks consciousness states, transitions, and generates comprehensive metrics.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import time
import json
from datetime import datetime, timedelta
import threading
import queue

logger = logging.getLogger(__name__)

class ConsciousnessLevel(Enum):
    """Levels of consciousness"""
    UNCONSCIOUS = 0
    MINIMAL = 1  # Minimal consciousness
    FLUCTUATING = 2  # Varying levels
    SUSTAINED = 3  # Sustained consciousness
    LUCID = 4  # High-level metacognitive awareness
    PEAK = 5  # Peak conscious states

@dataclass
class ConsciousnessSnapshot:
    """A snapshot of consciousness state at a moment in time"""
    timestamp: float
    level: ConsciousnessLevel
    phi_value: float  # IIT integrated information
    workspace_activation: float  # GWT global workspace
    attention_coherence: float  # Attention schema coherence
    prediction_error: float  # Predictive processing
    emotional_valence: float
    cognitive_load: float
    metacognitive_awareness: float
    sensory_integration: float
    memory_access: float
    self_model_coherence: float
    
    def overall_consciousness_score(self) -> float:
        """Calculate overall consciousness score"""
        return np.mean([
            self.phi_value,
            self.workspace_activation,
            self.attention_coherence,
            self.metacognitive_awareness,
            self.self_model_coherence
        ])

@dataclass
class ConsciousnessTransition:
    """Represents a transition between consciousness states"""
    from_level: ConsciousnessLevel
    to_level: ConsciousnessLevel
    timestamp: float
    duration: float
    trigger: str  # What caused the transition
    smooth: bool  # Whether transition was smooth or abrupt

class ConsciousnessMetrics:
    """Comprehensive consciousness metrics tracker"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
        # Time series data
        self.phi_history = deque(maxlen=window_size)
        self.workspace_history = deque(maxlen=window_size)
        self.attention_history = deque(maxlen=window_size)
        self.metacognition_history = deque(maxlen=window_size)
        
        # State tracking
        self.state_durations = defaultdict(float)
        self.transition_counts = defaultdict(int)
        
        # Consciousness quality metrics
        self.quality_metrics = {
            'stability': 0.0,
            'richness': 0.0,
            'coherence': 0.0,
            'continuity': 0.0,
            'depth': 0.0
        }
        
        # Peak states
        self.peak_states = []
        self.consciousness_interruptions = []
        
    def update(self, snapshot: ConsciousnessSnapshot):
        """Update metrics with new snapshot"""
        # Update histories
        self.phi_history.append(snapshot.phi_value)
        self.workspace_history.append(snapshot.workspace_activation)
        self.attention_history.append(snapshot.attention_coherence)
        self.metacognition_history.append(snapshot.metacognitive_awareness)
        
        # Update state duration
        self.state_durations[snapshot.level] += 1 / 60.0  # Assuming 60Hz updates
        
        # Check for peak states
        if snapshot.overall_consciousness_score() > 0.9:
            self.peak_states.append(snapshot)
            
        # Update quality metrics
        self._update_quality_metrics()
        
    def _update_quality_metrics(self):
        """Update consciousness quality metrics"""
        if len(self.phi_history) < 10:
            return
            
        # Stability: Low variance in consciousness measures
        self.quality_metrics['stability'] = 1.0 - np.std(self.phi_history) / (np.mean(self.phi_history) + 1e-8)
        
        # Richness: High values across multiple measures
        recent_values = [
            np.mean(list(self.phi_history)[-10:]),
            np.mean(list(self.workspace_history)[-10:]),
            np.mean(list(self.attention_history)[-10:]),
            np.mean(list(self.metacognition_history)[-10:])
        ]
        self.quality_metrics['richness'] = np.mean(recent_values)
        
        # Coherence: Correlation between different measures
        if len(self.phi_history) > 50:
            phi_work_corr = np.corrcoef(
                list(self.phi_history)[-50:],
                list(self.workspace_history)[-50:]
            )[0, 1]
            self.quality_metrics['coherence'] = abs(phi_work_corr)
            
        # Continuity: Absence of interruptions
        self.quality_metrics['continuity'] = 1.0 / (1.0 + len(self.consciousness_interruptions))
        
        # Depth: Sustained high-level states
        high_level_ratio = sum(1 for p in self.phi_history if p > 0.7) / len(self.phi_history)
        self.quality_metrics['depth'] = high_level_ratio

class ConsciousnessStream:
    """Real-time consciousness data stream"""
    
    def __init__(self, buffer_size: int = 100):
        self.buffer = deque(maxlen=buffer_size)
        self.subscribers = []
        self.is_streaming = False
        self.stream_thread = None
        self.data_queue = queue.Queue()
        
    def start_streaming(self):
        """Start the consciousness data stream"""
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_worker)
        self.stream_thread.start()
        
    def stop_streaming(self):
        """Stop the consciousness data stream"""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join()
            
    def _stream_worker(self):
        """Worker thread for streaming data"""
        while self.is_streaming:
            try:
                # Get data from queue with timeout
                data = self.data_queue.get(timeout=0.1)
                
                # Add to buffer
                self.buffer.append(data)
                
                # Notify subscribers
                for subscriber in self.subscribers:
                    subscriber(data)
                    
            except queue.Empty:
                continue
                
    def push(self, snapshot: ConsciousnessSnapshot):
        """Push new consciousness snapshot to stream"""
        self.data_queue.put(snapshot)
        
    def subscribe(self, callback: Callable[[ConsciousnessSnapshot], None]):
        """Subscribe to consciousness stream"""
        self.subscribers.append(callback)
        
    def get_buffer(self) -> List[ConsciousnessSnapshot]:
        """Get current buffer contents"""
        return list(self.buffer)

class ConsciousnessMonitor:
    """Main consciousness monitoring system"""
    
    def __init__(self, 
                 global_workspace=None,
                 iit_calculator=None,
                 attention_schema=None,
                 predictive_processor=None):
        
        # Core consciousness components
        self.global_workspace = global_workspace
        self.iit_calculator = iit_calculator
        self.attention_schema = attention_schema
        self.predictive_processor = predictive_processor
        
        # Monitoring components
        self.metrics = ConsciousnessMetrics()
        self.stream = ConsciousnessStream()
        
        # State tracking
        self.current_level = ConsciousnessLevel.MINIMAL
        self.last_snapshot = None
        self.monitoring_active = False
        
        # Transition detection
        self.transition_threshold = 0.2
        self.transition_history = deque(maxlen=50)
        
        # Alert system
        self.alert_thresholds = {
            'low_consciousness': 0.2,
            'high_consciousness': 0.9,
            'rapid_fluctuation': 0.5,
            'prolonged_unconscious': 60.0  # seconds
        }
        self.alerts = []
        
        # Update frequency
        self.update_frequency = 60  # Hz
        self.last_update = time.time()
        
    def update(self, inputs: Dict[str, Any]) -> ConsciousnessSnapshot:
        """Update consciousness monitoring with new inputs"""
        current_time = time.time()
        
        # Create consciousness snapshot
        snapshot = self._create_snapshot(inputs, current_time)
        
        # Update metrics
        self.metrics.update(snapshot)
        
        # Stream data
        if self.monitoring_active:
            self.stream.push(snapshot)
            
        # Detect level changes
        new_level = self._determine_consciousness_level(snapshot)
        if new_level != self.current_level:
            self._handle_transition(self.current_level, new_level, current_time)
            self.current_level = new_level
            
        # Check for alerts
        self._check_alerts(snapshot)
        
        # Store snapshot
        self.last_snapshot = snapshot
        self.last_update = current_time
        
        return snapshot
        
    def _create_snapshot(self, inputs: Dict[str, Any], timestamp: float) -> ConsciousnessSnapshot:
        """Create consciousness snapshot from inputs"""
        
        # Get values from different theories
        phi_value = 0.5  # Default
        if self.iit_calculator and 'system_state' in inputs:
            phi_value = self.iit_calculator.calculate_phi(inputs['system_state'])
            
        workspace_activation = 0.5  # Default
        if self.global_workspace and 'modules' in inputs:
            workspace_activation = self.global_workspace.get_activation_level()
            
        attention_coherence = 0.5  # Default
        if self.attention_schema and 'attention_state' in inputs:
            attention_coherence = inputs['attention_state'].get('coherence', 0.5)
            
        prediction_error = 0.0  # Default
        if self.predictive_processor and 'sensory_input' in inputs:
            prediction_error = inputs.get('prediction_error', 0.0)
            
        # Extract other values from inputs
        emotional_valence = inputs.get('emotional_valence', 0.0)
        cognitive_load = inputs.get('cognitive_load', 0.5)
        metacognitive_awareness = inputs.get('metacognitive_awareness', 0.3)
        sensory_integration = inputs.get('sensory_integration', 0.5)
        memory_access = inputs.get('memory_access', 0.5)
        self_model_coherence = inputs.get('self_model_coherence', 0.5)
        
        return ConsciousnessSnapshot(
            timestamp=timestamp,
            level=self.current_level,
            phi_value=phi_value,
            workspace_activation=workspace_activation,
            attention_coherence=attention_coherence,
            prediction_error=prediction_error,
            emotional_valence=emotional_valence,
            cognitive_load=cognitive_load,
            metacognitive_awareness=metacognitive_awareness,
            sensory_integration=sensory_integration,
            memory_access=memory_access,
            self_model_coherence=self_model_coherence
        )
        
    def _determine_consciousness_level(self, snapshot: ConsciousnessSnapshot) -> ConsciousnessLevel:
        """Determine consciousness level from snapshot"""
        score = snapshot.overall_consciousness_score()
        
        if score < 0.1:
            return ConsciousnessLevel.UNCONSCIOUS
        elif score < 0.3:
            return ConsciousnessLevel.MINIMAL
        elif score < 0.5:
            return ConsciousnessLevel.FLUCTUATING
        elif score < 0.7:
            return ConsciousnessLevel.SUSTAINED
        elif score < 0.9:
            return ConsciousnessLevel.LUCID
        else:
            return ConsciousnessLevel.PEAK
            
    def _handle_transition(self, from_level: ConsciousnessLevel, 
                          to_level: ConsciousnessLevel, timestamp: float):
        """Handle consciousness level transition"""
        
        # Calculate transition properties
        duration = timestamp - self.last_update if self.last_snapshot else 0.0
        
        # Determine if smooth or abrupt
        level_diff = abs(to_level.value - from_level.value)
        smooth = level_diff <= 1 and duration > 0.5
        
        # Determine trigger
        trigger = self._identify_transition_trigger()
        
        transition = ConsciousnessTransition(
            from_level=from_level,
            to_level=to_level,
            timestamp=timestamp,
            duration=duration,
            trigger=trigger,
            smooth=smooth
        )
        
        self.transition_history.append(transition)
        self.metrics.transition_counts[f"{from_level.name}->{to_level.name}"] += 1
        
    def _identify_transition_trigger(self) -> str:
        """Identify what triggered the consciousness transition"""
        if not self.last_snapshot:
            return "initialization"
            
        # Check various factors
        if hasattr(self, 'last_snapshot'):
            curr = self.last_snapshot
            
            # Large prediction error
            if curr.prediction_error > 0.7:
                return "high_prediction_error"
                
            # Emotional surge
            if abs(curr.emotional_valence) > 0.8:
                return "emotional_surge"
                
            # Attention shift
            if curr.attention_coherence < 0.3:
                return "attention_disruption"
                
            # Metacognitive event
            if curr.metacognitive_awareness > 0.8:
                return "metacognitive_insight"
                
        return "gradual_change"
        
    def _check_alerts(self, snapshot: ConsciousnessSnapshot):
        """Check for consciousness alerts"""
        
        # Low consciousness alert
        if snapshot.overall_consciousness_score() < self.alert_thresholds['low_consciousness']:
            self.alerts.append({
                'type': 'low_consciousness',
                'timestamp': snapshot.timestamp,
                'details': f"Consciousness score: {snapshot.overall_consciousness_score():.2f}"
            })
            
        # High consciousness alert (peak state)
        if snapshot.overall_consciousness_score() > self.alert_thresholds['high_consciousness']:
            self.alerts.append({
                'type': 'peak_consciousness',
                'timestamp': snapshot.timestamp,
                'details': f"Peak state detected: {snapshot.overall_consciousness_score():.2f}"
            })
            
        # Rapid fluctuation alert
        if len(self.metrics.phi_history) > 10:
            recent_std = np.std(list(self.metrics.phi_history)[-10:])
            if recent_std > self.alert_thresholds['rapid_fluctuation']:
                self.alerts.append({
                    'type': 'rapid_fluctuation',
                    'timestamp': snapshot.timestamp,
                    'details': f"High variability: {recent_std:.2f}"
                })
                
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.monitoring_active = True
        self.stream.start_streaming()
        logger.info("Consciousness monitoring started")
        
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        self.stream.stop_streaming()
        logger.info("Consciousness monitoring stopped")
        
    def get_current_state(self) -> Dict[str, Any]:
        """Get current consciousness state"""
        if not self.last_snapshot:
            return {'status': 'no_data'}
            
        return {
            'level': self.current_level.name,
            'score': self.last_snapshot.overall_consciousness_score(),
            'snapshot': self.last_snapshot,
            'quality_metrics': self.metrics.quality_metrics,
            'active_alerts': self.alerts[-5:],  # Last 5 alerts
            'monitoring_active': self.monitoring_active
        }
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            'current_level': self.current_level.name,
            'quality_metrics': self.metrics.quality_metrics,
            'state_distribution': dict(self.metrics.state_durations),
            'transition_counts': dict(self.metrics.transition_counts),
            'peak_states_count': len(self.metrics.peak_states),
            'average_consciousness': np.mean(self.metrics.phi_history) if self.metrics.phi_history else 0.0,
            'consciousness_stability': self.metrics.quality_metrics['stability'],
            'time_conscious': sum(
                duration for level, duration in self.metrics.state_durations.items()
                if level.value >= ConsciousnessLevel.SUSTAINED.value
            )
        }
        
    def get_timeline(self, duration: timedelta = timedelta(minutes=5)) -> List[Dict[str, Any]]:
        """Get consciousness timeline for specified duration"""
        cutoff_time = time.time() - duration.total_seconds()
        
        timeline = []
        for snapshot in self.stream.get_buffer():
            if snapshot.timestamp > cutoff_time:
                timeline.append({
                    'timestamp': snapshot.timestamp,
                    'level': snapshot.level.name,
                    'score': snapshot.overall_consciousness_score(),
                    'phi': snapshot.phi_value,
                    'workspace': snapshot.workspace_activation,
                    'attention': snapshot.attention_coherence
                })
                
        return timeline
        
    def export_data(self, filepath: str):
        """Export monitoring data to file"""
        data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'monitoring_duration': time.time() - (self.last_snapshot.timestamp if self.last_snapshot else time.time()),
                'total_snapshots': len(self.stream.get_buffer())
            },
            'metrics_summary': self.get_metrics_summary(),
            'quality_metrics': self.metrics.quality_metrics,
            'transitions': [
                {
                    'from': t.from_level.name,
                    'to': t.to_level.name,
                    'timestamp': t.timestamp,
                    'duration': t.duration,
                    'trigger': t.trigger,
                    'smooth': t.smooth
                }
                for t in self.transition_history
            ],
            'peak_states': [
                {
                    'timestamp': s.timestamp,
                    'score': s.overall_consciousness_score(),
                    'phi': s.phi_value
                }
                for s in self.metrics.peak_states[-100:]  # Last 100 peak states
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def subscribe_to_stream(self, callback: Callable[[ConsciousnessSnapshot], None]):
        """Subscribe to real-time consciousness stream"""
        self.stream.subscribe(callback)
        
    def get_consciousness_signature(self) -> Dict[str, Any]:
        """Get unique consciousness signature based on patterns"""
        if len(self.metrics.phi_history) < 100:
            return {'status': 'insufficient_data'}
            
        # Analyze patterns in consciousness data
        phi_array = np.array(list(self.metrics.phi_history))
        workspace_array = np.array(list(self.metrics.workspace_history))
        
        signature = {
            'mean_phi': np.mean(phi_array),
            'phi_variance': np.var(phi_array),
            'workspace_correlation': np.corrcoef(phi_array, workspace_array)[0, 1],
            'dominant_level': max(self.metrics.state_durations.items(), key=lambda x: x[1])[0].name,
            'transition_frequency': len(self.transition_history) / (time.time() - self.transition_history[0].timestamp) if self.transition_history else 0,
            'consciousness_type': self._classify_consciousness_type()
        }
        
        return signature
        
    def _classify_consciousness_type(self) -> str:
        """Classify the type of consciousness based on patterns"""
        
        # Analyze metrics to determine type
        stability = self.metrics.quality_metrics['stability']
        richness = self.metrics.quality_metrics['richness']
        coherence = self.metrics.quality_metrics['coherence']
        
        if stability > 0.8 and richness > 0.7:
            return "stable_rich"
        elif stability > 0.8 and richness < 0.3:
            return "stable_minimal"
        elif stability < 0.3 and richness > 0.7:
            return "fluctuating_rich"
        elif coherence > 0.8:
            return "highly_integrated"
        else:
            return "standard"