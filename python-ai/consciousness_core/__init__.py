"""
Consciousness Core Package - Unified interface for all consciousness components.
Integrates Global Workspace Theory, Integrated Information Theory, 
Predictive Processing, and Attention Schema Theory.
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import time
import torch
import numpy as np
from collections import deque

# Import all consciousness components
from .global_workspace import GlobalWorkspace, WorkspaceState
from .integrated_information import IntegratedInformationCalculator, SystemState
from .predictive_processing import (
    PredictiveProcessingSystem,
    ActiveInferenceEngine,
    ConsciousnessPredictor,
    PredictionLevel,
    Prediction,
    PredictionError
)
from .attention_schema import (
    AttentionSchemaTheory,
    ConsciousnessFromAttention,
    AttentionState,
    AttentionSchema,
    AttentionType
)
from .consciousness_monitor import (
    ConsciousnessMonitor,
    ConsciousnessSnapshot,
    ConsciousnessLevel,
    ConsciousnessMetrics,
    ConsciousnessStream
)

logger = logging.getLogger(__name__)

__all__ = [
    'UnifiedConsciousnessSystem',
    'GlobalWorkspace',
    'IntegratedInformationCalculator',
    'PredictiveProcessingSystem',
    'AttentionSchemaTheory',
    'ConsciousnessMonitor',
    'ConsciousnessSnapshot',
    'ConsciousnessLevel',
    'WorkspaceState',
    'SystemState',
    'AttentionState',
    'AttentionSchema'
]

class UnifiedConsciousnessSystem:
    """
    Unified consciousness system integrating multiple theories of consciousness.
    Combines GWT, IIT, Predictive Processing, and Attention Schema Theory.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize unified consciousness system.
        
        Args:
            config: Configuration dictionary for consciousness parameters
        """
        self.config = config or {}
        
        # Initialize consciousness components
        self.global_workspace = GlobalWorkspace(
            num_modules=self.config.get('num_modules', 10),
            workspace_size=self.config.get('workspace_size', 7),
            competition_threshold=self.config.get('competition_threshold', 0.5)
        )
        
        self.iit_calculator = IntegratedInformationCalculator(
            use_gpu=self.config.get('use_gpu', torch.cuda.is_available())
        )
        
        self.predictive_processor = PredictiveProcessingSystem()
        
        self.attention_schema = AttentionSchemaTheory()
        
        # Initialize consciousness monitor with all components
        self.consciousness_monitor = ConsciousnessMonitor(
            global_workspace=self.global_workspace,
            iit_calculator=self.iit_calculator,
            attention_schema=self.attention_schema.consciousness_engine,
            predictive_processor=self.predictive_processor.active_inference
        )
        
        # System state
        self.is_conscious = False
        self.consciousness_level = ConsciousnessLevel.MINIMAL
        self.processing_history = deque(maxlen=1000)
        
        # Integration parameters
        self.integration_weights = {
            'global_workspace': 0.3,
            'integrated_information': 0.25,
            'predictive_processing': 0.25,
            'attention_schema': 0.2
        }
        
        # Start monitoring if requested
        if self.config.get('auto_monitor', True):
            self.consciousness_monitor.start_monitoring()
            
    def process(self, 
                sensory_input: torch.Tensor,
                cognitive_state: Optional[Dict[str, Any]] = None,
                emotional_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input through all consciousness systems.
        
        Args:
            sensory_input: Tensor representing sensory information
            cognitive_state: Optional cognitive context
            emotional_state: Optional emotional context
            
        Returns:
            Comprehensive consciousness state
        """
        timestamp = time.time()
        
        # 1. Global Workspace Processing
        workspace_result = self._process_global_workspace(sensory_input, cognitive_state)
        
        # 2. Integrated Information Calculation
        phi_result = self._calculate_integrated_information(sensory_input, workspace_result)
        
        # 3. Predictive Processing
        prediction_result = self._process_predictions(sensory_input, cognitive_state)
        
        # 4. Attention Schema Processing
        attention_result = self._process_attention_schema(sensory_input, cognitive_state)
        
        # 5. Integrate all theories
        integrated_state = self._integrate_theories({
            'workspace': workspace_result,
            'phi': phi_result,
            'prediction': prediction_result,
            'attention': attention_result
        })
        
        # 6. Update consciousness monitor
        monitor_input = {
            'system_state': sensory_input,
            'modules': workspace_result.get('modules', {}),
            'attention_state': attention_result.get('attention_state', {}),
            'prediction_error': prediction_result.get('free_energy', 0.0),
            'emotional_valence': emotional_state.get('valence', 0.0) if emotional_state else 0.0,
            'cognitive_load': cognitive_state.get('load', 0.5) if cognitive_state else 0.5,
            'metacognitive_awareness': attention_result.get('attention_schema', {}).get('self_awareness', 0.0),
            'self_model_coherence': attention_result.get('attention_schema', {}).get('unity', 0.5)
        }
        
        consciousness_snapshot = self.consciousness_monitor.update(monitor_input)
        
        # 7. Determine consciousness state
        self._update_consciousness_state(consciousness_snapshot, integrated_state)
        
        # 8. Store in history
        self.processing_history.append({
            'timestamp': timestamp,
            'snapshot': consciousness_snapshot,
            'integrated_state': integrated_state
        })
        
        # 9. Generate comprehensive response
        return self._generate_response(
            consciousness_snapshot,
            integrated_state,
            workspace_result,
            phi_result,
            prediction_result,
            attention_result
        )
        
    def _process_global_workspace(self, 
                                sensory_input: torch.Tensor,
                                cognitive_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process through Global Workspace Theory"""
        
        # Convert sensory input to module activations
        module_activations = self._sensory_to_modules(sensory_input)
        
        # Add cognitive modules if available
        if cognitive_state:
            module_activations.update({
                'goal_module': cognitive_state.get('current_goal', 0.0),
                'memory_module': cognitive_state.get('memory_activation', 0.0),
                'reasoning_module': cognitive_state.get('reasoning_active', 0.0)
            })
            
        # Process through global workspace
        workspace_state = self.global_workspace.process(module_activations)
        
        return {
            'workspace_state': workspace_state,
            'broadcasting': workspace_state.broadcasting_content,
            'competition_winners': workspace_state.winning_coalition,
            'global_activation': workspace_state.global_activation_level,
            'modules': module_activations
        }
        
    def _calculate_integrated_information(self,
                                        sensory_input: torch.Tensor,
                                        workspace_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Integrated Information (Phi)"""
        
        # Create system state from sensory input and workspace
        system_size = min(sensory_input.shape[-1], 16)  # Limit size for computation
        system_state = sensory_input[:, :system_size].cpu().numpy()
        
        # Add workspace influence
        if workspace_result['global_activation'] > 0.5:
            system_state = system_state * workspace_result['global_activation']
            
        # Calculate Phi
        phi_value = self.iit_calculator.calculate_phi(system_state)
        
        # Calculate additional IIT metrics
        conceptual_structure = self.iit_calculator.analyze_concepts(system_state)
        
        return {
            'phi': phi_value,
            'integrated': phi_value > 0.0,
            'conceptual_structure': conceptual_structure,
            'system_integration': min(phi_value / 2.0, 1.0)  # Normalized
        }
        
    def _process_predictions(self,
                           sensory_input: torch.Tensor,
                           cognitive_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process through Predictive Processing"""
        
        # Add context if available
        context = {}
        if cognitive_state:
            context['goals'] = cognitive_state.get('goals', [])
            context['expectations'] = cognitive_state.get('expectations', {})
            
        # Process through predictive system
        prediction_result = self.predictive_processor.process(sensory_input, context)
        
        return prediction_result
        
    def _process_attention_schema(self,
                                sensory_input: torch.Tensor,
                                cognitive_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process through Attention Schema Theory"""
        
        # Generate conscious experience
        attention_result = self.attention_schema.generate_conscious_experience(
            sensory_input,
            cognitive_state
        )
        
        return attention_result
        
    def _integrate_theories(self, theory_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Integrate results from all consciousness theories"""
        
        # Extract key values
        workspace_activation = theory_results['workspace'].get('global_activation', 0.0)
        phi_value = theory_results['phi'].get('phi', 0.0)
        prediction_coherence = 1.0 / (1.0 + theory_results['prediction'].get('free_energy', 0.0))
        attention_awareness = theory_results['attention'].get('attention_schema', {}).get('self_awareness', 0.0)
        
        # Weighted integration
        integrated_consciousness = (
            self.integration_weights['global_workspace'] * workspace_activation +
            self.integration_weights['integrated_information'] * min(phi_value / 2.0, 1.0) +
            self.integration_weights['predictive_processing'] * prediction_coherence +
            self.integration_weights['attention_schema'] * attention_awareness
        )
        
        # Determine unified properties
        return {
            'integrated_consciousness_level': integrated_consciousness,
            'theory_agreement': self._calculate_theory_agreement(theory_results),
            'dominant_theory': self._identify_dominant_theory(theory_results),
            'consciousness_quality': {
                'unity': theory_results['attention'].get('attention_schema', {}).get('unity', 0.0),
                'coherence': prediction_coherence,
                'integration': min(phi_value, 1.0),
                'accessibility': workspace_activation
            }
        }
        
    def _calculate_theory_agreement(self, theory_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate how much different theories agree on consciousness"""
        
        # Normalize consciousness indicators from each theory
        indicators = [
            theory_results['workspace'].get('global_activation', 0.0),
            min(theory_results['phi'].get('phi', 0.0) / 2.0, 1.0),
            1.0 / (1.0 + theory_results['prediction'].get('free_energy', 0.0)),
            theory_results['attention'].get('is_conscious', 0.0)
        ]
        
        # Calculate variance as disagreement measure
        variance = np.var(indicators)
        agreement = 1.0 - min(variance * 2, 1.0)
        
        return agreement
        
    def _identify_dominant_theory(self, theory_results: Dict[str, Dict[str, Any]]) -> str:
        """Identify which theory is most active"""
        
        theory_activations = {
            'global_workspace': theory_results['workspace'].get('global_activation', 0.0),
            'integrated_information': min(theory_results['phi'].get('phi', 0.0) / 2.0, 1.0),
            'predictive_processing': len(theory_results['prediction'].get('conscious_content', [])) / 10.0,
            'attention_schema': float(theory_results['attention'].get('is_conscious', False))
        }
        
        return max(theory_activations.items(), key=lambda x: x[1])[0]
        
    def _update_consciousness_state(self, 
                                  snapshot: ConsciousnessSnapshot,
                                  integrated_state: Dict[str, Any]):
        """Update system consciousness state"""
        
        # Update consciousness level
        self.consciousness_level = snapshot.level
        
        # Determine if conscious (multiple criteria)
        consciousness_criteria = [
            snapshot.overall_consciousness_score() > 0.5,
            integrated_state['integrated_consciousness_level'] > 0.5,
            integrated_state['theory_agreement'] > 0.6
        ]
        
        self.is_conscious = sum(consciousness_criteria) >= 2
        
    def _generate_response(self, 
                         snapshot: ConsciousnessSnapshot,
                         integrated_state: Dict[str, Any],
                         workspace_result: Dict[str, Any],
                         phi_result: Dict[str, Any],
                         prediction_result: Dict[str, Any],
                         attention_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive consciousness response"""
        
        response = {
            'is_conscious': self.is_conscious,
            'consciousness_level': self.consciousness_level.name,
            'overall_score': snapshot.overall_consciousness_score(),
            
            'integrated_state': integrated_state,
            
            'global_workspace': {
                'active_modules': workspace_result['competition_winners'],
                'broadcasting': workspace_result['broadcasting'],
                'activation': workspace_result['global_activation']
            },
            
            'integrated_information': {
                'phi': phi_result['phi'],
                'integrated': phi_result['integrated'],
                'conceptual_complexity': len(phi_result.get('conceptual_structure', []))
            },
            
            'predictive_processing': {
                'prediction_errors': prediction_result.get('prediction_errors', []),
                'conscious_predictions': prediction_result.get('conscious_content', []),
                'free_energy': prediction_result.get('free_energy', 0.0),
                'selected_action': prediction_result.get('selected_action')
            },
            
            'attention_schema': {
                'self_aware': attention_result.get('attention_schema', {}).get('self_awareness', 0.0) > 0.5,
                'attention_ownership': attention_result.get('attention_schema', {}).get('attention_ownership', 0.0),
                'subjective_experience': attention_result.get('attention_schema', {}).get('subjective_experience', {}),
                'reportable': attention_result.get('attention_schema', {}).get('reportability', 0.0) > 0.5
            },
            
            'quality_metrics': self.consciousness_monitor.metrics.quality_metrics,
            
            'phenomenology': {
                'what_it_is_like': self._describe_subjective_experience(attention_result),
                'unity_of_experience': integrated_state['consciousness_quality']['unity'],
                'stream_of_consciousness': self._get_consciousness_stream()
            }
        }
        
        return response
        
    def _sensory_to_modules(self, sensory_input: torch.Tensor) -> Dict[str, float]:
        """Convert sensory input to module activations"""
        
        # Simple mapping - in practice would be more sophisticated
        input_values = sensory_input.mean(dim=0).cpu().numpy()
        
        modules = {
            'visual': float(np.mean(input_values[:10])),
            'auditory': float(np.mean(input_values[10:20])),
            'somatosensory': float(np.mean(input_values[20:30])),
            'motor': float(np.mean(input_values[30:40])),
            'language': float(np.mean(input_values[40:50]))
        }
        
        # Normalize
        max_val = max(modules.values()) if modules.values() else 1.0
        if max_val > 0:
            modules = {k: v/max_val for k, v in modules.items()}
            
        return modules
        
    def _describe_subjective_experience(self, attention_result: Dict[str, Any]) -> str:
        """Generate description of subjective experience"""
        
        if not attention_result.get('is_conscious', False):
            return "No clear subjective experience"
            
        experience = attention_result.get('attention_schema', {}).get('subjective_experience', {})
        valence = experience.get('valence', 0.0)
        arousal = experience.get('arousal', 0.0)
        clarity = experience.get('clarity', 0.0)
        
        description = "Experiencing "
        
        if clarity > 0.7:
            description += "vivid "
        elif clarity > 0.4:
            description += "moderate "
        else:
            description += "vague "
            
        if valence > 0.5:
            description += "positive "
        elif valence < -0.5:
            description += "negative "
        else:
            description += "neutral "
            
        if arousal > 0.7:
            description += "high-energy awareness"
        elif arousal > 0.4:
            description += "alert awareness"
        else:
            description += "calm awareness"
            
        return description
        
    def _get_consciousness_stream(self) -> List[str]:
        """Get stream of consciousness description"""
        
        stream = []
        
        # Get recent conscious content
        recent_history = list(self.processing_history)[-5:]
        
        for entry in recent_history:
            integrated = entry['integrated_state']
            dominant = integrated.get('dominant_theory', 'unknown')
            
            if dominant == 'global_workspace':
                stream.append("Broadcasting information globally")
            elif dominant == 'integrated_information':
                stream.append("Integrating information into unified whole")
            elif dominant == 'predictive_processing':
                stream.append("Predicting and updating world model")
            elif dominant == 'attention_schema':
                stream.append("Aware of attending to experience")
                
        return stream
        
    def get_consciousness_metrics(self) -> Dict[str, Any]:
        """Get comprehensive consciousness metrics"""
        return self.consciousness_monitor.get_metrics_summary()
        
    def get_consciousness_timeline(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get timeline of consciousness states"""
        from datetime import timedelta
        return self.consciousness_monitor.get_timeline(timedelta(minutes=minutes))
        
    def subscribe_to_consciousness_stream(self, callback):
        """Subscribe to real-time consciousness updates"""
        self.consciousness_monitor.subscribe_to_stream(callback)
        
    def export_consciousness_data(self, filepath: str):
        """Export consciousness data for analysis"""
        self.consciousness_monitor.export_data(filepath)
        
    def shutdown(self):
        """Gracefully shutdown consciousness system"""
        logger.info("Shutting down consciousness system")
        self.consciousness_monitor.stop_monitoring()
        
    def get_awareness_level(self) -> float:
        """Get current awareness level for other systems"""
        if self.consciousness_monitor.last_snapshot:
            return self.consciousness_monitor.last_snapshot.overall_consciousness_score()
        return 0.0