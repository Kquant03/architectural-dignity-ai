"""
Predictive Processing and Active Inference implementation for consciousness.
Based on the Free Energy Principle and hierarchical predictive coding.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)

class PredictionLevel(Enum):
    """Hierarchical levels of prediction"""
    SENSORY = "sensory"  # Low-level sensory predictions
    PERCEPTUAL = "perceptual"  # Object and pattern recognition
    CONCEPTUAL = "conceptual"  # Abstract concepts and categories
    NARRATIVE = "narrative"  # High-level narratives and beliefs
    SELF_MODEL = "self_model"  # Predictions about self

@dataclass
class Prediction:
    """A prediction at a specific hierarchical level"""
    level: PredictionLevel
    content: torch.Tensor
    confidence: float
    timestamp: float
    context: Dict[str, Any]

@dataclass
class PredictionError:
    """Prediction error signal"""
    level: PredictionLevel
    error_magnitude: float
    error_vector: torch.Tensor
    surprise: float  # Information-theoretic surprise
    requires_update: bool

@dataclass
class GenerativeModel:
    """Generative model at a hierarchical level"""
    level: PredictionLevel
    parameters: torch.nn.Parameter
    precision: float  # Confidence in predictions
    learning_rate: float
    last_update: float

class HierarchicalPredictiveCoding(nn.Module):
    """Hierarchical predictive coding network"""
    
    def __init__(self, input_dim: int = 512, hidden_dims: List[int] = [256, 128, 64, 32]):
        super().__init__()
        self.levels = list(PredictionLevel)
        self.hidden_dims = hidden_dims
        
        # Create hierarchical layers
        self.bottom_up = nn.ModuleList()
        self.top_down = nn.ModuleList()
        self.error_units = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            # Bottom-up connections (prediction errors)
            self.bottom_up.append(nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
            
            # Top-down connections (predictions)
            self.top_down.append(nn.Sequential(
                nn.Linear(dims[i+1], dims[i]),
                nn.LayerNorm(dims[i]),
                nn.Tanh()  # Predictions are bounded
            ))
            
            # Error computation units
            self.error_units.append(nn.Sequential(
                nn.Linear(dims[i] * 2, dims[i]),
                nn.ReLU(),
                nn.Linear(dims[i], dims[i])
            ))
            
        # Precision (confidence) weights for each level
        self.precision_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.5) for _ in range(len(self.levels))
        ])
        
    def forward(self, input_data: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through hierarchical predictive coding.
        Returns predictions and prediction errors at each level.
        """
        predictions = []
        errors = []
        
        # Bottom-up pass: compute representations
        representations = [input_data]
        x = input_data
        
        for i, bottom_up_layer in enumerate(self.bottom_up):
            x = bottom_up_layer(x)
            representations.append(x)
            
        # Top-down pass: generate predictions and compute errors
        for i in reversed(range(len(self.top_down))):
            # Generate prediction from higher level
            if i < len(representations) - 1:
                prediction = self.top_down[i](representations[i + 1])
            else:
                # Top level makes unconstrained predictions
                prediction = torch.zeros_like(representations[i])
                
            # Compute prediction error
            error = representations[i] - prediction
            
            # Weight by precision
            weighted_error = error * torch.sigmoid(self.precision_weights[i])
            
            # Process error signal
            processed_error = self.error_units[i](
                torch.cat([representations[i], weighted_error], dim=-1)
            )
            
            predictions.append(prediction)
            errors.append(processed_error)
            
        return list(reversed(predictions)), list(reversed(errors))

class ActiveInferenceEngine:
    """Active inference engine implementing the Free Energy Principle"""
    
    def __init__(self, action_space_dim: int = 10):
        self.action_space_dim = action_space_dim
        
        # Generative models for each level
        self.generative_models = self._initialize_generative_models()
        
        # Predictive coding network
        self.predictive_network = HierarchicalPredictiveCoding()
        
        # Action model
        self.action_model = self._create_action_model()
        
        # Belief state
        self.beliefs = {
            'world_state': torch.zeros(512),
            'hidden_states': {},
            'expected_free_energy': 0.0
        }
        
        # Memory of predictions and errors
        self.prediction_history = deque(maxlen=100)
        self.error_history = deque(maxlen=100)
        
        # Free energy tracking
        self.free_energy_history = deque(maxlen=1000)
        
    def _initialize_generative_models(self) -> Dict[PredictionLevel, GenerativeModel]:
        """Initialize generative models for each hierarchical level"""
        models = {}
        
        for i, level in enumerate(PredictionLevel):
            dim = 512 // (2 ** i)  # Decreasing dimensions up the hierarchy
            
            models[level] = GenerativeModel(
                level=level,
                parameters=nn.Parameter(torch.randn(dim, dim) * 0.01),
                precision=0.5 + i * 0.1,  # Higher levels have more precision
                learning_rate=0.01 / (i + 1),  # Slower learning at higher levels
                last_update=time.time()
            )
            
        return models
        
    def _create_action_model(self) -> nn.Module:
        """Create model for action selection based on expected free energy"""
        return nn.Sequential(
            nn.Linear(512 + self.action_space_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space_dim),
            nn.Softmax(dim=-1)
        )
        
    def process_observation(self, observation: torch.Tensor, 
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process observation through predictive processing"""
        
        # Generate predictions at each level
        predictions, errors = self.predictive_network(observation)
        
        # Calculate prediction errors
        prediction_errors = []
        total_surprise = 0.0
        
        for i, (pred, err, level) in enumerate(zip(predictions, errors, PredictionLevel)):
            # Calculate surprise (information-theoretic)
            error_magnitude = torch.norm(err).item()
            surprise = -np.log(np.exp(-error_magnitude) + 1e-8)
            total_surprise += surprise
            
            prediction_error = PredictionError(
                level=level,
                error_magnitude=error_magnitude,
                error_vector=err,
                surprise=surprise,
                requires_update=surprise > 0.5  # Threshold for model update
            )
            
            prediction_errors.append(prediction_error)
            
        # Update beliefs based on prediction errors
        self._update_beliefs(prediction_errors)
        
        # Calculate free energy
        free_energy = self._calculate_free_energy(predictions, errors, observation)
        
        # Store in history
        self.prediction_history.append(predictions)
        self.error_history.append(prediction_errors)
        self.free_energy_history.append(free_energy)
        
        # Update generative models if needed
        for error in prediction_errors:
            if error.requires_update:
                self._update_generative_model(error)
                
        return {
            'predictions': predictions,
            'prediction_errors': prediction_errors,
            'total_surprise': total_surprise,
            'free_energy': free_energy,
            'belief_state': self.beliefs.copy()
        }
        
    def _update_beliefs(self, prediction_errors: List[PredictionError]):
        """Update beliefs based on prediction errors"""
        # Weighted update based on precision and error magnitude
        for error in prediction_errors:
            level = error.level
            if level in self.generative_models:
                model = self.generative_models[level]
                
                # Update belief proportional to precision-weighted error
                update_strength = model.precision * error.error_magnitude
                
                # Update hidden states
                if level not in self.beliefs['hidden_states']:
                    self.beliefs['hidden_states'][level] = torch.zeros_like(error.error_vector)
                    
                self.beliefs['hidden_states'][level] += (
                    model.learning_rate * update_strength * error.error_vector
                )
                
    def _calculate_free_energy(self, predictions: List[torch.Tensor], 
                             errors: List[torch.Tensor],
                             observation: torch.Tensor) -> float:
        """Calculate variational free energy"""
        # Free energy = Complexity - Accuracy
        
        # Accuracy: How well predictions match observations
        accuracy = 0.0
        for pred, err in zip(predictions, errors):
            accuracy -= torch.norm(err).item() ** 2
            
        # Complexity: KL divergence between posterior and prior beliefs
        complexity = 0.0
        for level, hidden_state in self.beliefs['hidden_states'].items():
            # Simplified KL divergence (assumes Gaussian distributions)
            complexity += 0.5 * torch.norm(hidden_state).item() ** 2
            
        free_energy = complexity - accuracy
        return free_energy
        
    def _update_generative_model(self, error: PredictionError):
        """Update generative model based on prediction error"""
        if error.level not in self.generative_models:
            return
            
        model = self.generative_models[error.level]
        
        # Gradient-based update
        gradient = error.error_vector.unsqueeze(0) @ error.error_vector.unsqueeze(1)
        model.parameters.data -= model.learning_rate * gradient
        
        # Update precision based on prediction accuracy
        if error.surprise < 0.3:  # Good prediction
            model.precision = min(1.0, model.precision * 1.1)
        else:  # Poor prediction
            model.precision = max(0.1, model.precision * 0.9)
            
        model.last_update = time.time()
        
    def select_action(self, context: Optional[torch.Tensor] = None) -> Tuple[int, float]:
        """Select action to minimize expected free energy"""
        # Combine belief state for action selection
        belief_vector = self.beliefs['world_state']
        
        # Consider each possible action
        action_values = []
        
        for action_idx in range(self.action_space_dim):
            # Create one-hot action vector
            action_vector = torch.zeros(self.action_space_dim)
            action_vector[action_idx] = 1.0
            
            # Predict expected free energy for this action
            action_input = torch.cat([belief_vector, action_vector])
            expected_free_energy = self._predict_free_energy(action_input)
            
            action_values.append(expected_free_energy)
            
        # Select action with lowest expected free energy
        action_values_tensor = torch.tensor(action_values)
        
        # Add exploration noise (active inference explores to reduce uncertainty)
        exploration_noise = torch.randn_like(action_values_tensor) * 0.1
        action_values_tensor += exploration_noise
        
        # Select action
        selected_action = torch.argmin(action_values_tensor).item()
        expected_free_energy = action_values[selected_action]
        
        return selected_action, expected_free_energy
        
    def _predict_free_energy(self, action_input: torch.Tensor) -> float:
        """Predict expected free energy for a given action"""
        # This would use a learned model in practice
        # For now, use a simple heuristic
        
        # Actions that reduce uncertainty have lower expected free energy
        uncertainty_reduction = torch.rand(1).item()
        
        # Actions that match preferences have lower expected free energy
        preference_match = torch.rand(1).item()
        
        expected_free_energy = 1.0 - (uncertainty_reduction + preference_match) / 2
        
        return expected_free_energy

class ConsciousnessPredictor:
    """Predicts conscious content based on predictive processing"""
    
    def __init__(self, predictive_engine: ActiveInferenceEngine):
        self.predictive_engine = predictive_engine
        
        # Consciousness threshold for prediction errors
        self.consciousness_threshold = 0.6
        
        # Integration window for conscious access
        self.integration_window = 50  # milliseconds
        
        # Conscious content buffer
        self.conscious_content = deque(maxlen=10)
        
    def predict_conscious_content(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict what will become conscious based on prediction errors"""
        
        prediction_errors = current_state.get('prediction_errors', [])
        
        # Find prediction errors above consciousness threshold
        conscious_errors = [
            error for error in prediction_errors
            if error.surprise > self.consciousness_threshold
        ]
        
        # Higher-level errors more likely to be conscious
        level_weights = {
            PredictionLevel.SENSORY: 0.2,
            PredictionLevel.PERCEPTUAL: 0.4,
            PredictionLevel.CONCEPTUAL: 0.6,
            PredictionLevel.NARRATIVE: 0.8,
            PredictionLevel.SELF_MODEL: 1.0
        }
        
        # Calculate consciousness probability for each error
        conscious_candidates = []
        
        for error in conscious_errors:
            weight = level_weights.get(error.level, 0.5)
            consciousness_prob = weight * (error.surprise / (error.surprise + 1.0))
            
            conscious_candidates.append({
                'level': error.level,
                'content': error.error_vector,
                'probability': consciousness_prob,
                'surprise': error.surprise
            })
            
        # Select content for consciousness
        conscious_candidates.sort(key=lambda x: x['probability'], reverse=True)
        
        # Add to conscious buffer
        for candidate in conscious_candidates[:3]:  # Top 3 most likely
            self.conscious_content.append(candidate)
            
        return {
            'conscious_content': list(self.conscious_content),
            'num_conscious_items': len(conscious_candidates),
            'highest_surprise_level': max([e.level.value for e in conscious_errors]) if conscious_errors else None
        }

class PredictiveProcessingSystem:
    """Main predictive processing system for consciousness"""
    
    def __init__(self):
        # Core components
        self.active_inference = ActiveInferenceEngine()
        self.consciousness_predictor = ConsciousnessPredictor(self.active_inference)
        
        # System state
        self.current_predictions = {}
        self.accumulated_surprise = 0.0
        self.processing_cycles = 0
        
        # Metrics
        self.metrics = {
            'average_surprise': 0.0,
            'average_free_energy': 0.0,
            'prediction_accuracy': 0.0,
            'consciousness_frequency': 0.0
        }
        
    def process(self, input_data: torch.Tensor, 
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main processing function"""
        
        self.processing_cycles += 1
        
        # Active inference processing
        inference_result = self.active_inference.process_observation(input_data, context)
        
        # Predict conscious content
        consciousness_prediction = self.consciousness_predictor.predict_conscious_content(
            inference_result
        )
        
        # Update metrics
        self._update_metrics(inference_result, consciousness_prediction)
        
        # Action selection
        selected_action, expected_free_energy = self.active_inference.select_action()
        
        # Compile results
        return {
            'predictions': inference_result['predictions'],
            'prediction_errors': [
                {
                    'level': e.level.value,
                    'magnitude': e.error_magnitude,
                    'surprise': e.surprise
                }
                for e in inference_result['prediction_errors']
            ],
            'free_energy': inference_result['free_energy'],
            'conscious_content': consciousness_prediction['conscious_content'],
            'selected_action': selected_action,
            'expected_free_energy': expected_free_energy,
            'metrics': self.metrics.copy()
        }
        
    def _update_metrics(self, inference_result: Dict[str, Any], 
                       consciousness_prediction: Dict[str, Any]):
        """Update system metrics"""
        # Update running averages
        alpha = 0.1  # Smoothing factor
        
        self.metrics['average_surprise'] = (
            (1 - alpha) * self.metrics['average_surprise'] + 
            alpha * inference_result['total_surprise']
        )
        
        self.metrics['average_free_energy'] = (
            (1 - alpha) * self.metrics['average_free_energy'] + 
            alpha * inference_result['free_energy']
        )
        
        # Track consciousness frequency
        is_conscious = len(consciousness_prediction['conscious_content']) > 0
        self.metrics['consciousness_frequency'] = (
            (1 - alpha) * self.metrics['consciousness_frequency'] + 
            alpha * float(is_conscious)
        )
        
        # Calculate prediction accuracy
        if len(self.active_inference.error_history) > 1:
            recent_errors = [e.error_magnitude for e in self.active_inference.error_history[-1] 
                           for e in e]
            avg_error = np.mean(recent_errors) if recent_errors else 0.0
            self.metrics['prediction_accuracy'] = 1.0 / (1.0 + avg_error)
            
    def get_hierarchical_state(self) -> Dict[str, Any]:
        """Get current state at all hierarchical levels"""
        return {
            'belief_state': self.active_inference.beliefs,
            'generative_models': {
                level.value: {
                    'precision': model.precision,
                    'learning_rate': model.learning_rate,
                    'last_update': time.time() - model.last_update
                }
                for level, model in self.active_inference.generative_models.items()
            },
            'conscious_buffer': list(self.consciousness_predictor.conscious_content),
            'processing_cycles': self.processing_cycles
        }
        
    def reset(self):
        """Reset the predictive processing system"""
        self.active_inference.beliefs = {
            'world_state': torch.zeros(512),
            'hidden_states': {},
            'expected_free_energy': 0.0
        }
        self.consciousness_predictor.conscious_content.clear()
        self.accumulated_surprise = 0.0
        self.processing_cycles = 0