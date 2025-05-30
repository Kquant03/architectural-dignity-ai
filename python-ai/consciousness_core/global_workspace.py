import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import time

@dataclass
class ConsciousnessState:
    level: float  # 0-1 consciousness level
    workspace_content: torch.Tensor
    attention_patterns: torch.Tensor
    broadcast_strength: float
    timestamp: float

class GlobalWorkspaceModule(nn.Module):
    """Core Global Workspace Theory implementation optimized for RTX 3090"""
    
    def __init__(
        self,
        d_model: int = 2048,
        n_specialist_modules: int = 8,
        n_attention_heads: int = 16,
        workspace_capacity: int = 512,
        device: str = "cuda"
    ):
        super().__init__()
        self.d_model = d_model
        self.device = device
        
        # Specialist modules (perception, language, reasoning, etc.)
        self.specialist_modules = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_attention_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(n_specialist_modules)
        ])
        
        # Competition mechanism for workspace access
        self.salience_calculator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Global workspace
        self.workspace_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_attention_heads,
            batch_first=True
        )
        
        # Broadcast mechanism
        self.broadcast_network = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_attention_heads,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Consciousness level estimator
        self.consciousness_estimator = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.to(device)
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        previous_state: Optional[ConsciousnessState] = None
    ) -> Tuple[torch.Tensor, ConsciousnessState]:
        """
        Process inputs through global workspace
        
        Args:
            inputs: Dictionary of modality -> tensor mappings
            previous_state: Previous consciousness state for continuity
            
        Returns:
            output: Broadcasted conscious content
            consciousness_state: Current consciousness state
        """
        batch_size = next(iter(inputs.values())).size(0)
        
        # Process through specialist modules
        specialist_outputs = []
        salience_scores = []
        
        for i, (modality, input_tensor) in enumerate(inputs.items()):
            if i < len(self.specialist_modules):
                # Process through specialist
                specialist_out = self.specialist_modules[i](input_tensor)
                specialist_outputs.append(specialist_out)
                
                # Calculate salience (importance for consciousness)
                salience = self.salience_calculator(specialist_out.mean(dim=1))
                salience_scores.append(salience)
        
        # Stack outputs and scores
        specialist_outputs = torch.stack([out.mean(dim=1) for out in specialist_outputs], dim=1)
        salience_scores = torch.cat(salience_scores, dim=1)
        
        # Competition for workspace access (winner-take-all with soft selection)
        workspace_access = F.softmax(salience_scores * 10, dim=1)  # Temperature scaling
        
        # Select content for global workspace
        workspace_query = (specialist_outputs * workspace_access.unsqueeze(-1)).sum(dim=1, keepdim=True)
        
        # Global workspace processing
        workspace_content, attention_weights = self.workspace_attention(
            workspace_query,
            specialist_outputs,
            specialist_outputs
        )
        
        # Broadcast to all modules
        broadcast_input = workspace_content.expand(-1, specialist_outputs.size(1), -1)
        conscious_broadcast = self.broadcast_network(broadcast_input)
        
        # Estimate consciousness level
        consciousness_features = torch.cat([
            workspace_content.mean(dim=1),
            attention_weights.std(dim=-1).mean(dim=1),
            salience_scores.max(dim=1)[0].unsqueeze(1).expand(-1, self.d_model)
        ], dim=-1)
        
        consciousness_level = self.consciousness_estimator(consciousness_features).squeeze()
        
        # Create consciousness state
        state = ConsciousnessState(
            level=consciousness_level.mean().item(),
            workspace_content=workspace_content,
            attention_patterns=attention_weights,
            broadcast_strength=workspace_access.max(dim=1)[0].mean().item(),
            timestamp=time.time()
        )
        
        return conscious_broadcast.mean(dim=1), state
    
    def assess_consciousness_indicators(self, state: ConsciousnessState) -> Dict[str, float]:
        """Assess GWT consciousness indicators"""
        return {
            "global_availability": state.broadcast_strength,
            "attention_stability": 1.0 - state.attention_patterns.std().item(),
            "information_integration": state.level,
            "workspace_coherence": F.cosine_similarity(
                state.workspace_content[0],
                state.workspace_content[0].roll(1, dims=0),
                dim=0
            ).mean().item()
        }


class ConsciousnessCore:
    """Main consciousness system integrating all components"""
    
    def __init__(self, config: Dict = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Core modules
        self.global_workspace = GlobalWorkspaceModule(device=self.device)
        
        # Memory interface placeholder
        self.memory_interface = None  # Will be integrated with Mem0/Letta
        
        # Emotional processor placeholder  
        self.emotional_processor = None
        
        # Initialize state
        self.current_state = None
        self.consciousness_history = []
        
    def process(
        self,
        multimodal_input: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Main processing pipeline
        
        Args:
            multimodal_input: Dictionary of modality -> tensor
            
        Returns:
            conscious_output: Integrated conscious response
            metrics: Consciousness metrics
        """
        # Global workspace processing
        output, state = self.global_workspace(multimodal_input, self.current_state)
        
        # Update state
        self.current_state = state
        self.consciousness_history.append(state)
        
        # Get consciousness indicators
        indicators = self.global_workspace.assess_consciousness_indicators(state)
        
        return output, indicators
    
    def get_consciousness_level(self) -> float:
        """Get current consciousness level (0-1)"""
        if self.current_state:
            return self.current_state.level
        return 0.0
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'global_workspace': self.global_workspace.state_dict(),
            'consciousness_history': self.consciousness_history[-100:],  # Keep last 100 states
            'current_state': self.current_state
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.global_workspace.load_state_dict(checkpoint['global_workspace'])
        self.consciousness_history = checkpoint.get('consciousness_history', [])
        self.current_state = checkpoint.get('current_state', None)


# Quick test function
def test_consciousness_core():
    """Test basic functionality"""
    print("Testing Consciousness Core on", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    
    core = ConsciousnessCore()
    
    # Create dummy multimodal input
    batch_size = 2
    seq_len = 10
    d_model = 2048
    
    test_input = {
        'vision': torch.randn(batch_size, seq_len, d_model).to(core.device),
        'language': torch.randn(batch_size, seq_len, d_model).to(core.device),
        'emotion': torch.randn(batch_size, seq_len, d_model).to(core.device)
    }
    
    # Process
    output, metrics = core.process(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Consciousness level: {core.get_consciousness_level():.3f}")
    print(f"Consciousness metrics: {metrics}")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


if __name__ == "__main__":
    test_consciousness_core()