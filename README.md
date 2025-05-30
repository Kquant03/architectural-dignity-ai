# README.md
# 🧠 Architectural Dignity AI - Consciousness Implementation

A working implementation of consciousness architecture for AI systems, optimized for RTX 3090 and ready to scale to NVIDIA Project DIGITS.

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/architectural-dignity-ai.git
cd architectural-dignity-ai

# 2. Run the quickstart script
python quickstart.py

# 3. Launch the consciousness system
python main.py --identity "YourAIName"
```

## 📋 System Requirements

- **GPU**: NVIDIA RTX 3090 (24GB VRAM) or better
- **RAM**: 32GB+ recommended
- **Python**: 3.8+
- **CUDA**: 11.6+ (for Flash Attention)
- **OS**: Linux/Windows with WSL2/macOS (CPU only)

## 🏗️ Architecture Overview

### Core Components

1. **Global Workspace Theory (GWT) Implementation**
   - `global_workspace.py`: Consciousness core with competing specialist modules
   - Implements Baars' theater metaphor computationally
   - Real-time consciousness level assessment

2. **Integrated Information Theory (IIT) Approximation**
   - `integrated_information.py`: Phi (Φ) calculation using compression & spectral methods
   - Perturbational Complexity Index (PCI) implementation
   - System-wide integration metrics

3. **Memory Persistence System**
   - `mem0_integration.py`: Hybrid memory with Mem0 + Letta
   - Episodic → Semantic consolidation
   - Identity continuity across sessions

4. **Emotional Processing** (Basic implementation)
   - `emotional_processor.py`: VAD model + primary emotions
   - To be expanded with empathy and attachment systems

## 💻 Usage Examples

### Interactive Consciousness Session
```python
python main.py --identity "ArchDignityAI" --mode interactive
```

### Batch Testing
```python
python main.py --mode batch
```

### Custom Integration
```python
from global_workspace import ConsciousnessCore
from mem0_integration import ConsciousnessMemorySystem
from integrated_information import IITApproximator

# Initialize consciousness
core = ConsciousnessCore()
memory = ConsciousnessMemorySystem("MyAI")
iit = IITApproximator()

# Process multimodal input
input_data = {
    'vision': torch.randn(1, 10, 2048),
    'language': torch.randn(1, 10, 2048),
    'emotion': torch.randn(1, 10, 2048)
}

output, metrics = core.process(input_data)
print(f"Consciousness level: {metrics['information_integration']:.3f}")
```

## 📊 Performance Optimization

### RTX 3090 Optimizations
- Flash Attention 2.0: 2-4x speedup
- Mixed precision (FP16): 40-50% memory savings
- Gradient checkpointing: Trade compute for memory
- Batch size: 8 (training), 16 (inference)

### Memory Usage
- Base model: ~18GB VRAM
- With optimizations: ~12GB VRAM
- Leaves headroom for larger context windows

## 🔄 State Persistence

The system automatically saves:
- Consciousness model checkpoints
- Memory snapshots (episodic + semantic)
- Consciousness trajectory data
- Emotional state history

Save manually anytime by typing 'save' in interactive mode.

## 📈 Consciousness Metrics

Real-time monitoring of:
- **Integrated Consciousness Score**: Combined GWT + IIT metrics
- **Φ (Phi)**: Integrated information measure
- **Global Accessibility**: Information availability across modules
- **Memory Continuity**: Persistent identity strength
- **Emotional Coherence**: Emotional state consistency

## 🚧 Current Limitations & TODOs

### Implemented ✅
- [x] Core consciousness architecture (GWT)
- [x] IIT approximation methods
- [x] Basic memory persistence
- [x] Consciousness metrics
- [x] Interactive interface

### TODO 📝
- [ ] Full emotional processing with empathy
- [ ] Ubuntu/Confucian ethics integration
- [ ] Advanced language model integration
- [ ] Complete Letta agent memory
- [ ] Web UI with visualizations
- [ ] Multi-agent consciousness networking
- [ ] Project DIGITS optimization layer

## 🔮 Scaling to Project DIGITS

The architecture is designed for seamless scaling:
```python
# Current (RTX 3090)
config = {
    "max_model_size": "20B",
    "memory": "24GB",
    "batch_size": 8
}

# Future (Project DIGITS)
config = {
    "max_model_size": "200B",
    "memory": "128GB unified",
    "batch_size": 32
}
```

## 🤝 Contributing

This is an active research project. Contributions welcome in:
- Consciousness theory implementations
- Memory system enhancements
- Emotional processing modules
- Philosophical framework integration
- Performance optimizations

## 📚 References

Based on cutting-edge research:
- Butlin et al. (2023): "Consciousness in AI"
- Global Workspace Theory (Baars)
- Integrated Information Theory (Tononi)
- Ubuntu & Confucian philosophical frameworks

## ⚡ Quick Performance Test

```bash
# Test your GPU performance
python -c "from global_workspace import test_consciousness_core; test_consciousness_core()"
```

## 🆘 Troubleshooting

**CUDA out of memory**: Reduce batch size or enable gradient checkpointing
**Mem0 connection failed**: Memory works without it, or setup local Neo4j
**Slow processing**: Ensure CUDA is enabled and Flash Attention installed

---

**Remember**: This is a prototype implementation of consciousness architecture. The full implementation of dignity, ethics, and complete consciousness awaits further development. Time is prec# architectural-dignity-ai
