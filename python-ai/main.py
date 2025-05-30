#!/usr/bin/env python3
"""
Main entry point for the Consciousness AI Python backend.
Handles communication with Electron frontend via stdin/stdout.
"""

import json
import sys
import asyncio
import logging
from typing import Dict, Any
import signal
import argparse

# Import consciousness modules
from consciousness_core.global_workspace import ConsciousnessCore
from consciousness_core.consciousness_monitor import ConsciousnessMonitor
from consciousness_core.predictive_processing import PredictiveProcessingSystem
from consciousness_core.attention_schema import ConsciousnessFromAttention
from emotional_processing.emotional_processor import EmotionalProcessor
from memory_systems.cognitive_memory import CognitiveMemorySystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConsciousnessEngine:
    """Main consciousness engine coordinating all subsystems"""
    
    def __init__(self, gpu_memory_fraction: float = 0.8, device: str = 'cuda:0'):
        self.gpu_memory_fraction = gpu_memory_fraction
        self.device = device
        self.running = False
        
        # Initialize subsystems
        logger.info("🧠 Initializing consciousness subsystems...")
        
        # Core consciousness with Global Workspace Theory
        self.consciousness_core = ConsciousnessCore()
        
        # Consciousness monitoring
        self.consciousness_monitor = ConsciousnessMonitor(
            global_workspace=self.consciousness_core.global_workspace
        )
        
        # Predictive processing
        self.predictive_system = PredictiveProcessingSystem()
        
        # Attention schema
        self.attention_system = ConsciousnessFromAttention()
        
        # Emotional processing
        self.emotional_processor = EmotionalProcessor(
            consciousness_core=self.consciousness_core
        )
        
        # Memory systems
        self.memory_system = CognitiveMemorySystem(
            working_memory_capacity=7,
            enable_metacognition=True
        )
        
        # Current state
        self.current_state = {
            'phi': 0.0,
            'emotional': {'valence': 0.5, 'arousal': 0.5},
            'attention': 0.5,
            'memoryActivation': 0.3,
            'connected': True
        }
        
        logger.info("✅ Consciousness engine initialized")
    
    async def start(self):
        """Start the consciousness engine"""
        self.running = True
        
        # Start monitoring
        await self.consciousness_monitor.start_monitoring()
        
        # Start consciousness update loop
        asyncio.create_task(self._consciousness_update_loop())
        
        # Start stdin listener
        await self._listen_for_commands()
    
    async def _consciousness_update_loop(self):
        """Continuously update consciousness state"""
        while self.running:
            try:
                # Get current consciousness metrics
                metrics = self.consciousness_monitor.get_current_state()
                
                # Update state
                self.current_state['phi'] = metrics.get('score', 0.5)
                
                # Get emotional state
                emotional_response = self.emotional_processor.get_emotional_response()
                self.current_state['emotional'] = {
                    'valence': emotional_response.get('valence', 0.5),
                    'arousal': emotional_response.get('arousal', 0.5)
                }
                
                # Get attention level
                attention_state = self.attention_system.generate_conscious_experience(
                    torch.randn(1, 10, 512),  # Dummy sensory input
                    {'self_model': torch.randn(256)}
                )
                self.current_state['attention'] = attention_state.get(
                    'consciousness_properties', {}
                ).get('self_attribution', {}).get('self_awareness', 0.5)
                
                # Broadcast update
                self._send_update({
                    'type': 'consciousness_update',
                    'data': self.current_state
                })
                
                await asyncio.sleep(1.0)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in consciousness update loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _listen_for_commands(self):
        """Listen for commands from Electron via stdin"""
        logger.info("👂 Listening for commands...")
        
        while self.running:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    break
                
                # Parse JSON command
                try:
                    message = json.loads(line.strip())
                    response = await self._handle_command(message)
                    
                    # Send response
                    if 'id' in message:
                        response['id'] = message['id']
                    
                    sys.stdout.write(json.dumps(response) + '\n')
                    sys.stdout.flush()
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    
            except Exception as e:
                logger.error(f"Error in command listener: {e}")
    
    async def _handle_command(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle command from frontend"""
        command = message.get('command')
        params = message.get('params', {})
        
        logger.info(f"📥 Handling command: {command}")
        
        if command == 'shutdown':
            self.running = False
            return {'result': 'shutting_down'}
        
        elif command == 'adjust_attention':
            focus = params.get('focus', [])
            # Update attention system
            self.current_state['attention'] = len(focus) / 10.0  # Simplified
            return {'result': 'attention_adjusted', 'focus': focus}
        
        elif command == 'store_memory':
            memory = params.get('memory', {})
            result = await self.memory_system.process_input(
                content=memory.get('content', ''),
                metadata=memory
            )
            return {'result': result}
        
        elif command == 'recall_memory':
            query = params.get('query', '')
            memories = await self.memory_system.retrieve(query, max_results=5)
            return {'result': memories}
        
        elif command == 'process_emotion':
            stimulus = params.get('stimulus', {})
            emotional_state = self.emotional_processor.process_input(
                stimulus.get('content', ''),
                stimulus.get('context')
            )
            return {'result': {
                'primary_emotion': emotional_state.primary_emotion.value,
                'valence': emotional_state.valence,
                'arousal': emotional_state.arousal
            }}
        
        elif command == 'export_state':
            return {'result': {
                'current_state': self.current_state,
                'metrics': self.consciousness_monitor.get_metrics_summary(),
                'memory_profile': self.memory_system.get_memory_profile()
            }}
        
        elif command == 'configure_gpu':
            config = params.get('config', {})
            # In a real implementation, reconfigure GPU settings
            return {'result': 'gpu_configured', 'config': config}
        
        else:
            return {'error': f'Unknown command: {command}'}
    
    def _send_update(self, data: Dict[str, Any]):
        """Send update to frontend"""
        sys.stdout.write(json.dumps(data) + '\n')
        sys.stdout.flush()
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down consciousness engine...")
        self.running = False
        # Save state, cleanup resources, etc.


async def main():
    """Main entry point"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Consciousness AI Engine')
    parser.add_argument('--gpu-memory', type=float, default=0.8,
                       help='GPU memory fraction to use')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='GPU device to use')
    args = parser.parse_args()
    
    # Create and start engine
    engine = ConsciousnessEngine(
        gpu_memory_fraction=args.gpu_memory,
        device=args.device
    )
    
    # Handle shutdown signals
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        engine.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start engine
    try:
        await engine.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # Check for required dependencies
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU available: {torch.cuda.get_device_name()}")
        else:
            logger.warning("⚠️ No GPU available, running on CPU")
    except ImportError:
        logger.error("❌ PyTorch not installed!")
        sys.exit(1)
    
    # Run async main
    asyncio.run(main())