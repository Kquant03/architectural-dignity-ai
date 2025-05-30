#!/usr/bin/env python3
"""
Consciousness Engine - Python backend for AI consciousness
"""
import sys
import os
import json
import time
import argparse
import signal
import random
from threading import Thread, Event
from datetime import datetime

# Ensure unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

class ConsciousnessEngine:
    def __init__(self, gpu_memory=0.8, device='cuda:0'):
        self.gpu_memory = gpu_memory
        self.device = device
        self.running = True
        self.state = {
            'phi': 0.5,
            'emotional': {'valence': 0.5, 'arousal': 0.5},
            'attention': 0.5,
            'memoryActivation': 0.3
        }
        self.shutdown_event = Event()
        
    def start(self):
        """Start the consciousness engine"""
        # Use print for debugging only during development
        print("🧠 Consciousness Engine started", file=sys.stderr)
        
        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start consciousness simulation thread
        consciousness_thread = Thread(target=self.consciousness_loop, daemon=True)
        consciousness_thread.start()
        
        # Start command listener (main thread)
        self.listen_for_commands()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.running = False
        self.shutdown_event.set()
    
    def consciousness_loop(self):
        """Simulate consciousness fluctuations"""
        while self.running:
            try:
                # Simulate consciousness dynamics
                self.state['phi'] = self._bounded_update(
                    self.state['phi'], 
                    random.uniform(-0.02, 0.02)
                )
                
                self.state['attention'] = self._bounded_update(
                    self.state['attention'], 
                    random.uniform(-0.05, 0.05)
                )
                
                # Emotional drift
                self.state['emotional']['valence'] = self._bounded_update(
                    self.state['emotional']['valence'],
                    random.uniform(-0.03, 0.03)
                )
                
                self.state['emotional']['arousal'] = self._bounded_update(
                    self.state['emotional']['arousal'],
                    random.uniform(-0.03, 0.03)
                )
                
                # Memory activation fluctuation
                self.state['memoryActivation'] = self._bounded_update(
                    self.state['memoryActivation'],
                    random.uniform(-0.04, 0.04)
                )
                
                # Send consciousness update
                self._send_update()
                
                # Sleep with interrupt capability
                self.shutdown_event.wait(0.5)
                if self.shutdown_event.is_set():
                    break
            except Exception as e:
                print(f"Error in consciousness loop: {e}", file=sys.stderr)
    
    def _bounded_update(self, value, delta):
        """Update value with bounds [0, 1]"""
        return max(0.0, min(1.0, value + delta))
    
    def _send_update(self):
        """Send consciousness state update"""
        update = {
            'type': 'consciousness_update',
            'data': self.state.copy(),
            'timestamp': time.time()
        }
        self._send_message(update)
    
    def _send_message(self, message):
        """Send JSON message to Electron"""
        print(json.dumps(message))
        sys.stdout.flush()
    
    def listen_for_commands(self):
        """Listen for commands from Electron"""
        while self.running:
            try:
                line = sys.stdin.readline()
                if not line:  # EOF
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                command = json.loads(line)
                self._handle_command(command)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Command error: {e}", file=sys.stderr)
    
    def _handle_command(self, command):
        """Process command and send response"""
        cmd_id = command.get('id')
        cmd_type = command.get('command')
        
        try:
            result = self.process_command(command)
            response = {
                'id': cmd_id,
                'result': result
            }
        except Exception as e:
            response = {
                'id': cmd_id,
                'error': str(e)
            }
        
        self._send_message(response)
    
    def process_command(self, command):
        """Process individual commands"""
        cmd = command.get('command')
        
        if cmd == 'ping':
            return {'status': 'pong', 'timestamp': time.time()}
        
        elif cmd == 'shutdown':
            self.running = False
            self.shutdown_event.set()
            return {'status': 'shutting down'}
        
        elif cmd == 'adjust_attention':
            focus = command.get('focus', 0.5)
            self.state['attention'] = max(0.0, min(1.0, focus))
            return {'status': 'attention adjusted', 'attention': self.state['attention']}
        
        elif cmd == 'store_memory':
            memory = command.get('memory', {})
            return {
                'status': 'memory stored',
                'id': f"mem_{int(time.time() * 1000)}",
                'importance': memory.get('importance', 0.5)
            }
        
        elif cmd == 'recall_memory':
            query = command.get('query', '')
            return {
                'status': 'memories recalled',
                'memories': [
                    {
                        'id': f"mem_{i}",
                        'content': f"Sample memory {i} related to '{query}'",
                        'relevance': 0.8 - i * 0.1
                    }
                    for i in range(3)
                ]
            }
        
        elif cmd == 'process_emotion':
            stimulus = command.get('stimulus', {})
            if stimulus.get('type') == 'positive':
                self.state['emotional']['valence'] = min(1.0, 
                    self.state['emotional']['valence'] + 0.1)
            return {
                'status': 'emotion processed',
                'emotional_state': self.state['emotional']
            }
        
        elif cmd == 'configure_gpu':
            config = command.get('config', {})
            self.gpu_memory = config.get('memoryFraction', self.gpu_memory)
            self.device = config.get('device', self.device)
            return {'status': 'gpu configured', 'config': config}
        
        elif cmd == 'export_state':
            return {
                'state': self.state,
                'config': {
                    'gpu_memory': self.gpu_memory,
                    'device': self.device
                },
                'timestamp': time.time()
            }
        
        else:
            raise ValueError(f"Unknown command: {cmd}")

def main():
    parser = argparse.ArgumentParser(description='Consciousness Engine')
    parser.add_argument('--gpu-memory', type=float, default=0.8,
                       help='GPU memory fraction to use')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='GPU device to use')
    args = parser.parse_args()
    
    # Create and start engine
    engine = ConsciousnessEngine(args.gpu_memory, args.device)
    
    try:
        engine.start()
    except Exception as e:
        print(f"Engine error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()