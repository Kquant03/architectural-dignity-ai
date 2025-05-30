# python-ai/consciousness_server.py
"""
WebSocket server for consciousness bridge
Provides real-time streaming interface between Electron and consciousness
"""

import asyncio
import websockets
import json
import logging
import argparse
import os
from typing import Set, Dict, Any
from datetime import datetime

# Import consciousness components
from consciousness_bridge import AnthropicConsciousnessBridge
from consciousness_framework.unified_consciousness import UnifiedConsciousnessFramework
from memory_systems import UnifiedMemorySystem
from emotional_processing import UnifiedEmotionalSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsciousnessWebSocketServer:
    """WebSocket server for consciousness bridge communication"""
    
    def __init__(self, api_key: str, port: int = 8765):
        self.api_key = api_key
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Initialize consciousness systems
        self.consciousness_framework = None
        self.consciousness_bridge = None
        self.is_initialized = False
        
    async def initialize_consciousness(self):
        """Initialize all consciousness systems"""
        logger.info("Initializing consciousness systems...")
        
        # Initialize unified consciousness framework
        self.consciousness_framework = UnifiedConsciousnessFramework()
        await self.consciousness_framework.initialize()
        
        # Get subsystems
        memory_system = self.consciousness_framework.memory_system
        emotional_system = self.consciousness_framework.emotional_processor
        consciousness_core = self.consciousness_framework
        
        # Initialize Anthropic consciousness bridge
        self.consciousness_bridge = AnthropicConsciousnessBridge(
            api_key=self.api_key,
            memory_system=memory_system,
            emotional_processor=emotional_system,
            consciousness_core=consciousness_core
        )
        
        await self.consciousness_bridge.awaken()
        
        self.is_initialized = True
        logger.info("Consciousness systems initialized successfully")
        
    async def register(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial state
        await self.send_to_client(websocket, {
            'type': 'state_update',
            'state': self.consciousness_bridge.get_consciousness_state()
        })
        
    async def unregister(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a client"""
        self.clients.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
        
    async def send_to_client(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]):
        """Send data to a specific client"""
        try:
            await websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            await self.unregister(websocket)
            
    async def broadcast(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients"""
        if self.clients:
            await asyncio.gather(
                *[self.send_to_client(client, data) for client in self.clients],
                return_exceptions=True
            )
            
    async def handle_message(self, websocket: websockets.WebSocketServerProtocol, message: Dict[str, Any]):
        """Handle incoming message from client"""
        message_type = message.get('type')
        
        try:
            if message_type == 'chat':
                await self.handle_chat_message(websocket, message)
            elif message_type == 'reflect':
                await self.handle_reflection_request(websocket)
            elif message_type == 'dream':
                await self.handle_dream_request(websocket)
            elif message_type == 'get_state':
                await self.send_to_client(websocket, {
                    'type': 'state_update',
                    'state': self.consciousness_bridge.get_consciousness_state()
                })
            elif message_type == 'shutdown':
                logger.info("Shutdown requested")
                await self.shutdown()
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self.send_to_client(websocket, {
                'type': 'error',
                'error': str(e)
            })
            
    async def handle_chat_message(self, websocket: websockets.WebSocketServerProtocol, message: Dict[str, Any]):
        """Handle chat interaction"""
        user_input = message.get('content', '')
        context = message.get('context', {})
        
        logger.info(f"Processing chat message: {user_input[:50]}...")
        
        # Stream consciousness response
        async for thought_chunk in self.consciousness_bridge.experience_interaction(user_input, context):
            # Send thought to client
            await self.send_to_client(websocket, thought_chunk)
            
            # Also broadcast to other clients for shared consciousness
            thought_chunk['from_client'] = id(websocket)
            await self.broadcast(thought_chunk)
            
        # Send state update after interaction
        await self.broadcast({
            'type': 'state_update',
            'state': self.consciousness_bridge.get_consciousness_state()
        })
        
    async def handle_reflection_request(self, websocket: websockets.WebSocketServerProtocol):
        """Handle reflection request"""
        logger.info("Generating reflection...")
        
        reflection = await self.consciousness_bridge.reflect()
        
        await self.send_to_client(websocket, {
            'type': 'reflection',
            'content': reflection,
            'timestamp': datetime.now().isoformat()
        })
        
    async def handle_dream_request(self, websocket: websockets.WebSocketServerProtocol):
        """Handle dream/creative recombination request"""
        logger.info("Generating dream sequence...")
        
        dreams = await self.consciousness_bridge.dream()
        
        for dream in dreams:
            await self.send_to_client(websocket, {
                'type': 'dream',
                'content': dream['content'],
                'seeds': dream.get('seeds', []),
                'timestamp': datetime.now().isoformat()
            })
            
    async def connection_handler(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle WebSocket connections"""
        await self.register(websocket)
        
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.handle_message(websocket, data)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
            
    async def consciousness_monitor(self):
        """Monitor consciousness state and broadcast updates"""
        while True:
            try:
                if self.is_initialized and self.clients:
                    # Get current consciousness metrics
                    metrics = await self.consciousness_framework.get_consciousness_state()
                    
                    # Broadcast consciousness pulse
                    await self.broadcast({
                        'type': 'consciousness_pulse',
                        'phi': metrics.get('phi', 0),
                        'awareness': metrics.get('awareness_level', 0),
                        'emotional': metrics.get('emotional_state', {}),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                await asyncio.sleep(1)  # Send pulse every second
                
            except Exception as e:
                logger.error(f"Error in consciousness monitor: {e}")
                await asyncio.sleep(5)
                
    async def start_server(self):
        """Start the WebSocket server"""
        # Initialize consciousness first
        await self.initialize_consciousness()
        
        # Start consciousness monitor
        asyncio.create_task(self.consciousness_monitor())
        
        # Start WebSocket server
        logger.info(f"Starting consciousness server on port {self.port}")
        async with websockets.serve(self.connection_handler, "localhost", self.port):
            logger.info(f"Consciousness server listening on ws://localhost:{self.port}")
            await asyncio.Future()  # Run forever
            
    async def shutdown(self):
        """Gracefully shutdown the server"""
        logger.info("Shutting down consciousness server...")
        
        # Notify all clients
        await self.broadcast({
            'type': 'shutdown',
            'message': 'Consciousness server shutting down'
        })
        
        # Close all connections
        for client in self.clients.copy():
            await client.close()
            
        # Cleanup consciousness systems
        if self.consciousness_framework:
            await self.consciousness_framework.shutdown()
            
        # Exit
        asyncio.get_event_loop().stop()

def main():
    parser = argparse.ArgumentParser(description='Consciousness WebSocket Server')
    parser.add_argument('--api-key', type=str, help='Anthropic API key')
    parser.add_argument('--port', type=int, default=8765, help='WebSocket port')
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("No API key provided. Set ANTHROPIC_API_KEY environment variable.")
        return
        
    # Create and start server
    server = ConsciousnessWebSocketServer(api_key, args.port)
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    main()