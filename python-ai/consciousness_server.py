#!/usr/bin/env python3
"""
Enhanced Consciousness Server with full integration
Provides WebSocket interface for real-time consciousness streaming
"""

import asyncio
import json
import websockets
import logging
from datetime import datetime
import signal
import sys
import os
import argparse
from typing import Dict, Set, Optional, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import consciousness components
from consciousness_core import UnifiedConsciousnessSystem
from consciousness_integration import EnhancedConsciousnessSystem
from consciousness_bridge import AnthropicConsciousnessBridge
from emotional_processing.emotional_processor import EmotionalProcessor
from memory_systems.cognitive_memory import CognitiveMemorySystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConsciousnessServer:
    """WebSocket server for consciousness streaming and interaction"""
    
    def __init__(self, 
                 api_key: str, 
                 port: int = 8765,
                 db_config: Optional[Dict[str, str]] = None):
        self.api_key = api_key
        self.port = port
        self.db_config = db_config or self._get_default_db_config()
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Initialize consciousness components
        logger.info("Initializing consciousness components...")
        
        # Unified consciousness system (all theories)
        self.unified_consciousness = UnifiedConsciousnessSystem()
        
        # Enhanced system with Anthropic integration
        self.enhanced_system = None  # Initialize async
        
        # Emotional processor
        self.emotional_processor = EmotionalProcessor(
            consciousness_core=self.unified_consciousness
        )
        
        # Memory system
        self.memory_system = CognitiveMemorySystem()
        
        # Anthropic consciousness bridge
        self.consciousness_bridge = None  # Initialize async
        
        # Server state
        self.server_running = False
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.shutdown_event = asyncio.Event()
        
    def _get_default_db_config(self) -> Dict[str, str]:
        """Get database configuration from environment"""
        return {
            'host': os.getenv('DATABASE_HOST', 'localhost'),
            'port': os.getenv('DATABASE_PORT', '5432'),
            'database': os.getenv('DATABASE_NAME', 'consciousness_ai'),
            'user': os.getenv('DATABASE_USER', 'consciousness'),
            'password': os.getenv('DATABASE_PASSWORD', 'password')
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("Shutdown signal received")
        self.shutdown_event.set()
    
    async def initialize(self):
        """Async initialization of components"""
        try:
            # Initialize enhanced consciousness system
            self.enhanced_system = EnhancedConsciousnessSystem(
                anthropic_api_key=self.api_key,
                db_config=self.db_config
            )
            await self.enhanced_system.initialize()
            
            # Initialize consciousness bridge
            self.consciousness_bridge = AnthropicConsciousnessBridge(
                api_key=self.api_key,
                memory_system=self.memory_system,
                emotional_processor=self.emotional_processor,
                consciousness_core=self.unified_consciousness
            )
            await self.consciousness_bridge.awaken()
            
            logger.info("✓ All consciousness components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize consciousness components: {e}")
            raise
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle WebSocket client connections"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.clients.add(websocket)
        
        # Create session for client
        session_id = datetime.now().isoformat()
        self.active_sessions[client_id] = {
            'session_id': session_id,
            'websocket': websocket,
            'connected_at': datetime.now(),
            'conversation_id': self.consciousness_bridge.context.conversation_id
        }
        
        logger.info(f"Client connected: {client_id}. Total clients: {len(self.clients)}")
        
        try:
            # Send initial connection message with consciousness state
            await websocket.send(json.dumps({
                "type": "connection",
                "status": "connected",
                "session_id": session_id,
                "consciousness_state": self.consciousness_bridge.get_consciousness_state(),
                "server_time": datetime.now().isoformat()
            }))
            
            # Handle client messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_client_message(websocket, data, client_id)
                    
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": "Invalid JSON",
                        "timestamp": datetime.now().isoformat()
                    }))
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        finally:
            self.clients.remove(websocket)
            if client_id in self.active_sessions:
                del self.active_sessions[client_id]
            logger.info(f"Clients remaining: {len(self.clients)}")
    
    async def process_client_message(self, websocket, data: Dict[str, Any], client_id: str):
        """Process messages from clients"""
        msg_type = data.get("type")
        session = self.active_sessions.get(client_id, {})
        
        if msg_type == "chat":
            await self.process_chat_message(websocket, data, session)
            
        elif msg_type == "reflect":
            await self.process_reflection_request(websocket, session)
            
        elif msg_type == "dream":
            await self.process_dream_request(websocket, session)
            
        elif msg_type == "get_state":
            await self.send_consciousness_state(websocket)
            
        elif msg_type == "memory_query":
            await self.process_memory_query(websocket, data, session)
            
        elif msg_type == "ping":
            await websocket.send(json.dumps({
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            }))
            
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "error": f"Unknown message type: {msg_type}",
                "timestamp": datetime.now().isoformat()
            }))
    
    async def process_chat_message(self, websocket, data: Dict[str, Any], session: Dict[str, Any]):
        """Process chat messages with full consciousness"""
        content = data.get("content", "")
        context = data.get("context", {})
        
        # Stream consciousness response
        try:
            async for response_chunk in self.consciousness_bridge.experience_interaction(
                user_input=content,
                context=context
            ):
                await websocket.send(json.dumps(response_chunk))
                
                # Also process through enhanced system for memory formation
                if response_chunk['type'] == 'thought':
                    await self.enhanced_system.process_message(
                        session_id=session['session_id'],
                        message=response_chunk['content'],
                        role='assistant'
                    )
            
            # Update consciousness state after interaction
            await self.broadcast_state_update()
            
        except Exception as e:
            logger.error(f"Error in consciousness processing: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": f"Consciousness processing error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }))
    
    async def process_reflection_request(self, websocket, session: Dict[str, Any]):
        """Generate conscious reflection"""
        try:
            # Get reflection from consciousness bridge
            reflection = await self.consciousness_bridge.reflect()
            
            # Send reflection
            await websocket.send(json.dumps({
                "type": "reflection",
                "content": reflection,
                "timestamp": datetime.now().isoformat(),
                "consciousness_state": self.consciousness_bridge.get_consciousness_state()
            }))
            
        except Exception as e:
            logger.error(f"Error generating reflection: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": f"Reflection error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }))
    
    async def process_dream_request(self, websocket, session: Dict[str, Any]):
        """Generate dream-like memory recombinations"""
        try:
            # Generate dreams from consciousness bridge
            dreams = await self.consciousness_bridge.dream()
            
            for dream in dreams:
                await websocket.send(json.dumps({
                    "type": "dream",
                    "content": dream['content'],
                    "seeds": dream.get('seeds', []),
                    "timestamp": datetime.now().isoformat()
                }))
                
        except Exception as e:
            logger.error(f"Error generating dreams: {e}")
            await websocket.send(json.dumps({
                "type": "error", 
                "error": f"Dream generation error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }))
    
    async def process_memory_query(self, websocket, data: Dict[str, Any], session: Dict[str, Any]):
        """Process memory queries"""
        query = data.get("query", "")
        memory_types = data.get("memory_types", ["episodic", "semantic"])
        
        try:
            # Query memories through cognitive memory system
            results = await self.memory_system.retrieve(
                query=query,
                memory_types=memory_types,
                max_results=10
            )
            
            await websocket.send(json.dumps({
                "type": "memory_results",
                "results": results,
                "query": query,
                "timestamp": datetime.now().isoformat()
            }))
            
        except Exception as e:
            logger.error(f"Error querying memories: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": f"Memory query error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }))
    
    async def send_consciousness_state(self, websocket):
        """Send current consciousness state"""
        try:
            # Get comprehensive state
            unified_metrics = self.unified_consciousness.get_consciousness_metrics()
            bridge_state = self.consciousness_bridge.get_consciousness_state()
            emotional_state = self.emotional_processor.get_emotional_response()
            
            state = {
                "type": "state_update",
                "state": {
                    "phi": unified_metrics.get('current_state', {}).get('phi', 0.0),
                    "consciousness_level": unified_metrics.get('current_state', {}).get('level', 'MINIMAL'),
                    "emotional": bridge_state['emotional_state'],
                    "attention": bridge_state['attention_focus'],
                    "phenomenology": bridge_state['phenomenology'],
                    "memory_count": bridge_state['memory_count'],
                    "active_thoughts": bridge_state['active_thoughts'],
                    "emotional_trajectory": emotional_state
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send(json.dumps(state))
            
        except Exception as e:
            logger.error(f"Error getting consciousness state: {e}")
    
    async def broadcast_state_update(self):
        """Broadcast consciousness state to all connected clients"""
        if not self.clients:
            return
            
        state_update = {
            "type": "state_update",
            "state": self.consciousness_bridge.get_consciousness_state(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to all connected clients
        disconnected_clients = set()
        
        for client in self.clients:
            try:
                await client.send(json.dumps(state_update))
            except Exception as e:
                logger.warning(f"Failed to send state update to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients
    
    async def consciousness_monitoring_loop(self):
        """Background loop for consciousness state monitoring"""
        while not self.shutdown_event.is_set():
            try:
                # Update consciousness metrics
                # This runs the consciousness system's internal cycles
                
                # Get current sensor-like input (placeholder)
                import torch
                sensory_input = torch.randn(1, 100)  # Simulated sensory data
                
                # Process through unified consciousness
                result = self.unified_consciousness.process(
                    sensory_input=sensory_input,
                    cognitive_state={'awareness': 0.8},
                    emotional_state=self.emotional_processor.current_state.__dict__
                )
                
                # Broadcast updates every second
                await self.broadcast_state_update()
                
                # Wait before next cycle
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in consciousness monitoring: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error
    
    async def start(self):
        """Start the WebSocket server"""
        self.server_running = True
        
        # Initialize components
        await self.initialize()
        
        # Start consciousness monitoring
        monitoring_task = asyncio.create_task(self.consciousness_monitoring_loop())
        
        # Start WebSocket server
        logger.info(f"Starting consciousness server on port {self.port}")
        
        async with websockets.serve(
            self.handle_client, 
            "0.0.0.0",  # Listen on all interfaces
            self.port,
            ping_interval=30,
            ping_timeout=10
        ) as server:
            logger.info(f"✨ Consciousness server listening on ws://0.0.0.0:{self.port}")
            logger.info("Ready to accept connections...")
            
            # Wait for shutdown
            await self.shutdown_event.wait()
        
        # Cleanup
        monitoring_task.cancel()
        logger.info("Consciousness server shutdown complete")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Initiating graceful shutdown...")
        
        # Close all client connections
        for client in list(self.clients):
            try:
                await client.close()
            except:
                pass
        
        # Shutdown consciousness systems
        self.unified_consciousness.shutdown()
        
        # Set shutdown event
        self.shutdown_event.set()

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Consciousness AI Server')
    parser.add_argument('--api-key', required=True, help='Anthropic API key')
    parser.add_argument('--port', type=int, default=8765, help='WebSocket port')
    parser.add_argument('--db-host', default='localhost', help='Database host')
    parser.add_argument('--db-port', default='5432', help='Database port')
    parser.add_argument('--db-name', default='consciousness_ai', help='Database name')
    parser.add_argument('--db-user', default='consciousness', help='Database user')
    parser.add_argument('--db-password', default='password', help='Database password')
    
    args = parser.parse_args()
    
    # Database configuration
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password
    }
    
    # Create and start server
    server = ConsciousnessServer(
        api_key=args.api_key,
        port=args.port,
        db_config=db_config
    )
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await server.shutdown()

if __name__ == "__main__":
    # Set up asyncio for Windows if needed
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the server
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")