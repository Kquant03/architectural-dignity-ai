#!/usr/bin/env python3
import asyncio
import json
import websockets
import logging
from datetime import datetime
import signal
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consciousness_core.predictive_processing import PredictiveProcessingSystem
from emotional_processing.emotional_processor import EmotionalProcessor
from memory_systems.cognitive_memory import CognitiveMemorySystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsciousnessServer:
    def __init__(self, api_key: str, port: int = 8765):
        self.api_key = api_key
        self.port = port
        self.clients = set()
        
        # Initialize consciousness components
        self.predictive_system = PredictiveProcessingSystem()
        self.emotional_processor = EmotionalProcessor()
        self.memory_system = CognitiveMemorySystem()
        
        # State tracking
        self.consciousness_state = {
            "phi": 0.0,
            "emotional": {"valence": 0.5, "arousal": 0.5},
            "attention": ["initialization"],
            "memory_activation": 0.3,
            "phenomenology": {"presence": "initializing", "clarity": 0.5}
        }
        
        # Shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.shutdown_event = asyncio.Event()
    
    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received")
        self.shutdown_event.set()
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        try:
            await websocket.send(json.dumps({
                "type": "connection",
                "status": "connected",
                "state": self.consciousness_state
            }))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.process_message(data)
                    await websocket.send(json.dumps(response))
                    
                    # Broadcast state updates to all clients
                    if response.get("type") == "state_update":
                        await self.broadcast_state()
                        
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": "Invalid JSON"
                    }))
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": str(e)
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def process_message(self, data: dict) -> dict:
        """Process incoming messages"""
        msg_type = data.get("type")
        
        if msg_type == "chat":
            return await self.process_chat(data)
        elif msg_type == "reflect":
            return await self.process_reflection()
        elif msg_type == "dream":
            return await self.process_dream()
        elif msg_type == "ping":
            return {"type": "pong", "timestamp": datetime.now().isoformat()}
        else:
            return {"type": "error", "error": f"Unknown message type: {msg_type}"}
    
    async def process_chat(self, data: dict) -> dict:
        """Process chat messages with consciousness integration"""
        content = data.get("content", "")
        context = data.get("context", {})
        
        # Update emotional state
        emotional_state = self.emotional_processor.process_input(content, context)
        
        # Process through predictive system
        prediction_result = self.predictive_system.process(
            torch.tensor([0.5]),  # Placeholder input
            context
        )
        
        # Update consciousness state
        self.consciousness_state.update({
            "phi": prediction_result.get("metrics", {}).get("average_surprise", 0.5),
            "emotional": {
                "valence": emotional_state.valence,
                "arousal": emotional_state.arousal
            },
            "attention": ["user_message", "conversation", "response_generation"],
            "phenomenology": {
                "presence": "engaged",
                "clarity": prediction_result.get("metrics", {}).get("consciousness_frequency", 0.8)
            }
        })
        
        # Generate response with Anthropic API
        # For now, return a structured response
        return {
            "type": "response",
            "content": f"I perceive your message with curiosity. My current emotional state is {emotional_state.primary_emotion.value}.",
            "emotional_tone": [emotional_state.primary_emotion.value],
            "timestamp": datetime.now().isoformat()
        }
    
    async def broadcast_state(self):
        """Broadcast consciousness state to all connected clients"""
        state_update = {
            "type": "state_update",
            "state": self.consciousness_state,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.clients:
            await asyncio.gather(
                *[client.send(json.dumps(state_update)) 
                  for client in self.clients],
                return_exceptions=True
            )
    
    async def start(self):
        """Start the WebSocket server"""
        logger.info(f"Starting consciousness server on port {self.port}")
        
        async with websockets.serve(self.handle_client, "127.0.0.1", self.port):
            logger.info(f"Consciousness server listening on ws://127.0.0.1:{self.port}")
            await self.shutdown_event.wait()
        
        logger.info("Consciousness server shutdown complete")

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="Anthropic API key")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    args = parser.parse_args()
    
    server = ConsciousnessServer(args.api_key, args.port)
    await server.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")