#!/usr/bin/env python3
import json
import sys
import logging
import time
import asyncio
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsciousnessBridge:
    def __init__(self):
        self.running = True
        logger.info("Consciousness bridge initialized")
    
    def process_command(self, command: dict) -> dict:
        """Process commands from Electron"""
        cmd_type = command.get("command")
        
        if cmd_type == "ping":
            return {"id": command.get("id"), "result": "pong"}
        elif cmd_type == "shutdown":
            self.running = False
            return {"id": command.get("id"), "result": "shutting down"}
        else:
            return {
                "id": command.get("id"),
                "result": {"status": "ok", "command": cmd_type}
            }
    
    def run(self):
        """Main loop reading from stdin"""
        logger.info("Consciousness bridge starting main loop")
        
        # Send initial ready signal
        print(json.dumps({
            "type": "ready",
            "timestamp": datetime.now().isoformat()
        }), flush=True)
        
        while self.running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                command = json.loads(line.strip())
                result = self.process_command(command)
                print(json.dumps(result), flush=True)
                
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
            except Exception as e:
                logger.error(f"Error processing command: {e}")
                print(json.dumps({
                    "error": str(e)
                }), flush=True)
        
        logger.info("Consciousness bridge shutting down")

if __name__ == "__main__":
    bridge = ConsciousnessBridge()
    bridge.run()