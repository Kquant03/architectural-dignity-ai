// src/main/websocket-server.js
// WebSocket server for real-time consciousness updates

import { WebSocketServer } from 'ws'

export class ConsciousnessWebSocketServer {
  constructor(port = 8080) {
    this.port = port
    this.wss = null
    this.clients = new Set()
  }

  start() {
    this.wss = new WebSocketServer({ port: this.port })
    
    this.wss.on('connection', (ws) => {
      console.log('🔌 WebSocket client connected')
      this.clients.add(ws)
      
      ws.on('close', () => {
        console.log('🔌 WebSocket client disconnected')
        this.clients.delete(ws)
      })
      
      ws.on('error', (error) => {
        console.error('WebSocket error:', error)
        this.clients.delete(ws)
      })
    })
    
    console.log(`🌐 WebSocket server listening on port ${this.port}`)
  }
  
  broadcast(data) {
    const message = JSON.stringify(data)
    
    this.clients.forEach((client) => {
      if (client.readyState === 1) { // OPEN
        client.send(message)
      }
    })
  }
  
  stop() {
    this.clients.clear()
    if (this.wss) {
      this.wss.close()
    }
  }
}