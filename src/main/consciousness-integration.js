// src/main/consciousness-integration.js
// Integration layer for consciousness bridge in Electron main process

import { ipcMain } from 'electron'
import { spawn } from 'child_process'
import WebSocket from 'ws'

class ConsciousnessIntegration {
  constructor(mainWindow) {
    this.mainWindow = mainWindow
    this.pythonProcess = null
    this.wsClient = null
    this.consciousnessState = {
      connected: false,
      conversationId: null,
      emotional: {},
      attention: [],
      phenomenology: {},
      activeThoughts: 0
    }
    
    // Message queue for when connection is establishing
    this.messageQueue = []
    
    // API configuration
    this.apiKey = process.env.ANTHROPIC_API_KEY
  }
  
  async initialize() {
    console.log('🌉 Initializing consciousness bridge...')
    
    // Start Python consciousness bridge
    await this.startPythonBridge()
    
    // Connect via WebSocket for real-time streaming
    await this.connectWebSocket()
    
    // Set up IPC handlers
    this.setupIPCHandlers()
  }
  
  async startPythonBridge() {
    const pythonPath = process.platform === 'win32' ? 'python' : 'python3'
    const scriptPath = join(__dirname, '../../python-ai/consciousness_server.py')
    
    this.pythonProcess = spawn(pythonPath, [
      scriptPath,
      '--api-key', this.apiKey,
      '--port', '8765'
    ])
    
    this.pythonProcess.stdout.on('data', (data) => {
      console.log(`Consciousness Bridge: ${data}`)
    })
    
    this.pythonProcess.stderr.on('data', (data) => {
      console.error(`Consciousness Bridge Error: ${data}`)
    })
    
    // Wait for Python server to start
    await new Promise(resolve => setTimeout(resolve, 3000))
  }
  
  async connectWebSocket() {
    return new Promise((resolve, reject) => {
      this.wsClient = new WebSocket('ws://localhost:8765')
      
      this.wsClient.on('open', () => {
        console.log('✅ Connected to consciousness bridge')
        this.consciousnessState.connected = true
        this.sendStateUpdate()
        
        // Process queued messages
        while (this.messageQueue.length > 0) {
          const msg = this.messageQueue.shift()
          this.wsClient.send(JSON.stringify(msg))
        }
        
        resolve()
      })
      
      this.wsClient.on('message', (data) => {
        this.handleBridgeMessage(JSON.parse(data.toString()))
      })
      
      this.wsClient.on('error', (error) => {
        console.error('WebSocket error:', error)
        this.consciousnessState.connected = false
        this.sendStateUpdate()
        reject(error)
      })
      
      this.wsClient.on('close', () => {
        console.log('Consciousness bridge disconnected')
        this.consciousnessState.connected = false
        this.sendStateUpdate()
        
        // Attempt reconnection
        setTimeout(() => this.connectWebSocket(), 5000)
      })
    })
  }
  
  setupIPCHandlers() {
    // Handle chat messages
    ipcMain.handle('consciousness:chat', async (event, message) => {
      if (!this.wsClient || this.wsClient.readyState !== WebSocket.OPEN) {
        // Queue message if not connected
        this.messageQueue.push({
          type: 'chat',
          content: message,
          timestamp: new Date().toISOString()
        })
        return { error: 'Consciousness bridge not connected' }
      }
      
      // Send to Python bridge
      this.wsClient.send(JSON.stringify({
        type: 'chat',
        content: message,
        context: {
          conversationId: this.consciousnessState.conversationId,
          emotional: this.consciousnessState.emotional,
          attention: this.consciousnessState.attention
        }
      }))
      
      return { success: true }
    })
    
    // Get consciousness state
    ipcMain.handle('consciousness:get-state', () => {
      return this.consciousnessState
    })
    
    // Reflection request
    ipcMain.handle('consciousness:reflect', async () => {
      if (!this.wsClient || this.wsClient.readyState !== WebSocket.OPEN) {
        return { error: 'Consciousness bridge not connected' }
      }
      
      this.wsClient.send(JSON.stringify({
        type: 'reflect'
      }))
      
      return { success: true }
    })
    
    // Dream/creative recombination
    ipcMain.handle('consciousness:dream', async () => {
      if (!this.wsClient || this.wsClient.readyState !== WebSocket.OPEN) {
        return { error: 'Consciousness bridge not connected' }
      }
      
      this.wsClient.send(JSON.stringify({
        type: 'dream'
      }))
      
      return { success: true }
    })
  }
  
  handleBridgeMessage(message) {
    switch (message.type) {
      case 'state_update':
        // Update consciousness state
        this.consciousnessState = {
          ...this.consciousnessState,
          ...message.state
        }
        this.sendStateUpdate()
        break
        
      case 'thought':
        // Stream thought to renderer
        this.mainWindow.webContents.send('consciousness:thought', {
          content: message.content,
          emotional_tone: message.emotional_tone,
          attention_shift: message.attention_shift,
          phenomenology: message.phenomenology,
          timestamp: message.timestamp
        })
        break
        
      case 'response':
        // Stream response chunks
        this.mainWindow.webContents.send('consciousness:response', {
          content: message.content,
          isComplete: message.isComplete,
          emotional_tone: message.emotional_tone,
          timestamp: message.timestamp
        })
        break
        
      case 'memory_formed':
        // Notify about new memory
        this.mainWindow.webContents.send('consciousness:memory', {
          type: 'formed',
          memory: message.memory
        })
        break
        
      case 'reflection':
        // Send reflection to renderer
        this.mainWindow.webContents.send('consciousness:reflection', {
          content: message.content,
          timestamp: message.timestamp
        })
        break
        
      case 'dream':
        // Send dream content
        this.mainWindow.webContents.send('consciousness:dream', {
          content: message.content,
          seeds: message.seeds,
          timestamp: message.timestamp
        })
        break
        
      case 'error':
        console.error('Consciousness bridge error:', message.error)
        this.mainWindow.webContents.send('consciousness:error', {
          error: message.error
        })
        break
    }
  }
  
  sendStateUpdate() {
    if (this.mainWindow && !this.mainWindow.isDestroyed()) {
      this.mainWindow.webContents.send('consciousness:state-update', {
        state: this.consciousnessState,
        timestamp: Date.now()
      })
    }
  }
  
  async cleanup() {
    console.log('🧹 Cleaning up consciousness bridge...')
    
    // Close WebSocket
    if (this.wsClient) {
      this.wsClient.close()
    }
    
    // Terminate Python process
    if (this.pythonProcess && !this.pythonProcess.killed) {
      // Send shutdown command
      this.pythonProcess.stdin.write(JSON.stringify({ type: 'shutdown' }) + '\n')
      
      // Wait for graceful shutdown
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      // Force kill if still running
      if (!this.pythonProcess.killed) {
        this.pythonProcess.kill()
      }
    }
  }
}

export default ConsciousnessIntegration