// src/main/python-bridge.js
// Enhanced Python-Electron communication bridge

import { spawn } from 'child_process'
import { EventEmitter } from 'events'
import { join } from 'path'
import { is } from '@electron-toolkit/utils'

export class PythonBridge extends EventEmitter {
  constructor() {
    super()
    this.pythonProcess = null
    this.pendingRequests = new Map()
    this.messageBuffer = ''
    this.isConnected = false
    this.reconnectAttempts = 0
    this.maxReconnectAttempts = 5
  }

  async start(gpuConfig) {
    try {
      const pythonPath = is.dev 
        ? join(__dirname, '../../python-ai/main.py')
        : join(process.resourcesPath, 'python-ai/main.py')
      
      console.log('🐍 Starting Python process at:', pythonPath)
      
      // Check if Python is available
      const pythonExecutable = process.platform === 'win32' ? 'python' : 'python3'
      
      this.pythonProcess = spawn(pythonExecutable, [
        pythonPath,
        '--gpu-memory', gpuConfig.memoryFraction.toString(),
        '--device', gpuConfig.device
      ], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONUNBUFFERED: '1' }
      })

      this.pythonProcess.on('error', (error) => {
        console.error('❌ Failed to start Python process:', error)
        this.emit('error', { type: 'start_failed', error: error.message })
      })

      this.pythonProcess.stdout.on('data', (chunk) => {
        this.handleData(chunk)
      })

      this.pythonProcess.stderr.on('data', (data) => {
        const message = data.toString()
        console.error('Python Error:', message)
        this.emit('error', { type: 'runtime', message })
      })

      this.pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`)
        this.isConnected = false
        this.emit('disconnected', code)
        this.handleDisconnection()
      })

      // Wait for initial connection
      await this.waitForConnection()
      this.isConnected = true
      this.reconnectAttempts = 0
      this.emit('connected')
      
      return true
    } catch (error) {
      console.error('Failed to start Python bridge:', error)
      throw error
    }
  }

  handleData(chunk) {
    this.messageBuffer += chunk.toString()
    
    // Process complete messages (newline-delimited JSON)
    let newlineIndex
    while ((newlineIndex = this.messageBuffer.indexOf('\n')) !== -1) {
      const message = this.messageBuffer.slice(0, newlineIndex)
      this.messageBuffer = this.messageBuffer.slice(newlineIndex + 1)
      
      if (message.trim()) {
        try {
          const parsed = JSON.parse(message)
          this.handleMessage(parsed)
        } catch (error) {
          console.error('Failed to parse message:', message, error)
        }
      }
    }
  }

  handleMessage(message) {
    // Handle response to a request
    if (message.id && this.pendingRequests.has(message.id)) {
      const { resolve, reject } = this.pendingRequests.get(message.id)
      this.pendingRequests.delete(message.id)
      
      if (message.error) {
        reject(new Error(message.error))
      } else {
        resolve(message.result)
      }
    } 
    // Handle consciousness updates
    else if (message.type === 'consciousness_update') {
      this.emit('consciousness_update', message.data)
    }
    // Handle other message types
    else {
      this.emit('message', message)
    }
  }

  async send(command, params = {}) {
    if (!this.pythonProcess || !this.isConnected) {
      throw new Error('Python process not connected')
    }

    const id = Date.now().toString() + Math.random().toString(36).substr(2, 9)
    const message = { id, command, ...params }

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject })
      
      try {
        this.pythonProcess.stdin.write(JSON.stringify(message) + '\n')
      } catch (error) {
        this.pendingRequests.delete(id)
        reject(error)
        return
      }

      // Timeout after 5 seconds
      setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id)
          reject(new Error('Request timeout'))
        }
      }, 5000)
    })
  }

  async waitForConnection(timeout = 5000) {
    return new Promise((resolve, reject) => {
      const startTime = Date.now()
      let attempts = 0
      
      const checkConnection = () => {
        attempts++
        
        if (!this.pythonProcess || this.pythonProcess.killed) {
          reject(new Error('Python process not running'))
          return
        }
        
        // Give Python time to start up on first attempt
        if (attempts === 1) {
          setTimeout(checkConnection, 500)
          return
        }
        
        // Try to send a ping
        this.send('ping', {})
          .then(() => {
            console.log('✅ Python process responded to ping')
            resolve()
          })
          .catch((error) => {
            if (Date.now() - startTime < timeout) {
              setTimeout(checkConnection, 200)
            } else {
              reject(new Error(`Connection timeout after ${attempts} attempts`))
            }
          })
      }
      
      // Start checking after a brief delay
      setTimeout(checkConnection, 100)
    })
  }

  async handleDisconnection() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      console.log(`🔄 Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`)
      
      setTimeout(() => {
        this.start(this.lastGpuConfig).catch(error => {
          console.error('Reconnection failed:', error)
        })
      }, 2000 * this.reconnectAttempts)
    }
  }

  async shutdown() {
    if (this.pythonProcess) {
      try {
        await this.send('shutdown', {})
      } catch (error) {
        // Ignore errors during shutdown
      }
      
      setTimeout(() => {
        if (this.pythonProcess && !this.pythonProcess.killed) {
          this.pythonProcess.kill()
        }
      }, 1000)
    }
    
    this.removeAllListeners()
    this.pendingRequests.clear()
  }
}