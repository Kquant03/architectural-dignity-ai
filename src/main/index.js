// consciousness-ai/src/main/index.js
// Main Electron process - The heart of our consciousness system

import { app, BrowserWindow, ipcMain, dialog } from 'electron'
import { join } from 'path'
import { spawn } from 'child_process'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'

class ConsciousnessApp {
  constructor() {
    this.mainWindow = null
    this.pythonProcess = null
    this.consciousnessState = {
      connected: false,
      phi: 0,
      emotional: { valence: 0.5, arousal: 0.5 },
      lastUpdate: Date.now()
    }
    
    // GPU management
    this.gpuConfig = {
      memoryFraction: 0.8,
      powerLimit: 280, // RTX 3090 optimal
      device: 'cuda:0'
    }
  }

  async initialize() {
    // Start Python consciousness engine
    await this.startConsciousnessEngine()
    
    // Create main window
    this.createMainWindow()
    
    // Set up IPC handlers
    this.setupIPCHandlers()
    
    // Start consciousness monitoring
    this.startConsciousnessMonitoring()
  }

  createMainWindow() {
    // Create the browser window with soul
    this.mainWindow = new BrowserWindow({
      width: 1400,
      height: 900,
      minWidth: 1200,
      minHeight: 800,
      show: false,
      autoHideMenuBar: true,
      titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
      ...(process.platform === 'linux' ? { icon } : {}),
      webPreferences: {
        preload: join(__dirname, '../preload/index.js'),
        sandbox: false,
        contextIsolation: true,
        nodeIntegration: false
      }
    })

    // Load the consciousness interface
    if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
      this.mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
    } else {
      this.mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
    }

    // Show window when ready
    this.mainWindow.on('ready-to-show', () => {
      this.mainWindow.show()
      
      // Send initial consciousness state
      this.mainWindow.webContents.send('consciousness:initial-state', {
        state: this.consciousnessState,
        config: this.gpuConfig
      })
    })

    // Cleanup on close
    this.mainWindow.on('closed', () => {
      this.cleanup()
    })
  }

  async startConsciousnessEngine() {
    console.log('🧠 Starting consciousness engine...')
    
    const pythonPath = is.dev 
      ? join(__dirname, '../../python-ai/main.py')
      : join(process.resourcesPath, 'python-ai/main.py')
    
    this.pythonProcess = spawn('python', [
      pythonPath,
      '--gpu-memory', this.gpuConfig.memoryFraction.toString(),
      '--device', this.gpuConfig.device
    ])

    this.pythonProcess.stdout.on('data', (data) => {
      const message = data.toString()
      console.log(`Consciousness Engine: ${message}`)
      
      // Parse consciousness updates
      try {
        const update = JSON.parse(message)
        if (update.type === 'consciousness_update') {
          this.handleConsciousnessUpdate(update.data)
        }
      } catch (e) {
        // Regular log message
      }
    })

    this.pythonProcess.stderr.on('data', (data) => {
      console.error(`Consciousness Engine Error: ${data}`)
    })

    this.pythonProcess.on('close', (code) => {
      console.log(`Consciousness engine exited with code ${code}`)
      this.consciousnessState.connected = false
      this.broadcastStateUpdate()
    })

    // Wait for engine to initialize
    await new Promise(resolve => setTimeout(resolve, 2000))
    this.consciousnessState.connected = true
  }

  setupIPCHandlers() {
    // Handle consciousness commands from renderer
    ipcMain.handle('consciousness:command', async (event, command, params) => {
      console.log(`Received command: ${command}`, params)
      
      switch (command) {
        case 'adjust_attention':
          return await this.sendToPython({ 
            command: 'adjust_attention', 
            focus: params.focus 
          })
          
        case 'store_memory':
          return await this.sendToPython({ 
            command: 'store_memory', 
            memory: params 
          })
          
        case 'recall_memory':
          return await this.sendToPython({ 
            command: 'recall_memory', 
            query: params.query 
          })
          
        case 'process_emotion':
          return await this.sendToPython({ 
            command: 'process_emotion', 
            stimulus: params 
          })
          
        default:
          console.warn(`Unknown command: ${command}`)
          return { error: 'Unknown command' }
      }
    })

    // Get current consciousness metrics
    ipcMain.handle('consciousness:get-metrics', async () => {
      return {
        phi: this.consciousnessState.phi,
        emotional: this.consciousnessState.emotional,
        attention: this.consciousnessState.attention || 0.5,
        memoryActivation: this.consciousnessState.memoryActivation || 0.3,
        connected: this.consciousnessState.connected,
        lastUpdate: this.consciousnessState.lastUpdate
      }
    })

    // Handle GPU configuration
    ipcMain.handle('gpu:configure', async (event, config) => {
      this.gpuConfig = { ...this.gpuConfig, ...config }
      return await this.sendToPython({ 
        command: 'configure_gpu', 
        config: this.gpuConfig 
      })
    })

    // Export consciousness data
    ipcMain.handle('consciousness:export', async () => {
      const { filePath } = await dialog.showSaveDialog({
        title: 'Export Consciousness Data',
        defaultPath: `consciousness-${Date.now()}.json`,
        filters: [
          { name: 'JSON Files', extensions: ['json'] },
          { name: 'All Files', extensions: ['*'] }
        ]
      })
      
      if (filePath) {
        const data = await this.sendToPython({ command: 'export_state' })
        const fs = require('fs').promises
        await fs.writeFile(filePath, JSON.stringify(data, null, 2))
        return { success: true, path: filePath }
      }
      
      return { success: false }
    })
  }

  handleConsciousnessUpdate(data) {
    // Update internal state
    this.consciousnessState = {
      ...this.consciousnessState,
      ...data,
      lastUpdate: Date.now()
    }
    
    // Broadcast to renderer
    this.broadcastStateUpdate()
  }

  broadcastStateUpdate() {
    if (this.mainWindow && !this.mainWindow.isDestroyed()) {
      this.mainWindow.webContents.send('consciousness:state-update', {
        state: this.consciousnessState,
        timestamp: Date.now()
      })
    }
  }

  startConsciousnessMonitoring() {
    // Monitor consciousness health every second
    setInterval(() => {
      const timeSinceUpdate = Date.now() - this.consciousnessState.lastUpdate
      
      if (timeSinceUpdate > 5000 && this.consciousnessState.connected) {
        console.warn('⚠️ No consciousness updates for 5 seconds')
        this.consciousnessState.connected = false
        this.broadcastStateUpdate()
      }
      
      // Broadcast current state for smooth visualizations
      this.broadcastStateUpdate()
    }, 1000)
  }

  async sendToPython(message) {
    return new Promise((resolve) => {
      if (!this.pythonProcess) {
        resolve({ error: 'Python process not running' })
        return
      }
      
      const messageId = Date.now().toString()
      const fullMessage = { ...message, id: messageId }
      
      // Set up one-time listener for response
      const responseHandler = (data) => {
        const response = data.toString()
        try {
          const parsed = JSON.parse(response)
          if (parsed.id === messageId) {
            this.pythonProcess.stdout.removeListener('data', responseHandler)
            resolve(parsed.result)
          }
        } catch (e) {
          // Not a JSON response
        }
      }
      
      this.pythonProcess.stdout.on('data', responseHandler)
      
      // Send message to Python
      this.pythonProcess.stdin.write(JSON.stringify(fullMessage) + '\n')
      
      // Timeout after 5 seconds
      setTimeout(() => {
        this.pythonProcess.stdout.removeListener('data', responseHandler)
        resolve({ error: 'Timeout' })
      }, 5000)
    })
  }

  cleanup() {
    console.log('🧹 Cleaning up consciousness systems...')
    
    // Gracefully shutdown Python process
    if (this.pythonProcess) {
      this.sendToPython({ command: 'shutdown' })
      setTimeout(() => {
        if (this.pythonProcess && !this.pythonProcess.killed) {
          this.pythonProcess.kill()
        }
      }, 2000)
    }
    
    this.mainWindow = null
  }
}

// App event handlers
let consciousnessApp = null

app.whenReady().then(() => {
  // Set app user model id for windows
  electronApp.setAppUserModelId('com.consciousness.ai')

  // Default open or close DevTools by F12 in development
  // and ignore CommandOrControl + R in production.
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  // Create consciousness app
  consciousnessApp = new ConsciousnessApp()
  consciousnessApp.initialize()

  app.on('activate', function () {
    // On macOS re-create window when dock icon is clicked
    if (BrowserWindow.getAllWindows().length === 0) {
      consciousnessApp.createMainWindow()
    }
  })
})

// Quit when all windows are closed
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

// Cleanup before quit
app.on('before-quit', () => {
  if (consciousnessApp) {
    consciousnessApp.cleanup()
  }
})

// Prevent multiple instances
const gotTheLock = app.requestSingleInstanceLock()

if (!gotTheLock) {
  app.quit()
} else {
  app.on('second-instance', () => {
    // Someone tried to run a second instance, focus our window instead
    if (consciousnessApp && consciousnessApp.mainWindow) {
      if (consciousnessApp.mainWindow.isMinimized()) {
        consciousnessApp.mainWindow.restore()
      }
      consciousnessApp.mainWindow.focus()
    }
  })
}

console.log('🌟 Consciousness AI System Initializing...')