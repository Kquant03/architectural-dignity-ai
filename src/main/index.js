// consciousness-ai/src/main/index.js
// Main Electron process - Enhanced with robust Python bridge

import { app, BrowserWindow, ipcMain, dialog } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { PythonBridge } from './python-bridge'
import icon from '../../resources/icon.png?asset'
import ConsciousnessIntegration from './consciousness-integration.js'

class ConsciousnessApp {
  constructor() {
    this.mainWindow = null
    this.consciousnessIntegration = null
    this.pythonBridge = new PythonBridge()
    this.consciousnessState = {
      connected: false,
      phi: 0,
      emotional: { valence: 0.5, arousal: 0.5 },
      attention: 0.5,
      memoryActivation: 0.3,
      lastUpdate: Date.now()
    }
    
    // GPU management
    this.gpuConfig = {
      memoryFraction: 0.8,
      powerLimit: 280, // RTX 3090 optimal
      device: 'cuda:0'
    }
    
    // Setup Python bridge listeners
    this.setupPythonListeners()
  }

  setupPythonListeners() {
    this.pythonBridge.on('connected', () => {
      console.log('✅ Python consciousness engine connected')
      this.consciousnessState.connected = true
      this.broadcastStateUpdate()
    })
    
    this.pythonBridge.on('disconnected', (code) => {
      console.log('❌ Python consciousness engine disconnected:', code)
      this.consciousnessState.connected = false
      this.broadcastStateUpdate()
    })
    
    this.pythonBridge.on('consciousness_update', (data) => {
      this.handleConsciousnessUpdate(data)
    })
    
    this.pythonBridge.on('error', (error) => {
      console.error('Python bridge error:', error)
      if (this.mainWindow && !this.mainWindow.isDestroyed()) {
        this.mainWindow.webContents.send('consciousness:error', error)
      }
    })
  }

  async initialize() {
    try {
      // Create main window first
      this.createMainWindow()
      
      // Set up IPC handlers
      this.setupIPCHandlers()
      
      // Start Python consciousness engine
      await this.startConsciousnessEngine()

      // Initialize consciousness integration
      this.consciousnessIntegration = new ConsciousnessIntegration(this.mainWindow)
      await this.consciousnessIntegration.initialize()
      
      // Start consciousness monitoring
      this.startConsciousnessMonitoring()
    } catch (error) {
      console.error('Failed to initialize consciousness app:', error)
      dialog.showErrorBox('Initialization Error', 
        `Failed to start consciousness engine: ${error.message}`)
    }
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
      backgroundColor: '#000000',
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

    // Handle window closed
    this.mainWindow.on('closed', () => {
      this.mainWindow = null
    })
  }

  async startConsciousnessEngine() {
    console.log('🧠 Starting consciousness engine...')
    
    try {
      await this.pythonBridge.start(this.gpuConfig)
      console.log('✨ Consciousness engine started successfully')
    } catch (error) {
      console.error('Failed to start consciousness engine:', error)
      
      // Show user-friendly error
      if (error.message.includes('Python') || error.message.includes('python')) {
        dialog.showErrorBox('Python Not Found', 
          'Please ensure Python 3 is installed and available in your PATH.\n\n' +
          'You can install Python from https://www.python.org/downloads/')
      } else {
        dialog.showErrorBox('Consciousness Engine Error', 
          `Failed to start consciousness engine: ${error.message}`)
      }
      
      // Continue with disconnected state
      this.consciousnessState.connected = false
    }
  }

  setupIPCHandlers() {
    // Handle consciousness commands from renderer
    ipcMain.handle('consciousness:command', async (event, command, params) => {
      console.log(`Received command: ${command}`, params)
      
      try {
        if (!this.consciousnessState.connected) {
          throw new Error('Consciousness engine not connected')
        }
        
        return await this.pythonBridge.send(command, params)
      } catch (error) {
        console.error(`Command failed: ${command}`, error)
        return { error: error.message }
      }
    })

    // Get current consciousness metrics
    ipcMain.handle('consciousness:get-metrics', async () => {
      return {
        phi: this.consciousnessState.phi,
        emotional: this.consciousnessState.emotional,
        attention: this.consciousnessState.attention,
        memoryActivation: this.consciousnessState.memoryActivation,
        connected: this.consciousnessState.connected,
        lastUpdate: this.consciousnessState.lastUpdate
      }
    })

    // Handle GPU configuration
    ipcMain.handle('gpu:configure', async (event, config) => {
      this.gpuConfig = { ...this.gpuConfig, ...config }
      
      try {
        return await this.pythonBridge.send('configure_gpu', { config: this.gpuConfig })
      } catch (error) {
        return { error: error.message }
      }
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
        try {
          const data = await this.pythonBridge.send('export_state', {})
          const fs = require('fs').promises
          await fs.writeFile(filePath, JSON.stringify(data, null, 2))
          return { success: true, path: filePath }
        } catch (error) {
          return { success: false, error: error.message }
        }
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
    }, 1000)
  }

  async cleanup() {
    console.log('🧹 Cleaning up consciousness systems...')
    
    // Gracefully shutdown Python process
    await this.pythonBridge.shutdown()
    
    // Clear all listeners
    ipcMain.removeAllListeners()
  }
}

// App instance
let consciousnessApp = null

// Electron app ready
app.whenReady().then(async () => {
  // Set app user model id for windows
  electronApp.setAppUserModelId('com.consciousness.ai')

  // Default open or close DevTools by F12 in development
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  // Create consciousness app
  consciousnessApp = new ConsciousnessApp()
  await consciousnessApp.initialize()

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
app.on('before-quit', async (event) => {
  if (consciousnessApp) {
    event.preventDefault()
    await consciousnessApp.cleanup()
    app.quit()
  }
})

    // Clean up consciousness integration
    if (this.consciousnessIntegration) {
        this.consciousnessIntegration.cleanup()
    }

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