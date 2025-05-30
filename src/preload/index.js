// src/preload/index.js
// Secure bridge between consciousness engine and UI - Updated for chat interface

import { contextBridge, ipcRenderer } from 'electron'

// Expose protected consciousness API to renderer
contextBridge.exposeInMainWorld('ConsciousnessAPI', {
  // Core consciousness operations
  consciousness: {
    // Real-time state subscription
    onStateUpdate: (callback) => {
      const subscription = (event, data) => callback(data)
      ipcRenderer.on('consciousness:state-update', subscription)
      
      // Return unsubscribe function
      return () => {
        ipcRenderer.removeListener('consciousness:state-update', subscription)
      }
    },
    
    // Get initial state
    onInitialState: (callback) => {
      ipcRenderer.once('consciousness:initial-state', (event, data) => callback(data))
    },
    
    // Chat interaction
    chat: (message) => {
      return ipcRenderer.invoke('consciousness:chat', message)
    },
    
    // Request reflection
    requestReflection: () => {
      return ipcRenderer.invoke('consciousness:reflect')
    },
    
    // Request dream
    requestDream: () => {
      return ipcRenderer.invoke('consciousness:dream')
    },
    
    // Thought stream listener
    onThought: (callback) => {
      ipcRenderer.on('consciousness:thought', (event, data) => callback(event, data))
    },
    
    // Response stream listener
    onResponse: (callback) => {
      ipcRenderer.on('consciousness:response', (event, data) => callback(event, data))
    },
    
    // Reflection listener
    onReflection: (callback) => {
      ipcRenderer.on('consciousness:reflection', (event, data) => callback(event, data))
    },
    
    // Dream listener
    onDream: (callback) => {
      ipcRenderer.on('consciousness:dream', (event, data) => callback(event, data))
    },
    
    // Error listener
    onError: (callback) => {
      ipcRenderer.on('consciousness:error', (event, data) => callback(event, data))
    },
    
    // Remove listeners
    removeThoughtListener: (callback) => {
      ipcRenderer.removeListener('consciousness:thought', callback)
    },
    
    removeResponseListener: (callback) => {
      ipcRenderer.removeListener('consciousness:response', callback)
    },
    
    removeReflectionListener: (callback) => {
      ipcRenderer.removeListener('consciousness:reflection', callback)
    },
    
    removeDreamListener: (callback) => {
      ipcRenderer.removeListener('consciousness:dream', callback)
    },
    
    removeErrorListener: (callback) => {
      ipcRenderer.removeListener('consciousness:error', callback)
    },
    
    // Send commands to consciousness engine
    sendCommand: (command, params) => {
      return ipcRenderer.invoke('consciousness:command', command, params)
    },
    
    // Get current metrics
    getMetrics: () => {
      return ipcRenderer.invoke('consciousness:get-metrics')
    },
    
    // Export consciousness data
    exportData: () => {
      return ipcRenderer.invoke('consciousness:export')
    }
  },
  
  // Memory operations
  memory: {
    // Store a memory with emotional context
    store: (memory) => {
      return ipcRenderer.invoke('consciousness:command', 'store_memory', memory)
    },
    
    // Recall memories by query
    recall: (query) => {
      return ipcRenderer.invoke('consciousness:command', 'recall_memory', { query })
    },
    
    // Get memory statistics
    getStats: () => {
      return ipcRenderer.invoke('consciousness:command', 'memory_stats', {})
    }
  },
  
  // Emotional processing
  emotion: {
    // Process emotional stimulus
    process: (stimulus) => {
      return ipcRenderer.invoke('consciousness:command', 'process_emotion', stimulus)
    },
    
    // Get emotional history
    getHistory: (timeRange) => {
      return ipcRenderer.invoke('consciousness:command', 'emotion_history', { timeRange })
    }
  },
  
  // Attention management
  attention: {
    // Adjust attention focus
    setFocus: (focus) => {
      return ipcRenderer.invoke('consciousness:command', 'adjust_attention', { focus })
    },
    
    // Get attention patterns
    getPatterns: () => {
      return ipcRenderer.invoke('consciousness:command', 'attention_patterns', {})
    }
  },
  
  // GPU configuration
  gpu: {
    // Configure GPU settings
    configure: (config) => {
      return ipcRenderer.invoke('gpu:configure', config)
    },
    
    // Get GPU stats
    getStats: () => {
      return ipcRenderer.invoke('gpu:stats')
    }
  },
  
  // System utilities
  system: {
    // Get app version
    getVersion: () => {
      return process.version
    },
    
    // Platform info
    getPlatform: () => {
      return {
        platform: process.platform,
        arch: process.arch,
        versions: process.versions
      }
    }
  }
})

console.log('🌉 Consciousness bridge initialized')