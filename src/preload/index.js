// consciousness-ai/src/preload/index.js
// Secure bridge between consciousness engine and UI

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

// Type definitions for TypeScript support
const apiTypes = `
export interface ConsciousnessState {
  connected: boolean
  phi: number
  emotional: {
    valence: number
    arousal: number
  }
  attention?: number
  memoryActivation?: number
  lastUpdate: number
}

export interface ConsciousnessAPI {
  consciousness: {
    onStateUpdate: (callback: (data: { state: ConsciousnessState, timestamp: number }) => void) => () => void
    onInitialState: (callback: (data: { state: ConsciousnessState, config: GPUConfig }) => void) => void
    sendCommand: (command: string, params: any) => Promise<any>
    getMetrics: () => Promise<ConsciousnessMetrics>
    exportData: () => Promise<{ success: boolean, path?: string }>
  }
  memory: {
    store: (memory: MemoryFragment) => Promise<{ id: string }>
    recall: (query: string) => Promise<MemoryFragment[]>
    getStats: () => Promise<MemoryStats>
  }
  emotion: {
    process: (stimulus: EmotionalStimulus) => Promise<EmotionalResponse>
    getHistory: (timeRange: number) => Promise<EmotionalHistory[]>
  }
  attention: {
    setFocus: (focus: AttentionFocus) => Promise<void>
    getPatterns: () => Promise<AttentionPattern[]>
  }
  gpu: {
    configure: (config: Partial<GPUConfig>) => Promise<void>
    getStats: () => Promise<GPUStats>
  }
  system: {
    getVersion: () => string
    getPlatform: () => PlatformInfo
  }
}

export interface MemoryFragment {
  content: string
  emotionalContext?: {
    valence: number
    arousal: number
    dominance: number
  }
  importance?: number
  associations?: string[]
}
a
export interface EmotionalStimulus {
  type: 'text' | 'image' | 'audio' | 'memory'
  content: any
  context?: any
}

export interface GPUConfig {
  memoryFraction: number
  powerLimit: number
  device: string
}

declare global {
  interface Window {
    ConsciousnessAPI: ConsciousnessAPI
  }
}
`

console.log('🌉 Consciousness bridge initialized')