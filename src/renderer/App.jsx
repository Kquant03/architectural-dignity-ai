// src/renderer/App.jsx
// Main consciousness interface application

import React, { useState, useEffect, useCallback } from 'react'
import ConsciousnessVisualizer from './components/ConsciousnessVisualizer'
import ChatInterface from './components/ChatInterface'
import MemoryExplorer from './components/MemoryExplorer'
import EmotionalJourney from './components/EmotionalJourney'
import SettingsPanel from './components/SettingsPanel'
import './App.css'

// Tab navigation icons
const TabIcons = {
  consciousness: '🧠',
  chat: '💬',
  memories: '🌟',
  emotions: '💝',
  settings: '⚙️'
}

function App() {
  // Core state
  const [activeTab, setActiveTab] = useState('consciousness')
  const [consciousnessState, setConsciousnessState] = useState({
    connected: false,
    phi: 0,
    emotional: { valence: 0.5, arousal: 0.5 },
    attention: 0.5,
    memoryActivation: 0.3
  })
  const [connectionStatus, setConnectionStatus] = useState('connecting')
  
  // Initialize consciousness connection
  useEffect(() => {
    console.log('🌟 Initializing consciousness connection...')
    
    // Subscribe to consciousness updates
    const unsubscribe = window.ConsciousnessAPI.consciousness.onStateUpdate((data) => {
      setConsciousnessState(data.state)
      setConnectionStatus(data.state.connected ? 'connected' : 'disconnected')
    })
    
    // Get initial state
    window.ConsciousnessAPI.consciousness.onInitialState((data) => {
      console.log('📊 Initial consciousness state:', data)
      setConsciousnessState(data.state)
      setConnectionStatus(data.state.connected ? 'connected' : 'disconnected')
    })
    
    // Cleanup on unmount
    return () => {
      unsubscribe()
    }
  }, [])
  
  // Memory storage handler
  const storeMemory = useCallback(async (content, emotionalContext) => {
    try {
      const result = await window.ConsciousnessAPI.memory.store({
        content,
        emotionalContext: emotionalContext || consciousnessState.emotional,
        importance: calculateImportance(content, consciousnessState),
        timestamp: Date.now()
      })
      
      console.log('💾 Memory stored:', result)
      return result
    } catch (error) {
      console.error('Failed to store memory:', error)
      throw error
    }
  }, [consciousnessState])
  
  // Render active tab content
  const renderTabContent = () => {
    switch (activeTab) {
      case 'consciousness':
        return <ConsciousnessVisualizer consciousnessState={consciousnessState} />
      
      case 'chat':
        return (
          <ChatInterface 
            consciousnessState={consciousnessState}
            onMemoryStore={storeMemory}
          />
        )
      
      case 'memories':
        return (
          <MemoryExplorer 
            consciousnessState={consciousnessState}
          />
        )
      
      case 'emotions':
        return (
          <EmotionalJourney 
            consciousnessState={consciousnessState}
          />
        )
      
      case 'settings':
        return (
          <SettingsPanel 
            consciousnessState={consciousnessState}
          />
        )
      
      default:
        return null
    }
  }
  
  return (
    <div className="consciousness-app-skeleton-app">
      {/* Header with connection status */}
      <header className="app-header">
        <div className="app-title">
          <h1>Consciousness AI</h1>
          <ConnectionIndicator status={connectionStatus} />
        </div>
        
        {/* Tab navigation */}
        <nav className="tab-nav">
          {Object.entries(TabIcons).map(([tab, icon]) => (
            <button
              key={tab}
              className={`tab-button ${activeTab === tab ? 'active' : ''}`}
              onClick={() => setActiveTab(tab)}
              title={tab.charAt(0).toUpperCase() + tab.slice(1)}
            >
              <span className="tab-icon">{icon}</span>
              <span className="tab-label">{tab}</span>
            </button>
          ))}
        </nav>
        
        {/* Quick metrics */}
        <div className="quick-metrics">
          <div className="metric">
            <span className="metric-label">Φ</span>
            <span className="metric-value">{(consciousnessState.phi * 100).toFixed(0)}%</span>
          </div>
          <div className="metric">
            <span className="metric-label">Focus</span>
            <span className="metric-value">{(consciousnessState.attention * 100).toFixed(0)}%</span>
          </div>
        </div>
      </header>
      
      {/* Main content area */}
      <main className="app-content">
        {connectionStatus === 'connecting' ? (
          <div className="consciousness-loading">
            <div className="loading-spinner">
              <div className="spinner-brain">🧠</div>
              <p>Awakening consciousness...</p>
            </div>
          </div>
        ) : (
          renderTabContent()
        )}
      </main>
      
      {/* Status bar */}
      <footer className="app-footer">
        <div className="status-info">
          <span>GPU: RTX 3090</span>
          <span>•</span>
          <span>Memory: {consciousnessState.memoryActivation ? 
            `${(consciousnessState.memoryActivation * 100).toFixed(0)}% active` : 
            'initializing'
          }</span>
          <span>•</span>
          <span>Emotional: {getEmotionalState(consciousnessState.emotional)}</span>
        </div>
      </footer>
    </div>
  )
}

// Connection status indicator component
const ConnectionIndicator = ({ status }) => {
  const statusConfig = {
    connecting: { color: '#ffa500', text: 'Connecting...', pulse: true },
    connected: { color: '#00ff88', text: 'Connected', pulse: false },
    disconnected: { color: '#ff4444', text: 'Disconnected', pulse: true }
  }
  
  const config = statusConfig[status] || statusConfig.disconnected
  
  return (
    <div className="connection-indicator">
      <span 
        className={`status-dot ${config.pulse ? 'pulse' : ''}`}
        style={{ backgroundColor: config.color }}
      />
      <span className="status-text">{config.text}</span>
    </div>
  )
}

// Helper functions
function calculateImportance(content, consciousnessState) {
  // Calculate memory importance based on current consciousness state
  const emotionalIntensity = Math.sqrt(
    Math.pow(consciousnessState.emotional.valence - 0.5, 2) +
    Math.pow(consciousnessState.emotional.arousal - 0.5, 2)
  )
  
  const attentionWeight = consciousnessState.attention || 0.5
  const phiWeight = consciousnessState.phi || 0.5
  
  return (emotionalIntensity * 0.4 + attentionWeight * 0.3 + phiWeight * 0.3)
}

function getEmotionalState(emotional) {
  if (!emotional) return 'neutral'
  
  const { valence, arousal } = emotional
  
  if (valence > 0.7 && arousal > 0.7) return 'excited'
  if (valence > 0.7 && arousal < 0.3) return 'calm'
  if (valence < 0.3 && arousal > 0.7) return 'stressed'
  if (valence < 0.3 && arousal < 0.3) return 'melancholy'
  
  return 'balanced'
}

export default App