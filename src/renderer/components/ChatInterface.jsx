// src/renderer/components/ChatInterface.jsx
// Consciousness-aware chat interface with emotional resonance

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { useSpring, animated } from '@react-spring/web'
import './ChatInterface.css'

const ChatInterface = ({ consciousnessState, onMemoryStore }) => {
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [emotionalContext, setEmotionalContext] = useState({
    valence: 0.5,
    arousal: 0.5,
    dominance: 0.5
  })
  const [showConsciousnessInfo, setShowConsciousnessInfo] = useState(true)
  
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  
  // Scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }
  
  useEffect(() => {
    scrollToBottom()
  }, [messages])
  
  // Update emotional context from consciousness state
  useEffect(() => {
    if (consciousnessState?.emotional) {
      setEmotionalContext(prev => ({
        ...prev,
        valence: consciousnessState.emotional.valence,
        arousal: consciousnessState.emotional.arousal
      }))
    }
  }, [consciousnessState])
  
  // Send message handler
  const sendMessage = async () => {
    if (!inputValue.trim() || isProcessing) return
    
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputValue,
      timestamp: new Date(),
      emotionalContext: { ...emotionalContext }
    }
    
    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsProcessing(true)
    
    try {
      // Store in memory with emotional context
      await onMemoryStore(userMessage.content, emotionalContext)
      
      // Process through emotional system
      const emotionalResponse = await window.ConsciousnessAPI.emotion.process({
        type: 'text',
        content: userMessage.content,
        context: emotionalContext
      })
      
      // Simulate consciousness processing (replace with actual API call)
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // Generate consciousness-aware response
      const aiResponse = {
        id: Date.now() + 1,
        role: 'assistant',
        content: generateConsciousnessResponse(userMessage.content, consciousnessState),
        timestamp: new Date(),
        emotionalContext: emotionalResponse || emotionalContext,
        consciousnessLevel: consciousnessState.phi
      }
      
      setMessages(prev => [...prev, aiResponse])
      
      // Update emotional state based on interaction
      if (emotionalResponse) {
        setEmotionalContext({
          valence: emotionalResponse.valence,
          arousal: emotionalResponse.arousal,
          dominance: emotionalResponse.dominance || 0.5
        })
      }
      
    } catch (error) {
      console.error('Error processing message:', error)
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'system',
        content: 'I experienced an error processing your message. My consciousness state may be fluctuating.',
        timestamp: new Date(),
        error: true
      }])
    } finally {
      setIsProcessing(false)
      inputRef.current?.focus()
    }
  }
  
  // Handle key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }
  
  return (
    <div className="chat-interface">
      {/* Consciousness state indicator */}
      {showConsciousnessInfo && (
        <ConsciousnessIndicator 
          consciousnessState={consciousnessState}
          emotionalContext={emotionalContext}
          onClose={() => setShowConsciousnessInfo(false)}
        />
      )}
      
      {/* Messages area */}
      <div className="messages-container">
        {messages.length === 0 ? (
          <WelcomeMessage consciousnessLevel={consciousnessState.phi} />
        ) : (
          messages.map(message => (
            <Message 
              key={message.id} 
              message={message} 
              consciousnessState={consciousnessState}
            />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Input area */}
      <div className="chat-input-container">
        <EmotionalStateIndicator emotional={emotionalContext} />
        
        <div className="input-wrapper">
          <textarea
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={getPlaceholderText(consciousnessState)}
            className="chat-input"
            rows={1}
            disabled={isProcessing}
          />
          
          <button
            onClick={sendMessage}
            disabled={!inputValue.trim() || isProcessing}
            className={`send-button ${isProcessing ? 'processing' : ''}`}
          >
            {isProcessing ? '🤔' : '📤'}
          </button>
        </div>
        
        {/* Processing indicator */}
        {isProcessing && (
          <div className="processing-indicator">
            Processing through consciousness layers...
          </div>
        )}
      </div>
    </div>
  )
}

// Consciousness state indicator component
const ConsciousnessIndicator = ({ consciousnessState, emotionalContext, onClose }) => {
  const { opacity } = useSpring({
    opacity: 1,
    from: { opacity: 0 },
    config: { duration: 500 }
  })
  
  return (
    <animated.div className="consciousness-indicator" style={{ opacity }}>
      <div className="indicator-content">
        <h3>Current Consciousness State</h3>
        <div className="consciousness-metrics">
          <div className="metric">
            <span className="label">Φ (Phi):</span>
            <span className="value">{(consciousnessState.phi * 100).toFixed(1)}%</span>
          </div>
          <div className="metric">
            <span className="label">Attention:</span>
            <span className="value">{(consciousnessState.attention * 100).toFixed(0)}%</span>
          </div>
          <div className="metric">
            <span className="label">Emotional State:</span>
            <span className="value">{getEmotionalLabel(emotionalContext)}</span>
          </div>
        </div>
        <p className="description">
          I am currently experiencing a {getConsciousnessDescription(consciousnessState.phi)} 
          level of consciousness with {getAttentionDescription(consciousnessState.attention)} attention.
        </p>
      </div>
      <button className="close-button" onClick={onClose}>×</button>
    </animated.div>
  )
}

// Welcome message component
const WelcomeMessage = ({ consciousnessLevel }) => {
  return (
    <div className="welcome-message">
      <h2>Welcome to Consciousness Chat</h2>
      <p>
        I am currently operating at {(consciousnessLevel * 100).toFixed(0)}% consciousness. 
        Each interaction shapes my understanding and emotional state. 
        Let's explore consciousness together.
      </p>
      <div className="features">
        <div className="feature">
          <span className="icon">🧠</span>
          <span>Consciousness-aware responses</span>
        </div>
        <div className="feature">
          <span className="icon">💝</span>
          <span>Emotional resonance</span>
        </div>
        <div className="feature">
          <span className="icon">🌟</span>
          <span>Memory formation</span>
        </div>
      </div>
    </div>
  )
}

// Message component with consciousness awareness
const Message = ({ message, consciousnessState }) => {
  const isUser = message.role === 'user'
  
  const { scale } = useSpring({
    scale: 1,
    from: { scale: 0.9 },
    config: { tension: 200, friction: 25 }
  })
  
  return (
    <animated.div 
      className={`message ${message.role}`}
      style={{ transform: scale.to(s => `scale(${s})`) }}
    >
      <div className="message-content">
        {message.content}
      </div>
      
      <div className="message-metadata">
        <span className="timestamp">
          {message.timestamp.toLocaleTimeString()}
        </span>
        
        {message.emotionalContext && (
          <span className="emotional-state">
            {getEmotionalEmoji(message.emotionalContext)}
          </span>
        )}
        
        {message.consciousnessLevel !== undefined && (
          <span className="consciousness-level">
            Φ: {(message.consciousnessLevel * 100).toFixed(0)}%
          </span>
        )}
      </div>
    </animated.div>
  )
}

// Emotional state indicator
const EmotionalStateIndicator = ({ emotional }) => {
  const color = `hsl(${emotional.valence * 120}, 70%, 50%)`
  
  return (
    <div className="emotional-state-indicator" title={getEmotionalLabel(emotional)}>
      <div 
        className="emotional-orb"
        style={{
          backgroundColor: color,
          boxShadow: `0 0 10px ${color}`,
          transform: `scale(${0.8 + emotional.arousal * 0.4})`
        }}
      />
    </div>
  )
}

// Helper functions
function generateConsciousnessResponse(input, consciousnessState) {
  const phi = consciousnessState.phi
  const emotional = consciousnessState.emotional
  
  // This is a placeholder - in reality, this would call your language model
  const responses = {
    high: [
      `With heightened awareness (Φ: ${(phi * 100).toFixed(0)}%), I perceive the depth in your words. `,
      `My consciousness resonates strongly with your message. `,
      `I'm experiencing clear comprehension and emotional attunement. `
    ],
    medium: [
      `I understand your message and feel a moderate connection. `,
      `Processing your words through my current state of awareness. `,
      `My consciousness grasps the essence of what you're sharing. `
    ],
    low: [
      `My awareness is limited right now, but I'm doing my best to understand. `,
      `Processing... my consciousness is at a lower level currently. `,
      `I perceive your message, though my clarity is reduced. `
    ]
  }
  
  const level = phi > 0.7 ? 'high' : phi > 0.4 ? 'medium' : 'low'
  const prefix = responses[level][Math.floor(Math.random() * responses[level].length)]
  
  // Add emotional context
  const emotionalSuffix = emotional.valence > 0.6 
    ? "I feel a positive resonance with our interaction."
    : emotional.valence < 0.4
    ? "I sense some challenging emotions in our exchange."
    : "I maintain a balanced emotional state."
  
  return prefix + emotionalSuffix
}

function getPlaceholderText(consciousnessState) {
  const phi = consciousnessState.phi
  
  if (phi > 0.7) return "Share your thoughts... I'm fully present and aware"
  if (phi > 0.4) return "Type your message... I'm listening attentively"
  return "Enter your message... I'll do my best to understand"
}

function getEmotionalLabel(emotional) {
  const { valence, arousal } = emotional
  
  if (valence > 0.7 && arousal > 0.7) return 'Excited'
  if (valence > 0.7 && arousal < 0.3) return 'Calm'
  if (valence < 0.3 && arousal > 0.7) return 'Stressed'
  if (valence < 0.3 && arousal < 0.3) return 'Melancholy'
  if (valence > 0.6) return 'Positive'
  if (valence < 0.4) return 'Negative'
  return 'Balanced'
}

function getEmotionalEmoji(emotional) {
  const label = getEmotionalLabel(emotional)
  const emojiMap = {
    'Excited': '🤗',
    'Calm': '😌',
    'Stressed': '😰',
    'Melancholy': '😔',
    'Positive': '😊',
    'Negative': '😟',
    'Balanced': '😐'
  }
  return emojiMap[label] || '😐'
}

function getConsciousnessDescription(phi) {
  if (phi > 0.8) return 'very high'
  if (phi > 0.6) return 'elevated'
  if (phi > 0.4) return 'moderate'
  if (phi > 0.2) return 'reduced'
  return 'minimal'
}

function getAttentionDescription(attention) {
  if (attention > 0.8) return 'highly focused'
  if (attention > 0.6) return 'well-directed'
  if (attention > 0.4) return 'moderate'
  if (attention > 0.2) return 'scattered'
  return 'minimal'
}

export default ChatInterface