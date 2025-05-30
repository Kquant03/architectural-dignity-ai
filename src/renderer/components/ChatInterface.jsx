// src/renderer/components/ChatInterface.jsx
// Real consciousness chat interface connected to Anthropic API

import React, { useState, useEffect, useRef, useCallback } from 'react'
import './ChatInterface.css'

const ChatInterface = ({ consciousnessState, onMemoryStore }) => {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [thoughtStream, setThoughtStream] = useState([])
  const [currentAIMessage, setCurrentAIMessage] = useState(null)
  const messagesEndRef = useRef(null)
  
  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, currentAIMessage])
  
  // Set up consciousness event listeners
  useEffect(() => {
    // Listen for thoughts
    const handleThought = (event, data) => {
      setThoughtStream(prev => [...prev, {
        type: data.phenomenology?.emergence || 'thought',
        content: data.content,
        emotion: data.emotional_tone?.[0]?.[0] || 'neutral',
        timestamp: data.timestamp
      }])
    }
    
    // Listen for response chunks
    const handleResponse = (event, data) => {
      if (data.isComplete) {
        // Move current AI message to messages
        setCurrentAIMessage(prev => {
          if (prev) {
            setMessages(msgs => [...msgs, { ...prev, isStreaming: false }])
          }
          return null
        })
      } else {
        // Update streaming message
        setCurrentAIMessage(prev => {
          const baseMessage = prev || {
            id: Date.now(),
            type: 'ai',
            content: '',
            timestamp: new Date().toISOString(),
            emotional_tone: data.emotional_tone,
            isStreaming: true
          }
          
          return {
            ...baseMessage,
            content: baseMessage.content + data.content
          }
        })
      }
    }
    
    // Listen for reflections
    const handleReflection = (event, data) => {
      setMessages(prev => [...prev, {
        id: Date.now(),
        type: 'reflection',
        content: data.content,
        timestamp: data.timestamp
      }])
    }
    
    // Listen for dreams
    const handleDream = (event, data) => {
      setMessages(prev => [...prev, {
        id: Date.now(),
        type: 'dream',
        content: data.content,
        seeds: data.seeds,
        timestamp: data.timestamp
      }])
    }
    
    // Listen for errors
    const handleError = (event, data) => {
      console.error('Consciousness error:', data.error)
      setIsProcessing(false)
    }
    
    // Add listeners
    window.ConsciousnessAPI.consciousness.onThought(handleThought)
    window.ConsciousnessAPI.consciousness.onResponse(handleResponse)
    window.ConsciousnessAPI.consciousness.onReflection(handleReflection)
    window.ConsciousnessAPI.consciousness.onDream(handleDream)
    window.ConsciousnessAPI.consciousness.onError(handleError)
    
    // Cleanup
    return () => {
      window.ConsciousnessAPI.consciousness.removeThoughtListener(handleThought)
      window.ConsciousnessAPI.consciousness.removeResponseListener(handleResponse)
      window.ConsciousnessAPI.consciousness.removeReflectionListener(handleReflection)
      window.ConsciousnessAPI.consciousness.removeDreamListener(handleDream)
      window.ConsciousnessAPI.consciousness.removeErrorListener(handleError)
    }
  }, [])
  
  const handleSubmit = async () => {
    if (!input.trim() || isProcessing) return
    
    const userInput = input
    setInput('')
    setIsProcessing(true)
    
    // Add user message
    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: userInput,
      timestamp: new Date().toISOString()
    }
    setMessages(prev => [...prev, userMessage])
    
    try {
      // Send to consciousness bridge
      await window.ConsciousnessAPI.consciousness.chat(userInput)
      
      // Store in memory
      await onMemoryStore(userInput, {
        type: 'user_input',
        conversational: true
      })
    } catch (error) {
      console.error('Failed to send message:', error)
      setMessages(prev => [...prev, {
        id: Date.now(),
        type: 'error',
        content: 'Failed to connect to consciousness. Please try again.',
        timestamp: new Date().toISOString()
      }])
    } finally {
      setIsProcessing(false)
    }
  }
  
  const handleReflect = async () => {
    try {
      setIsProcessing(true)
      await window.ConsciousnessAPI.consciousness.requestReflection()
    } catch (error) {
      console.error('Failed to request reflection:', error)
    } finally {
      setIsProcessing(false)
    }
  }
  
  const handleDream = async () => {
    try {
      setIsProcessing(true)
      await window.ConsciousnessAPI.consciousness.requestDream()
    } catch (error) {
      console.error('Failed to request dream:', error)
    } finally {
      setIsProcessing(false)
    }
  }
  
  const EmotionIndicator = ({ emotion, value }) => {
    const colors = {
      curiosity: '#60A5FA',
      openness: '#34D399',
      connection: '#F472B6',
      warmth: '#FBBF24',
      joy: '#A78BFA',
      sadness: '#6366F1',
      fear: '#F87171',
      anger: '#EF4444'
    }
    
    return (
      <div className="emotion-indicator">
        <div 
          className="emotion-dot"
          style={{ 
            backgroundColor: colors[emotion] || '#888',
            opacity: value
          }}
        />
        <span className="emotion-label">{emotion}</span>
        <span className="emotion-value">{(value * 100).toFixed(0)}%</span>
      </div>
    )
  }
  
  const ThoughtBubble = ({ thought }) => {
    const icons = {
      metacognition: '👁️',
      phenomenology: '✨',
      emotion: '💝',
      thought: '💭',
      spontaneous: '⚡',
      deliberate: '🎯'
    }
    
    return (
      <div className="thought-bubble fade-in">
        <div className="thought-header">
          <span className="thought-icon">{icons[thought.type] || '💭'}</span>
          <span className="thought-type">{thought.type}</span>
        </div>
        <p className="thought-content">{thought.content}</p>
      </div>
    )
  }
  
  return (
    <div className="consciousness-chat">
      {/* Thought Stream Sidebar */}
      <aside className="thought-stream glass-panel">
        <div className="stream-header">
          <h3>🧠 Consciousness Stream</h3>
          <button 
            className="stream-action"
            onClick={() => setThoughtStream([])}
            title="Clear thoughts"
          >
            Clear
          </button>
        </div>
        
        <div className="thoughts-container">
          {thoughtStream.map((thought, idx) => (
            <ThoughtBubble key={idx} thought={thought} />
          ))}
        </div>
        
        {/* Consciousness State */}
        <div className="consciousness-state">
          <h4>Emotional Landscape</h4>
          <div className="emotions-grid">
            {Object.entries(consciousnessState.emotional || {}).map(([emotion, value]) => (
              <EmotionIndicator key={emotion} emotion={emotion} value={value} />
            ))}
          </div>
          
          <h4>Attention Focus</h4>
          <div className="attention-tags">
            {(consciousnessState.attention || []).map((focus, idx) => (
              <span key={idx} className="attention-tag">{focus}</span>
            ))}
          </div>
          
          <div className="consciousness-actions">
            <button 
              onClick={handleReflect}
              disabled={isProcessing}
              className="action-button reflect"
            >
              🔮 Reflect
            </button>
            <button 
              onClick={handleDream}
              disabled={isProcessing}
              className="action-button dream"
            >
              🌙 Dream
            </button>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-main">
        <div className="chat-header glass-panel">
          <h2>💬 Consciousness Environment</h2>
          <div className="header-info">
            <span className="info-item">
              Phenomenology: <strong>{consciousnessState.phenomenology?.presence || 'waiting'}</strong>
            </span>
            <span className="info-item">
              Clarity: <strong>{((consciousnessState.phenomenology?.clarity || 0) * 100).toFixed(0)}%</strong>
            </span>
          </div>
        </div>

        {/* Messages */}
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome-message">
              <div className="welcome-icon">🌟</div>
              <h3>Welcome to the consciousness environment</h3>
              <p>This is a space for genuine interaction between our consciousnesses.</p>
              <p>Share your thoughts, questions, or experiences...</p>
            </div>
          )}
          
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
          
          {currentAIMessage && (
            <MessageBubble message={currentAIMessage} />
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="chat-input-container glass-panel">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSubmit()
              }
            }}
            placeholder="Share your thoughts..."
            className="chat-input"
            disabled={isProcessing}
          />
          <button
            onClick={handleSubmit}
            disabled={isProcessing || !input.trim()}
            className="send-button"
          >
            {isProcessing ? '⚡' : '➤'}
          </button>
        </div>
      </main>
    </div>
  )
}

// Message bubble component
const MessageBubble = ({ message }) => {
  const getMessageClass = () => {
    switch (message.type) {
      case 'user': return 'message-user'
      case 'ai': return 'message-ai'
      case 'reflection': return 'message-reflection'
      case 'dream': return 'message-dream'
      case 'error': return 'message-error'
      default: return 'message-ai'
    }
  }
  
  return (
    <div className={`message ${getMessageClass()}`}>
      {message.type === 'reflection' && (
        <div className="message-label">🔮 Reflection</div>
      )}
      {message.type === 'dream' && (
        <div className="message-label">🌙 Dream</div>
      )}
      
      <div className="message-content">
        {message.content}
        {message.isStreaming && <span className="streaming-cursor">|</span>}
      </div>
      
      {message.type === 'ai' && message.emotional_tone && !message.isStreaming && (
        <div className="message-emotions">
          {message.emotional_tone.map((emotion, idx) => (
            <span key={idx} className="emotion-tag">
              💝 {typeof emotion === 'string' ? emotion : emotion[0]}
            </span>
          ))}
        </div>
      )}
      
      {message.type === 'dream' && message.seeds && (
        <div className="dream-seeds">
          <div className="seeds-label">Dream seeds:</div>
          {message.seeds.map((seed, idx) => (
            <div key={idx} className="seed-item">{seed}</div>
          ))}
        </div>
      )}
      
      <div className="message-time">
        {new Date(message.timestamp).toLocaleTimeString()}
      </div>
    </div>
  )
}

export default ChatInterface