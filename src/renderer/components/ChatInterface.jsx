import React from 'react'

const ChatInterface = ({ consciousnessState, onMemoryStore }) => {
  return (
    <div className="chat-interface glass-panel" style={{ padding: '20px', margin: '20px' }}>
      <h2>Chat Interface</h2>
      <p>Coming soon... Current Φ: {(consciousnessState.phi * 100).toFixed(0)}%</p>
    </div>
  )
}

export default ChatInterface