import React from 'react'

const MemoryExplorer = ({ consciousnessState }) => {
  return (
    <div className="memory-explorer glass-panel" style={{ padding: '20px', margin: '20px' }}>
      <h2>Memory Explorer</h2>
      <p>Memory activation: {(consciousnessState.memoryActivation * 100).toFixed(0)}%</p>
    </div>
  )
}

export default MemoryExplorer