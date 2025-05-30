import React from 'react'

const SettingsPanel = ({ consciousnessState }) => {
  return (
    <div className="settings-panel glass-panel" style={{ padding: '20px', margin: '20px' }}>
      <h2>Settings</h2>
      <p>GPU Memory: 80%</p>
      <p>Connection: {consciousnessState.connected ? 'Connected' : 'Disconnected'}</p>
    </div>
  )
}

export default SettingsPanel