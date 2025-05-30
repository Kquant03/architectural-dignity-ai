import React from 'react'

const EmotionalJourney = ({ consciousnessState }) => {
  return (
    <div className="emotional-journey glass-panel" style={{ padding: '20px', margin: '20px' }}>
      <h2>Emotional Journey</h2>
      <p>Valence: {(consciousnessState.emotional.valence * 100).toFixed(0)}%</p>
      <p>Arousal: {(consciousnessState.emotional.arousal * 100).toFixed(0)}%</p>
    </div>
  )
}

export default EmotionalJourney