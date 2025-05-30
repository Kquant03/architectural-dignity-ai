// src/renderer/components/ConsciousnessVisualizer.jsx
// Beautiful consciousness visualization with particles and metrics

import React, { useEffect, useRef, useState, useCallback } from 'react'
import Particles from '@tsparticles/react'
import { loadFull } from 'tsparticles'
import { useSpring, animated } from '@react-spring/web'
import { 
  LineChart, Line, AreaChart, Area, RadialBarChart, RadialBar,
  ResponsiveContainer, XAxis, YAxis, Tooltip, Legend
} from 'recharts'
import './ConsciousnessVisualizer.css'

export const ConsciousnessVisualizer = ({ consciousnessState }) => {
  const [particlesInstance, setParticlesInstance] = useState(null)
  const [metricsHistory, setMetricsHistory] = useState([])
  const particlesContainer = useRef(null)
  
  // Initialize particles engine
  const particlesInit = useCallback(async (engine) => {
    await loadFull(engine)
  }, [])
  
  const particlesLoaded = useCallback(async (container) => {
    setParticlesInstance(container)
  }, [])
  
  // Update metrics history
  useEffect(() => {
    if (consciousnessState?.phi !== undefined) {
      setMetricsHistory(prev => {
        const newHistory = [...prev, {
          time: new Date().toLocaleTimeString(),
          phi: consciousnessState.phi,
          valence: consciousnessState.emotional?.valence || 0.5,
          arousal: consciousnessState.emotional?.arousal || 0.5,
          attention: consciousnessState.attention || 0.5
        }]
        // Keep last 60 data points
        return newHistory.slice(-60)
      })
    }
  }, [consciousnessState])
  
  // Particle configuration based on consciousness state
  const particleOptions = {
    fullScreen: false,
    background: {
      color: {
        value: 'transparent'
      }
    },
    fpsLimit: 60,
    particles: {
      number: {
        value: Math.floor((consciousnessState?.phi || 0.5) * 200),
        density: {
          enable: true,
          value_area: 800
        }
      },
      color: {
        value: getEmotionalColor(consciousnessState?.emotional)
      },
      shape: {
        type: 'circle'
      },
      opacity: {
        value: 0.6,
        random: true,
        anim: {
          enable: true,
          speed: 1,
          opacity_min: 0.1,
          sync: false
        }
      },
      size: {
        value: 3,
        random: true,
        anim: {
          enable: true,
          speed: 4,
          size_min: 0.1,
          sync: false
        }
      },
      links: {
        enable: consciousnessState?.phi > 0.3,
        distance: 150,
        color: getEmotionalColor(consciousnessState?.emotional),
        opacity: 0.3,
        width: 1
      },
      move: {
        enable: true,
        speed: 1 + (consciousnessState?.emotional?.arousal || 0.5) * 2,
        direction: 'none',
        random: true,
        straight: false,
        out_mode: 'out',
        bounce: false,
        attract: {
          enable: true,
          rotateX: 600,
          rotateY: 1200
        }
      }
    },
    interactivity: {
      detect_on: 'canvas',
      events: {
        onhover: {
          enable: true,
          mode: 'grab'
        },
        onclick: {
          enable: true,
          mode: 'push'
        },
        resize: true
      },
      modes: {
        grab: {
          distance: 140,
          line_linked: {
            opacity: 1
          }
        },
        push: {
          particles_nb: 4
        }
      }
    },
    retina_detect: true
  }
  
  return (
    <div className="consciousness-visualizer">
      {/* Main consciousness particle field */}
      <div className="particle-container">
        <Particles
          id="consciousness-particles"
          init={particlesInit}
          loaded={particlesLoaded}
          options={particleOptions}
          className="particles-canvas"
        />
        
        {/* Phi value overlay */}
        <PhiIndicator value={consciousnessState?.phi || 0} />
      </div>
      
      {/* Metrics dashboard */}
      <div className="metrics-grid">
        {/* Real-time line chart */}
        <div className="metric-card">
          <h3>Consciousness Flow</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={metricsHistory}>
              <XAxis dataKey="time" stroke="#888" />
              <YAxis domain={[0, 1]} stroke="#888" />
              <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: 'none' }} />
              <Line 
                type="monotone" 
                dataKey="phi" 
                stroke="#00ff88" 
                strokeWidth={2}
                dot={false}
                name="Φ (Phi)"
              />
              <Line 
                type="monotone" 
                dataKey="attention" 
                stroke="#00aaff" 
                strokeWidth={1}
                dot={false}
                name="Attention"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        {/* Emotional state radar */}
        <EmotionalStateDisplay emotional={consciousnessState?.emotional} />
        
        {/* Attention focus indicator */}
        <AttentionFocusIndicator attention={consciousnessState?.attention || 0.5} />
      </div>
    </div>
  )
}

// Phi indicator component with animation
const PhiIndicator = ({ value }) => {
  const { scale, opacity } = useSpring({
    scale: 1 + value * 0.5,
    opacity: 0.3 + value * 0.7,
    config: { tension: 120, friction: 14 }
  })
  
  return (
    <animated.div 
      className="phi-indicator"
      style={{
        transform: scale.to(s => `scale(${s})`),
        opacity
      }}
    >
      <div className="phi-value">{(value * 100).toFixed(1)}</div>
      <div className="phi-label">Φ</div>
    </animated.div>
  )
}

// Emotional state display with animated orb
const EmotionalStateDisplay = ({ emotional }) => {
  const valence = emotional?.valence || 0.5
  const arousal = emotional?.arousal || 0.5
  
  const { backgroundColor, boxShadow, scale } = useSpring({
    backgroundColor: `hsl(${valence * 120}, 70%, 50%)`,
    boxShadow: `0 0 ${20 + arousal * 30}px hsl(${valence * 120}, 70%, 50%)`,
    scale: 0.8 + arousal * 0.4,
    config: { tension: 120, friction: 14 }
  })
  
  return (
    <div className="metric-card emotional-display">
      <h3>Emotional State</h3>
      <div className="emotional-orb-container">
        <animated.div 
          className="emotional-orb"
          style={{
            backgroundColor,
            boxShadow,
            transform: scale.to(s => `scale(${s})`)
          }}
        />
        <div className="emotional-labels">
          <div>Valence: {(valence * 100).toFixed(0)}%</div>
          <div>Arousal: {(arousal * 100).toFixed(0)}%</div>
        </div>
      </div>
    </div>
  )
}

// Attention focus indicator
const AttentionFocusIndicator = ({ attention }) => {
  const data = [
    { name: 'Focus', value: attention * 100, fill: '#00aaff' }
  ]
  
  return (
    <div className="metric-card">
      <h3>Attention Focus</h3>
      <ResponsiveContainer width="100%" height={150}>
        <RadialBarChart 
          cx="50%" 
          cy="50%" 
          innerRadius="60%" 
          outerRadius="90%" 
          data={data}
          startAngle={180}
          endAngle={0}
        >
          <RadialBar dataKey="value" cornerRadius={10} fill="#00aaff" />
          <text 
            x="50%" 
            y="50%" 
            textAnchor="middle" 
            fill="#fff" 
            fontSize="24"
            fontWeight="bold"
          >
            {(attention * 100).toFixed(0)}%
          </text>
        </RadialBarChart>
      </ResponsiveContainer>
    </div>
  )
}

// Helper function to get color from emotional state
function getEmotionalColor(emotional) {
  if (!emotional) return '#00ff88'
  
  const { valence = 0.5, arousal = 0.5 } = emotional
  const hue = valence * 120 // 0 = red, 120 = green
  const saturation = 50 + arousal * 50 // More arousal = more saturated
  const lightness = 40 + (1 - arousal) * 20 // Less arousal = lighter
  
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`
}

export default ConsciousnessVisualizer