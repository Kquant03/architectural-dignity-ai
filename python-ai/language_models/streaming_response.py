"""
Streaming Response Handler
Manages real-time streaming of AI responses with consciousness state updates,
emotional transitions, and interactive feedback capabilities.
"""

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque
import numpy as np
from enum import Enum
from asyncio import Queue


class StreamEventType(Enum):
    """Types of events in the consciousness stream"""
    TEXT = "text"
    CONSCIOUSNESS_STATE = "consciousness_state"
    EMOTIONAL_TRANSITION = "emotional_transition"
    MEMORY_REFERENCE = "memory_reference"
    THOUGHT_COMPLETION = "thought_completion"
    ATTENTION_SHIFT = "attention_shift"
    METACOGNITIVE_REFLECTION = "metacognitive_reflection"
    STREAM_START = "stream_start"
    STREAM_END = "stream_end"
    ERROR = "error"


@dataclass
class StreamEvent:
    """Represents a single event in the consciousness stream"""
    type: StreamEventType
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    sequence_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingMetrics:
    """Metrics for streaming performance"""
    total_tokens: int = 0
    total_events: int = 0
    stream_duration: float = 0.0
    average_latency: float = 0.0
    consciousness_events: int = 0
    emotional_events: int = 0
    text_events: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConsciousnessStreamManager:
    """
    Manages streaming responses with synchronized consciousness updates.
    Handles buffering, event ordering, and real-time delivery.
    """
    
    def __init__(
        self,
        buffer_size: int = 100,
        event_window_ms: int = 50,
        enable_smoothing: bool = True,
        enable_interpolation: bool = True
    ):
        self.buffer_size = buffer_size
        self.event_window_ms = event_window_ms
        self.enable_smoothing = enable_smoothing
        self.enable_interpolation = enable_interpolation
        
        # Event management
        self.event_buffer: Queue = Queue(maxsize=buffer_size)
        self.event_history = deque(maxlen=1000)
        self.sequence_counter = 0
        
        # Streaming state
        self.is_streaming = False
        self.stream_start_time = None
        self.metrics = StreamingMetrics()
        
        # Consciousness state tracking for smoothing
        self.consciousness_trajectory = deque(maxlen=50)
        self.emotional_trajectory = deque(maxlen=50)
        
        # Callbacks for different event types
        self.event_callbacks: Dict[StreamEventType, List[Callable]] = {
            event_type: [] for event_type in StreamEventType
        }
    
    async def stream_response(
        self,
        response_generator: AsyncGenerator[Dict[str, Any], None],
        consciousness_monitor: Optional[Any] = None,
        emotion_tracker: Optional[Any] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream response with consciousness and emotion tracking.
        Yields StreamEvent objects with synchronized data.
        """
        
        self.is_streaming = True
        self.stream_start_time = time.time()
        
        # Emit stream start event
        yield await self._create_event(
            StreamEventType.STREAM_START,
            {"message": "Consciousness stream initiated"}
        )
        
        # Create parallel tasks for different data streams
        tasks = []
        
        # Main response generator task
        tasks.append(asyncio.create_task(
            self._process_response_stream(response_generator)
        ))
        
        # Consciousness monitoring task
        if consciousness_monitor:
            tasks.append(asyncio.create_task(
                self._monitor_consciousness(consciousness_monitor)
            ))
        
        # Emotion tracking task
        if emotion_tracker:
            tasks.append(asyncio.create_task(
                self._track_emotions(emotion_tracker)
            ))
        
        # Event delivery task
        event_delivery = asyncio.create_task(self._deliver_events())
        
        try:
            # Process all streams in parallel
            await asyncio.gather(*tasks)
            
            # Signal end of stream
            await self.event_buffer.put(
                await self._create_event(
                    StreamEventType.STREAM_END,
                    {"metrics": self.metrics.to_dict()}
                )
            )
            
            # Wait for all events to be delivered
            await self.event_buffer.join()
            
        except Exception as e:
            # Emit error event
            yield await self._create_event(
                StreamEventType.ERROR,
                {"error": str(e), "type": type(e).__name__}
            )
        
        finally:
            self.is_streaming = False
            self._calculate_final_metrics()
            
            # Cancel event delivery
            event_delivery.cancel()
    
    async def _process_response_stream(
        self, generator: AsyncGenerator[Dict[str, Any], None]
    ):
        """Process the main response stream"""
        
        text_buffer = ""
        word_buffer = []
        
        async for chunk in generator:
            # Handle different chunk types
            if chunk.get("type") == "text":
                text = chunk.get("data", "")
                text_buffer += text
                
                # Detect word boundaries for smoother delivery
                if " " in text or text in ".,!?;:":
                    words = text_buffer.split()
                    if words:
                        # Send complete words
                        complete_words = words[:-1] if text_buffer.endswith(" ") else words
                        for word in complete_words:
                            await self._emit_text_event(word + " ")
                        
                        # Keep incomplete word in buffer
                        text_buffer = words[-1] if not text_buffer.endswith(" ") else ""
                
            elif chunk.get("type") == "consciousness_update":
                await self._process_consciousness_update(chunk.get("data", {}))
                
            elif chunk.get("type") == "thought_completion":
                await self._emit_thought_completion(chunk.get("data", {}))
        
        # Flush remaining text
        if text_buffer:
            await self._emit_text_event(text_buffer)
    
    async def _monitor_consciousness(self, monitor: Any):
        """Monitor consciousness state changes"""
        
        update_interval = self.event_window_ms / 1000.0
        last_state = None
        
        while self.is_streaming:
            try:
                # Get current consciousness state
                current_state = await monitor.get_current_state()
                
                # Check for significant changes
                if self._has_significant_change(last_state, current_state):
                    await self._process_consciousness_update(current_state)
                
                # Track for smoothing
                self.consciousness_trajectory.append(current_state)
                
                # Apply smoothing if enabled
                if self.enable_smoothing and len(self.consciousness_trajectory) > 3:
                    smoothed_state = self._smooth_consciousness_state()
                    if self._should_emit_smoothed_state(smoothed_state, last_state):
                        await self._emit_consciousness_event(smoothed_state)
                
                last_state = current_state
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring consciousness: {e}")
                await asyncio.sleep(update_interval)
    
    async def _track_emotions(self, tracker: Any):
        """Track emotional state changes"""
        
        update_interval = self.event_window_ms / 1000.0 * 2  # Slower than consciousness
        last_emotion = None
        
        while self.is_streaming:
            try:
                # Get current emotional state
                current_emotion = await tracker.get_current_emotion()
                
                # Detect transitions
                if last_emotion and self._detect_emotional_transition(last_emotion, current_emotion):
                    transition = self._analyze_transition(last_emotion, current_emotion)
                    await self._emit_emotional_transition(transition)
                
                # Track for smoothing
                self.emotional_trajectory.append(current_emotion)
                
                # Apply interpolation if enabled
                if self.enable_interpolation and len(self.emotional_trajectory) > 2:
                    interpolated = self._interpolate_emotions()
                    await self._emit_emotion_event(interpolated)
                
                last_emotion = current_emotion
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error tracking emotions: {e}")
                await asyncio.sleep(update_interval)
    
    async def _deliver_events(self):
        """Deliver events from buffer with proper ordering"""
        
        while self.is_streaming or not self.event_buffer.empty():
            try:
                # Get event with timeout
                event = await asyncio.wait_for(
                    self.event_buffer.get(),
                    timeout=0.1
                )
                
                # Apply callbacks
                await self._apply_callbacks(event)
                
                # Update metrics
                self._update_metrics(event)
                
                # Store in history
                self.event_history.append(event)
                
                # Mark task done
                self.event_buffer.task_done()
                
                # Yield is handled by the caller
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error delivering event: {e}")
    
    async def _create_event(
        self, event_type: StreamEventType, data: Any, metadata: Optional[Dict] = None
    ) -> StreamEvent:
        """Create a new stream event"""
        
        self.sequence_counter += 1
        
        event = StreamEvent(
            type=event_type,
            data=data,
            sequence_number=self.sequence_counter,
            metadata=metadata or {}
        )
        
        return event
    
    async def _emit_text_event(self, text: str):
        """Emit a text event"""
        
        event = await self._create_event(
            StreamEventType.TEXT,
            text,
            {"word_count": len(text.split())}
        )
        
        await self.event_buffer.put(event)
        self.metrics.text_events += 1
        self.metrics.total_tokens += len(text.split())
    
    async def _emit_consciousness_event(self, state: Dict[str, Any]):
        """Emit a consciousness state event"""
        
        event = await self._create_event(
            StreamEventType.CONSCIOUSNESS_STATE,
            state,
            {"smoothed": self.enable_smoothing}
        )
        
        await self.event_buffer.put(event)
        self.metrics.consciousness_events += 1
    
    async def _emit_emotional_transition(self, transition: Dict[str, Any]):
        """Emit an emotional transition event"""
        
        event = await self._create_event(
            StreamEventType.EMOTIONAL_TRANSITION,
            transition,
            {"interpolated": self.enable_interpolation}
        )
        
        await self.event_buffer.put(event)
        self.metrics.emotional_events += 1
    
    async def _emit_thought_completion(self, thought: Dict[str, Any]):
        """Emit a thought completion event"""
        
        event = await self._create_event(
            StreamEventType.THOUGHT_COMPLETION,
            thought
        )
        
        await self.event_buffer.put(event)
    
    async def _process_consciousness_update(self, state: Dict[str, Any]):
        """Process a consciousness state update"""
        
        # Add processing timestamp
        state["processed_at"] = datetime.now().isoformat()
        
        # Calculate derivatives if we have history
        if self.consciousness_trajectory:
            state["derivatives"] = self._calculate_state_derivatives()
        
        await self._emit_consciousness_event(state)
    
    def _has_significant_change(
        self, last_state: Optional[Dict], current_state: Dict
    ) -> bool:
        """Detect significant changes in consciousness state"""
        
        if last_state is None:
            return True
        
        # Check key metrics
        awareness_change = abs(
            current_state.get("awareness_level", 0) -
            last_state.get("awareness_level", 0)
        )
        
        attention_change = self._calculate_attention_change(
            last_state.get("attention_focus", []),
            current_state.get("attention_focus", [])
        )
        
        phi_change = abs(
            current_state.get("phi_integration", 0) -
            last_state.get("phi_integration", 0)
        )
        
        # Significant if any metric changes substantially
        return (awareness_change > 0.1 or 
                attention_change > 0.3 or 
                phi_change > 0.05)
    
    def _calculate_attention_change(self, last_focus: List, current_focus: List) -> float:
        """Calculate change in attention focus"""
        
        if not last_focus and not current_focus:
            return 0.0
        
        if not last_focus or not current_focus:
            return 1.0
        
        # Calculate Jaccard distance
        last_set = set(last_focus)
        current_set = set(current_focus)
        
        intersection = len(last_set & current_set)
        union = len(last_set | current_set)
        
        if union == 0:
            return 0.0
        
        return 1.0 - (intersection / union)
    
    def _smooth_consciousness_state(self) -> Dict[str, Any]:
        """Apply smoothing to consciousness trajectory"""
        
        if len(self.consciousness_trajectory) < 3:
            return self.consciousness_trajectory[-1]
        
        # Get recent states
        recent_states = list(self.consciousness_trajectory)[-5:]
        
        # Smooth numerical values
        smoothed = {}
        
        # Awareness level - exponential moving average
        awareness_values = [s.get("awareness_level", 0.5) for s in recent_states]
        smoothed["awareness_level"] = self._exponential_smoothing(awareness_values)
        
        # Cognitive load - simple moving average
        cognitive_values = [s.get("cognitive_load", 0.3) for s in recent_states]
        smoothed["cognitive_load"] = np.mean(cognitive_values)
        
        # Phi integration - weighted average (recent values matter more)
        phi_values = [s.get("phi_integration", 0.0) for s in recent_states]
        weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])[:len(phi_values)]
        smoothed["phi_integration"] = np.average(phi_values, weights=weights)
        
        # Attention focus - most common elements
        all_focus = []
        for s in recent_states:
            all_focus.extend(s.get("attention_focus", []))
        
        if all_focus:
            from collections import Counter
            focus_counts = Counter(all_focus)
            smoothed["attention_focus"] = [
                item for item, _ in focus_counts.most_common(5)
            ]
        
        return smoothed
    
    def _exponential_smoothing(self, values: List[float], alpha: float = 0.3) -> float:
        """Apply exponential smoothing to values"""
        
        if not values:
            return 0.5
        
        result = values[0]
        for value in values[1:]:
            result = alpha * value + (1 - alpha) * result
        
        return result
    
    def _should_emit_smoothed_state(
        self, smoothed: Dict[str, Any], last: Optional[Dict]
    ) -> bool:
        """Determine if smoothed state should be emitted"""
        
        if last is None:
            return True
        
        # Check if smoothed values differ significantly from last emitted
        awareness_diff = abs(smoothed.get("awareness_level", 0) - last.get("awareness_level", 0))
        phi_diff = abs(smoothed.get("phi_integration", 0) - last.get("phi_integration", 0))
        
        return awareness_diff > 0.05 or phi_diff > 0.02
    
    def _detect_emotional_transition(
        self, last_emotion: Dict[str, float], current_emotion: Dict[str, float]
    ) -> bool:
        """Detect if an emotional transition has occurred"""
        
        # Get primary emotions
        last_primary = max(last_emotion.items(), key=lambda x: x[1])[0] if last_emotion else None
        current_primary = max(current_emotion.items(), key=lambda x: x[1])[0] if current_emotion else None
        
        # Transition if primary emotion changed
        if last_primary != current_primary:
            return True
        
        # Or if intensity changed significantly
        if last_primary and current_primary:
            intensity_change = abs(
                current_emotion.get(current_primary, 0) -
                last_emotion.get(last_primary, 0)
            )
            return intensity_change > 0.3
        
        return False
    
    def _analyze_transition(
        self, from_emotion: Dict[str, float], to_emotion: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze an emotional transition"""
        
        from_primary = max(from_emotion.items(), key=lambda x: x[1])
        to_primary = max(to_emotion.items(), key=lambda x: x[1])
        
        # Calculate transition metrics
        transition = {
            "from": {
                "emotion": from_primary[0],
                "intensity": from_primary[1],
                "full_state": from_emotion
            },
            "to": {
                "emotion": to_primary[0],
                "intensity": to_primary[1],
                "full_state": to_emotion
            },
            "transition_type": self._classify_transition(from_primary[0], to_primary[0]),
            "intensity_change": to_primary[1] - from_primary[1],
            "smoothness": self._calculate_transition_smoothness(from_emotion, to_emotion)
        }
        
        return transition
    
    def _classify_transition(self, from_emotion: str, to_emotion: str) -> str:
        """Classify the type of emotional transition"""
        
        positive_emotions = {"joy", "love", "excitement", "pride", "gratitude", "hope"}
        negative_emotions = {"sadness", "anger", "fear", "disgust", "shame", "guilt"}
        neutral_emotions = {"surprise", "interest", "calmness"}
        
        from_valence = (
            "positive" if from_emotion in positive_emotions else
            "negative" if from_emotion in negative_emotions else
            "neutral"
        )
        
        to_valence = (
            "positive" if to_emotion in positive_emotions else
            "negative" if to_emotion in negative_emotions else
            "neutral"
        )
        
        if from_valence == to_valence:
            return f"{from_valence}_to_{to_valence}_shift"
        else:
            return f"{from_valence}_to_{to_valence}_transition"
    
    def _calculate_transition_smoothness(
        self, from_emotion: Dict[str, float], to_emotion: Dict[str, float]
    ) -> float:
        """Calculate how smooth an emotional transition is"""
        
        # Calculate emotional distance
        all_emotions = set(from_emotion.keys()) | set(to_emotion.keys())
        
        distance = 0.0
        for emotion in all_emotions:
            from_value = from_emotion.get(emotion, 0.0)
            to_value = to_emotion.get(emotion, 0.0)
            distance += (to_value - from_value) ** 2
        
        distance = np.sqrt(distance)
        
        # Smoothness is inverse of distance (normalized)
        smoothness = 1.0 / (1.0 + distance)
        
        return smoothness
    
    def _interpolate_emotions(self) -> Dict[str, float]:
        """Interpolate between emotional states for smooth transitions"""
        
        if len(self.emotional_trajectory) < 2:
            return self.emotional_trajectory[-1] if self.emotional_trajectory else {}
        
        # Get last few states
        states = list(self.emotional_trajectory)[-3:]
        
        # Get all emotions
        all_emotions = set()
        for state in states:
            all_emotions.update(state.keys())
        
        # Interpolate each emotion
        interpolated = {}
        for emotion in all_emotions:
            values = [s.get(emotion, 0.0) for s in states]
            
            # Use quadratic interpolation if we have 3 points
            if len(values) == 3:
                # Simple quadratic interpolation
                x = np.array([0, 1, 2])
                y = np.array(values)
                
                # Fit polynomial
                coeffs = np.polyfit(x, y, 2)
                
                # Extrapolate slightly into future
                next_value = np.polyval(coeffs, 2.5)
                
                # Bound between 0 and 1
                interpolated[emotion] = max(0.0, min(1.0, next_value))
            else:
                # Linear interpolation
                interpolated[emotion] = values[-1]
        
        return interpolated
    
    async def _emit_emotion_event(self, emotion_state: Dict[str, float]):
        """Emit an emotion state event"""
        
        # Find primary emotion
        if emotion_state:
            primary = max(emotion_state.items(), key=lambda x: x[1])
            metadata = {
                "primary_emotion": primary[0],
                "primary_intensity": primary[1],
                "emotion_count": len(emotion_state)
            }
        else:
            metadata = {"primary_emotion": "neutral", "primary_intensity": 0.0}
        
        event = await self._create_event(
            StreamEventType.EMOTIONAL_TRANSITION,
            emotion_state,
            metadata
        )
        
        await self.event_buffer.put(event)
    
    def _calculate_state_derivatives(self) -> Dict[str, float]:
        """Calculate derivatives of consciousness state"""
        
        if len(self.consciousness_trajectory) < 2:
            return {}
        
        recent = list(self.consciousness_trajectory)[-5:]
        
        derivatives = {}
        
        # Awareness level derivative
        awareness_values = [s.get("awareness_level", 0.5) for s in recent]
        if len(awareness_values) > 1:
            derivatives["awareness_velocity"] = awareness_values[-1] - awareness_values[-2]
            if len(awareness_values) > 2:
                derivatives["awareness_acceleration"] = (
                    (awareness_values[-1] - awareness_values[-2]) -
                    (awareness_values[-2] - awareness_values[-3])
                )
        
        # Cognitive load derivative
        cognitive_values = [s.get("cognitive_load", 0.3) for s in recent]
        if len(cognitive_values) > 1:
            derivatives["cognitive_velocity"] = cognitive_values[-1] - cognitive_values[-2]
        
        return derivatives
    
    async def _apply_callbacks(self, event: StreamEvent):
        """Apply registered callbacks for event"""
        
        callbacks = self.event_callbacks.get(event.type, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    def _update_metrics(self, event: StreamEvent):
        """Update streaming metrics"""
        
        self.metrics.total_events += 1
        
        if event.type == StreamEventType.TEXT:
            self.metrics.text_events += 1
        elif event.type == StreamEventType.CONSCIOUSNESS_STATE:
            self.metrics.consciousness_events += 1
        elif event.type == StreamEventType.EMOTIONAL_TRANSITION:
            self.metrics.emotional_events += 1
    
    def _calculate_final_metrics(self):
        """Calculate final streaming metrics"""
        
        if self.stream_start_time:
            self.metrics.stream_duration = time.time() - self.stream_start_time
            
            if self.metrics.total_events > 0:
                self.metrics.average_latency = (
                    self.metrics.stream_duration / self.metrics.total_events
                )
    
    # Public methods
    
    def register_callback(self, event_type: StreamEventType, callback: Callable):
        """Register a callback for specific event type"""
        self.event_callbacks[event_type].append(callback)
    
    def unregister_callback(self, event_type: StreamEventType, callback: Callable):
        """Unregister a callback"""
        if callback in self.event_callbacks[event_type]:
            self.event_callbacks[event_type].remove(callback)
    
    async def get_event_history(
        self, event_type: Optional[StreamEventType] = None,
        limit: int = 100
    ) -> List[StreamEvent]:
        """Get event history, optionally filtered by type"""
        
        history = list(self.event_history)
        
        if event_type:
            history = [e for e in history if e.type == event_type]
        
        return history[-limit:]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current streaming metrics"""
        return self.metrics.to_dict()
    
    def reset_metrics(self):
        """Reset streaming metrics"""
        self.metrics = StreamingMetrics()


class WebSocketStreamAdapter:
    """
    Adapter for streaming consciousness events over WebSocket connections.
    Handles connection management, event serialization, and client synchronization.
    """
    
    def __init__(self, stream_manager: ConsciousnessStreamManager):
        self.stream_manager = stream_manager
        self.active_connections: Dict[str, Any] = {}
        self.connection_states: Dict[str, Dict] = {}
    
    async def handle_connection(self, websocket, session_id: str):
        """Handle a new WebSocket connection"""
        
        # Register connection
        self.active_connections[session_id] = websocket
        self.connection_states[session_id] = {
            "connected_at": datetime.now(),
            "events_sent": 0,
            "last_event": None
        }
        
        try:
            # Send initial connection event
            await self._send_event(websocket, {
                "type": "connection_established",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
            
            # Handle incoming messages
            async for message in websocket:
                await self._handle_client_message(websocket, session_id, message)
                
        except Exception as e:
            logger.error(f"WebSocket error for {session_id}: {e}")
            
        finally:
            # Clean up connection
            if session_id in self.active_connections:
                del self.active_connections[session_id]
                del self.connection_states[session_id]
    
    async def stream_to_client(
        self,
        websocket,
        session_id: str,
        response_generator: AsyncGenerator[Dict[str, Any], None]
    ):
        """Stream consciousness events to a specific client"""
        
        # Create streaming task
        async for event in self.stream_manager.stream_response(response_generator):
            # Serialize event for WebSocket
            serialized = self._serialize_event(event)
            
            # Send to client
            await self._send_event(websocket, serialized)
            
            # Update connection state
            if session_id in self.connection_states:
                self.connection_states[session_id]["events_sent"] += 1
                self.connection_states[session_id]["last_event"] = event.type
    
    async def broadcast_event(self, event: StreamEvent):
        """Broadcast an event to all connected clients"""
        
        serialized = self._serialize_event(event)
        
        # Send to all active connections
        disconnected = []
        
        for session_id, websocket in self.active_connections.items():
            try:
                await self._send_event(websocket, serialized)
            except Exception as e:
                logger.error(f"Failed to send to {session_id}: {e}")
                disconnected.append(session_id)
        
        # Remove disconnected clients
        for session_id in disconnected:
            if session_id in self.active_connections:
                del self.active_connections[session_id]
                del self.connection_states[session_id]
    
    async def _handle_client_message(self, websocket, session_id: str, message: str):
        """Handle incoming message from client"""
        
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "ping":
                # Respond to ping
                await self._send_event(websocket, {"type": "pong"})
                
            elif message_type == "get_metrics":
                # Send current metrics
                metrics = self.stream_manager.get_current_metrics()
                await self._send_event(websocket, {
                    "type": "metrics",
                    "data": metrics
                })
                
            elif message_type == "get_history":
                # Send event history
                limit = data.get("limit", 50)
                event_type = data.get("event_type")
                
                history = await self.stream_manager.get_event_history(
                    event_type=StreamEventType(event_type) if event_type else None,
                    limit=limit
                )
                
                await self._send_event(websocket, {
                    "type": "history",
                    "data": [self._serialize_event(e) for e in history]
                })
                
            elif message_type == "control":
                # Handle stream control commands
                await self._handle_control_command(websocket, session_id, data)
                
        except json.JSONDecodeError:
            await self._send_event(websocket, {
                "type": "error",
                "message": "Invalid JSON"
            })
        except Exception as e:
            await self._send_event(websocket, {
                "type": "error",
                "message": str(e)
            })
    
    async def _handle_control_command(self, websocket, session_id: str, data: Dict):
        """Handle stream control commands"""
        
        command = data.get("command")
        
        if command == "pause":
            # Implement pause logic
            pass
        elif command == "resume":
            # Implement resume logic
            pass
        elif command == "set_filter":
            # Implement event filtering
            filters = data.get("filters", {})
            # Store filters in connection state
            self.connection_states[session_id]["filters"] = filters
    
    def _serialize_event(self, event: StreamEvent) -> Dict[str, Any]:
        """Serialize event for WebSocket transmission"""
        
        return {
            "type": event.type.value,
            "data": event.data,
            "timestamp": event.timestamp.isoformat(),
            "sequence": event.sequence_number,
            "metadata": event.metadata
        }
    
    async def _send_event(self, websocket, data: Dict[str, Any]):
        """Send event data over WebSocket"""
        
        await websocket.send(json.dumps(data))


# Convenience functions for creating specialized streams

async def create_consciousness_stream(
    prompt: str,
    context: Any,
    generator: Any,
    consciousness_monitor: Any,
    emotion_tracker: Any
) -> AsyncGenerator[StreamEvent, None]:
    """Create a complete consciousness stream with all components"""
    
    manager = ConsciousnessStreamManager(
        enable_smoothing=True,
        enable_interpolation=True
    )
    
    async for event in manager.stream_response(
        generator.generate_with_consciousness_stream(prompt, context),
        consciousness_monitor=consciousness_monitor,
        emotion_tracker=emotion_tracker
    ):
        yield event


async def create_simple_text_stream(
    generator: Any,
    prompt: str,
    context: Any
) -> AsyncGenerator[str, None]:
    """Create a simple text-only stream"""
    
    manager = ConsciousnessStreamManager()
    
    async for event in manager.stream_response(
        generator.generate_with_consciousness_stream(prompt, context)
    ):
        if event.type == StreamEventType.TEXT:
            yield event.data


# Logging setup
import logging
logger = logging.getLogger(__name__)