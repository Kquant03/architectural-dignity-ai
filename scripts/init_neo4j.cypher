// Neo4j Database Setup for Consciousness AI Memory Network
// This creates the graph structure for associative memory

// =====================================================
// CONSTRAINTS AND INDEXES
// =====================================================

// Unique constraints
CREATE CONSTRAINT memory_id_unique IF NOT EXISTS
FOR (m:Memory) REQUIRE m.id IS UNIQUE;

CREATE CONSTRAINT user_id_unique IF NOT EXISTS
FOR (u:User) REQUIRE u.id IS UNIQUE;

CREATE CONSTRAINT session_id_unique IF NOT EXISTS
FOR (s:Session) REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT concept_name_unique IF NOT EXISTS
FOR (c:Concept) REQUIRE c.name IS UNIQUE;

// Indexes for performance
CREATE INDEX memory_timestamp IF NOT EXISTS
FOR (m:Memory) ON (m.timestamp);

CREATE INDEX memory_importance IF NOT EXISTS
FOR (m:Memory) ON (m.importance);

CREATE INDEX memory_type IF NOT EXISTS
FOR (m:Memory) ON (m.type);

CREATE INDEX emotion_valence IF NOT EXISTS
FOR (e:Emotion) ON (e.valence);

CREATE INDEX concept_activation IF NOT EXISTS
FOR (c:Concept) ON (c.activation);

// Full-text search indexes
CREATE FULLTEXT INDEX memory_content_search IF NOT EXISTS
FOR (m:Memory) ON EACH [m.content];

CREATE FULLTEXT INDEX concept_description_search IF NOT EXISTS
FOR (c:Concept) ON EACH [c.description];

// =====================================================
// NODE TYPES
// =====================================================

// Memory Node - Core memory unit
// Properties:
// - id: UUID
// - content: Text content of memory
// - type: episodic, semantic, procedural
// - importance: 0.0-1.0
// - emotional_valence: -1.0 to 1.0
// - timestamp: DateTime
// - access_count: Integer
// - embedding: List<Float> (vector representation)

// User Node - Represents a user
// Properties:
// - id: UUID
// - username: String
// - created_at: DateTime

// Session Node - Conversation session
// Properties:
// - id: UUID
// - created_at: DateTime
// - emotional_summary: Map

// Concept Node - Abstract concepts extracted from memories
// Properties:
// - name: String
// - description: String
// - activation: Float (current activation level)
// - frequency: Integer (how often referenced)

// Emotion Node - Emotional states
// Properties:
// - name: String (one of 27 Berkeley emotions)
// - valence: Float
// - arousal: Float
// - dominance: Float

// Context Node - Contextual information
// Properties:
// - id: UUID
// - type: temporal, spatial, social, task
// - description: String

// =====================================================
// RELATIONSHIP TYPES
// =====================================================

// REMEMBERED_BY - Memory belongs to user
// Properties:
// - strength: Float (0.0-1.0)

// IN_SESSION - Memory created in session
// Properties:
// - turn_number: Integer

// ASSOCIATES_WITH - Memory associates with another memory
// Properties:
// - strength: Float (0.0-1.0)
// - type: semantic, temporal, emotional, causal
// - bidirectional: Boolean

// EVOKES - Memory evokes emotion
// Properties:
// - intensity: Float (0.0-1.0)
// - timestamp: DateTime

// CONTAINS_CONCEPT - Memory contains concept
// Properties:
// - relevance: Float (0.0-1.0)
// - extraction_confidence: Float

// RELATED_TO - Concept relates to another concept
// Properties:
// - relation_type: String (is_a, part_of, causes, etc.)
// - strength: Float

// ACTIVATED_BY - Concept activated by memory
// Properties:
// - activation_strength: Float
// - timestamp: DateTime

// IN_CONTEXT - Memory occurred in context
// Properties:
// - relevance: Float

// TRIGGERS - Memory triggers another memory
// Properties:
// - trigger_type: String (reminder, association, emotion)
// - delay_ms: Integer

// =====================================================
// INITIAL DATA STRUCTURES
// =====================================================

// Create emotion nodes for Berkeley 27
WITH [
  {name: 'admiration', valence: 0.7, arousal: 0.5, dominance: 0.4},
  {name: 'adoration', valence: 0.8, arousal: 0.6, dominance: 0.3},
  {name: 'aesthetic_appreciation', valence: 0.6, arousal: 0.3, dominance: 0.5},
  {name: 'amusement', valence: 0.8, arousal: 0.7, dominance: 0.6},
  {name: 'anger', valence: -0.7, arousal: 0.8, dominance: 0.7},
  {name: 'anxiety', valence: -0.6, arousal: 0.7, dominance: 0.2},
  {name: 'awe', valence: 0.5, arousal: 0.6, dominance: 0.2},
  {name: 'awkwardness', valence: -0.3, arousal: 0.5, dominance: 0.2},
  {name: 'boredom', valence: -0.2, arousal: 0.1, dominance: 0.3},
  {name: 'calmness', valence: 0.4, arousal: 0.1, dominance: 0.5},
  {name: 'confusion', valence: -0.3, arousal: 0.4, dominance: 0.2},
  {name: 'craving', valence: 0.1, arousal: 0.6, dominance: 0.4},
  {name: 'disgust', valence: -0.8, arousal: 0.5, dominance: 0.6},
  {name: 'empathic_pain', valence: -0.6, arousal: 0.4, dominance: 0.3},
  {name: 'entrancement', valence: 0.5, arousal: 0.4, dominance: 0.2},
  {name: 'excitement', valence: 0.8, arousal: 0.9, dominance: 0.6},
  {name: 'fear', valence: -0.8, arousal: 0.8, dominance: 0.1},
  {name: 'horror', valence: -0.9, arousal: 0.9, dominance: 0.1},
  {name: 'interest', valence: 0.4, arousal: 0.6, dominance: 0.5},
  {name: 'joy', valence: 0.9, arousal: 0.7, dominance: 0.6},
  {name: 'nostalgia', valence: 0.2, arousal: 0.3, dominance: 0.4},
  {name: 'relief', valence: 0.6, arousal: 0.2, dominance: 0.5},
  {name: 'romance', valence: 0.8, arousal: 0.5, dominance: 0.4},
  {name: 'sadness', valence: -0.7, arousal: 0.2, dominance: 0.2},
  {name: 'satisfaction', valence: 0.7, arousal: 0.3, dominance: 0.6},
  {name: 'sexual_desire', valence: 0.6, arousal: 0.8, dominance: 0.5},
  {name: 'surprise', valence: 0.1, arousal: 0.8, dominance: 0.3}
] AS emotion
MERGE (e:Emotion {name: emotion.name})
SET e.valence = emotion.valence,
    e.arousal = emotion.arousal,
    e.dominance = emotion.dominance;

// Create core concept nodes
WITH [
  {name: 'self', description: 'The conscious entity experiencing'},
  {name: 'other', description: 'Entities outside of self'},
  {name: 'time', description: 'Temporal experience and flow'},
  {name: 'space', description: 'Spatial awareness and relations'},
  {name: 'causality', description: 'Cause and effect relationships'},
  {name: 'intention', description: 'Goals and purposeful action'},
  {name: 'emotion', description: 'Affective experiences'},
  {name: 'thought', description: 'Cognitive processes'},
  {name: 'memory', description: 'Stored experiences and knowledge'},
  {name: 'attention', description: 'Focused awareness'},
  {name: 'consciousness', description: 'Aware experience itself'}
] AS concept
MERGE (c:Concept {name: concept.name})
SET c.description = concept.description,
    c.activation = 0.5,
    c.frequency = 0;

// Create concept relationships
MATCH (self:Concept {name: 'self'})
MATCH (consciousness:Concept {name: 'consciousness'})
MERGE (self)-[:RELATED_TO {relation_type: 'enables', strength: 0.9}]->(consciousness);

MATCH (attention:Concept {name: 'attention'})
MATCH (consciousness:Concept {name: 'consciousness'})
MERGE (attention)-[:RELATED_TO {relation_type: 'component_of', strength: 0.8}]->(consciousness);

MATCH (memory:Concept {name: 'memory'})
MATCH (self:Concept {name: 'self'})
MERGE (memory)-[:RELATED_TO {relation_type: 'constructs', strength: 0.7}]->(self);

// Create context type nodes
WITH [
  {type: 'temporal', description: 'Time-based context'},
  {type: 'spatial', description: 'Location-based context'},
  {type: 'social', description: 'Social interaction context'},
  {type: 'task', description: 'Task or goal context'},
  {type: 'emotional', description: 'Emotional state context'},
  {type: 'cognitive', description: 'Cognitive state context'}
] AS ctx
MERGE (c:ContextType {type: ctx.type})
SET c.description = ctx.description;

// =====================================================
// FUNCTIONS AND PROCEDURES
// =====================================================

// Procedure to create memory with full relationships
CALL apoc.custom.asProcedure(
  'createMemory',
  'MERGE (m:Memory {id: $memory_id})
   SET m.content = $content,
       m.type = $type,
       m.importance = $importance,
       m.emotional_valence = $emotional_valence,
       m.timestamp = datetime($timestamp),
       m.access_count = 0,
       m.embedding = $embedding
   
   WITH m
   MATCH (u:User {id: $user_id})
   MERGE (m)-[:REMEMBERED_BY {strength: $importance}]->(u)
   
   WITH m, u
   MATCH (s:Session {id: $session_id})
   MERGE (m)-[:IN_SESSION {turn_number: $turn_number}]->(s)
   
   RETURN m',
  'WRITE',
  [['m', 'NODE']],
  [['memory_id', 'STRING'], ['content', 'STRING'], ['type', 'STRING'],
   ['importance', 'FLOAT'], ['emotional_valence', 'FLOAT'], 
   ['timestamp', 'STRING'], ['embedding', 'LIST OF FLOAT'],
   ['user_id', 'STRING'], ['session_id', 'STRING'], ['turn_number', 'INTEGER']]
);

// Procedure to find associated memories
CALL apoc.custom.asProcedure(
  'findAssociatedMemories',
  'MATCH (m:Memory {id: $memory_id})
   MATCH (m)-[a:ASSOCIATES_WITH]-(associated:Memory)
   RETURN associated, a.strength AS strength, a.type AS association_type
   ORDER BY a.strength DESC
   LIMIT $limit',
  'READ',
  [['associated', 'NODE'], ['strength', 'FLOAT'], ['association_type', 'STRING']],
  [['memory_id', 'STRING'], ['limit', 'INTEGER']]
);

// Procedure to activate concept network
CALL apoc.custom.asProcedure(
  'activateConceptNetwork',
  'MATCH (m:Memory {id: $memory_id})
   MATCH (m)-[r:CONTAINS_CONCEPT]->(c:Concept)
   SET c.activation = c.activation * 0.9 + r.relevance * $activation_strength * 0.1,
       c.frequency = c.frequency + 1
   WITH c
   MATCH (c)-[rel:RELATED_TO]-(related:Concept)
   SET related.activation = related.activation * 0.9 + rel.strength * c.activation * 0.1
   RETURN c, related',
  'WRITE',
  [['c', 'NODE'], ['related', 'NODE']],
  [['memory_id', 'STRING'], ['activation_strength', 'FLOAT']]
);

// Procedure for memory consolidation
CALL apoc.custom.asProcedure(
  'consolidateMemories',
  'MATCH (u:User {id: $user_id})
   MATCH (m:Memory)-[:REMEMBERED_BY]->(u)
   WHERE m.timestamp > datetime() - duration({days: 1})
   AND m.type = "episodic"
   WITH m
   ORDER BY m.importance DESC
   LIMIT 100
   
   WITH collect(m) AS memories
   UNWIND memories AS m1
   UNWIND memories AS m2
   WHERE id(m1) < id(m2)
   AND apoc.algo.cosineSimilarity(m1.embedding, m2.embedding) > $similarity_threshold
   
   MERGE (m1)-[a:ASSOCIATES_WITH {type: "semantic"}]-(m2)
   SET a.strength = apoc.algo.cosineSimilarity(m1.embedding, m2.embedding)
   
   RETURN count(a) AS associations_created',
  'WRITE',
  [['associations_created', 'INTEGER']],
  [['user_id', 'STRING'], ['similarity_threshold', 'FLOAT']]
);

// =====================================================
// EXAMPLE QUERIES
// =====================================================

// Find memories that evoke similar emotions
// MATCH (m1:Memory {id: $memory_id})
// MATCH (m1)-[:EVOKES]->(e1:Emotion)
// MATCH (m2:Memory)-[:EVOKES]->(e2:Emotion)
// WHERE m1 <> m2 
// AND abs(e1.valence - e2.valence) < 0.2
// AND abs(e1.arousal - e2.arousal) < 0.2
// RETURN m2, e2
// ORDER BY m2.importance DESC
// LIMIT 10;

// Trace activation through concept network
// MATCH path = (m:Memory {id: $memory_id})-[:CONTAINS_CONCEPT|:RELATED_TO*1..3]-(c:Concept)
// WHERE c.activation > 0.5
// RETURN path;

// Find memories in similar contexts
// MATCH (m1:Memory {id: $memory_id})-[:IN_CONTEXT]->(ctx1:Context)
// MATCH (m2:Memory)-[:IN_CONTEXT]->(ctx2:Context)
// WHERE m1 <> m2 AND ctx1.type = ctx2.type
// RETURN m2, ctx2
// ORDER BY m2.timestamp DESC
// LIMIT 20;

// Get emotional journey for a session
// MATCH (s:Session {id: $session_id})
// MATCH (m:Memory)-[:IN_SESSION]->(s)
// MATCH (m)-[e:EVOKES]->(emotion:Emotion)
// RETURN m.timestamp AS time, emotion.name AS emotion, 
//        emotion.valence AS valence, e.intensity AS intensity
// ORDER BY m.timestamp;

// =====================================================
// MAINTENANCE QUERIES
// =====================================================

// Decay concept activation over time
// MATCH (c:Concept)
// WHERE c.activation > 0.1
// SET c.activation = c.activation * 0.95;

// Prune weak associations
// MATCH ()-[a:ASSOCIATES_WITH]->()
// WHERE a.strength < 0.1
// DELETE a;

// Update memory importance based on access patterns
// MATCH (m:Memory)
// WHERE m.access_count > 10
// SET m.importance = m.importance * 0.9 + 0.1;