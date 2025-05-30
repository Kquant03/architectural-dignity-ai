"""
Mem0 Integration Module
Provides advanced memory capabilities with hybrid storage, semantic search,
and consciousness-aware memory formation and retrieval.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import hashlib

# Mem0 imports
from mem0 import Memory
from mem0.configs import MemoryConfig
from mem0.memory.graph_memory import GraphMemory
from mem0.memory.vector_memory import VectorMemory
from mem0.memory.key_value_memory import KeyValueMemory

# Database and embedding imports
import asyncpg
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Represents a single memory entry"""
    id: str
    user_id: str
    session_id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    emotional_valence: float = 0.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    memory_type: str = "episodic"  # episodic, semantic, procedural
    associations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        # Convert datetime to ISO format
        data['timestamp'] = self.timestamp.isoformat()
        if self.last_accessed:
            data['last_accessed'] = self.last_accessed.isoformat()
        # Don't include embedding in dict (stored separately)
        data.pop('embedding', None)
        return data


@dataclass
class MemorySearchResult:
    """Result from memory search"""
    memory: MemoryEntry
    score: float
    relevance_type: str  # semantic, temporal, emotional, associative
    context: Optional[Dict[str, Any]] = None


class ConsciousnessAwareMemory:
    """
    Advanced memory system integrating Mem0 with consciousness-aware
    memory formation, consolidation, and retrieval.
    """
    
    def __init__(
        self,
        db_config: Dict[str, str],
        mem0_config: Optional[Dict[str, Any]] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_graph_memory: bool = True,
        enable_vector_memory: bool = True,
        enable_key_value_memory: bool = True,
        consolidation_interval: int = 300,  # 5 minutes
        max_memory_size: int = 1000000
    ):
        self.db_config = db_config
        self.embedding_model_name = embedding_model
        self.enable_graph_memory = enable_graph_memory
        self.enable_vector_memory = enable_vector_memory
        self.enable_key_value_memory = enable_key_value_memory
        self.consolidation_interval = consolidation_interval
        self.max_memory_size = max_memory_size
        
        # Initialize components
        self.db_pool = None
        self.embedding_model = SentenceTransformer(embedding_model)
        self.mem0 = None
        self.memory_graph = nx.DiGraph() if enable_graph_memory else None
        
        # Memory buffers
        self.working_memory = deque(maxlen=50)
        self.consolidation_buffer = []
        self.attention_weights = {}
        
        # Consciousness integration
        self.consciousness_state = None
        self.emotional_context = {}
        
        # Memory statistics
        self.memory_stats = {
            "total_memories": 0,
            "episodic_count": 0,
            "semantic_count": 0,
            "consolidations": 0,
            "retrievals": 0
        }
        
        # Initialize Mem0 configuration
        self.mem0_config = self._create_mem0_config(mem0_config)
    
    async def initialize(self):
        """Initialize database connection and Mem0"""
        # Create database pool
        self.db_pool = await asyncpg.create_pool(**self.db_config)
        
        # Initialize database schema
        await self._initialize_database()
        
        # Initialize Mem0
        self.mem0 = Memory.from_config(config_dict=self.mem0_config)
        
        # Start background tasks
        asyncio.create_task(self._consolidation_loop())
        
        logger.info("ConsciousnessAwareMemory initialized successfully")
    
    async def _initialize_database(self):
        """Initialize database schema"""
        async with self.db_pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create memories table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(255) NOT NULL,
                    session_id VARCHAR(255),
                    content TEXT NOT NULL,
                    embedding vector(384),  -- Dimension for all-MiniLM-L6-v2
                    metadata JSONB DEFAULT '{}',
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    importance FLOAT DEFAULT 0.5,
                    emotional_valence FLOAT DEFAULT 0.0,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMPTZ,
                    memory_type VARCHAR(50) DEFAULT 'episodic',
                    associations TEXT[] DEFAULT '{}',
                    
                    -- Indexes
                    INDEX idx_user_session (user_id, session_id),
                    INDEX idx_timestamp (timestamp DESC),
                    INDEX idx_importance (importance DESC),
                    INDEX idx_memory_type (memory_type)
                )
            """)
            
            # Create HNSW index for vector similarity search
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS memories_embedding_idx 
                ON memories USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)
            
            # Create memory associations table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_associations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source_memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
                    target_memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
                    association_type VARCHAR(50),
                    strength FLOAT DEFAULT 0.5,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    
                    UNIQUE(source_memory_id, target_memory_id)
                )
            """)
            
            # Create conversations table for context
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id VARCHAR(255) NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    turn_number INTEGER,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                )
            """)
            
            logger.info("Database schema initialized")
    
    def _create_mem0_config(self, custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create Mem0 configuration"""
        config = {
            "version": "v1.0",
            "llm": {
                "provider": "openai",  # Will be overridden by our API
                "config": {
                    "model": "gpt-4",
                    "temperature": 0.1,
                    "max_tokens": 2000
                }
            },
            "history_db_path": "./data/history.db",
            "embedding_model": {
                "provider": "huggingface",
                "config": {
                    "model": self.embedding_model_name
                }
            }
        }
        
        # Add storage configurations
        if self.enable_vector_memory:
            config["vector_store"] = {
                "provider": "postgresql",
                "config": self.db_config
            }
        
        if self.enable_graph_memory:
            config["graph_store"] = {
                "provider": "neo4j",  # Or custom implementation
                "config": {
                    "url": os.environ.get("NEO4J_URL", "bolt://localhost:7687"),
                    "username": os.environ.get("NEO4J_USERNAME", "neo4j"),
                    "password": os.environ.get("NEO4J_PASSWORD", "password")
                }
            }
        
        # Merge with custom config
        if custom_config:
            config.update(custom_config)
        
        return config
    
    async def add_memory(
        self,
        content: Union[str, Dict[str, Any]],
        user_id: str,
        session_id: str,
        memory_type: str = "episodic",
        importance: Optional[float] = None,
        emotional_valence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """Add a new memory with consciousness-aware processing"""
        
        # Extract content if dict
        if isinstance(content, dict):
            memory_content = json.dumps(content)
            metadata = metadata or {}
            metadata.update(content)
        else:
            memory_content = content
        
        # Generate embedding
        embedding = self.embedding_model.encode(memory_content)
        
        # Calculate importance if not provided
        if importance is None:
            importance = await self._calculate_importance(memory_content, embedding)
        
        # Detect emotional valence if not provided
        if emotional_valence is None and self.emotional_context:
            emotional_valence = self._extract_emotional_valence(memory_content)
        
        # Create memory entry
        memory = MemoryEntry(
            id=hashlib.md5(f"{user_id}:{session_id}:{memory_content}:{datetime.now()}".encode()).hexdigest(),
            user_id=user_id,
            session_id=session_id,
            content=memory_content,
            embedding=embedding,
            metadata=metadata or {},
            importance=importance,
            emotional_valence=emotional_valence or 0.0,
            memory_type=memory_type
        )
        
        # Store in database
        await self._store_memory(memory)
        
        # Add to Mem0
        if self.mem0:
            mem0_result = self.mem0.add(
                messages=[{"role": "user", "content": memory_content}],
                user_id=user_id,
                metadata={
                    "session_id": session_id,
                    "importance": importance,
                    "emotional_valence": emotional_valence,
                    "memory_type": memory_type
                }
            )
            memory.metadata["mem0_id"] = mem0_result.get("id")
        
        # Add to working memory
        self.working_memory.append(memory)
        
        # Update memory graph
        if self.memory_graph is not None:
            await self._update_memory_graph(memory)
        
        # Update statistics
        self.memory_stats["total_memories"] += 1
        self.memory_stats[f"{memory_type}_count"] += 1
        
        logger.debug(f"Added {memory_type} memory for user {user_id}")
        
        return memory
    
    async def search_memories(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        top_k: int = 10,
        memory_types: Optional[List[str]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        min_importance: float = 0.0,
        include_associations: bool = True
    ) -> List[MemorySearchResult]:
        """Search memories with multi-modal retrieval"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Parallel search strategies
        search_tasks = []
        
        # Semantic search
        search_tasks.append(
            self._semantic_search(query_embedding, user_id, session_id, top_k * 2)
        )
        
        # Temporal search if time range specified
        if time_range:
            search_tasks.append(
                self._temporal_search(user_id, session_id, time_range, top_k)
            )
        
        # Emotional resonance search if emotional context exists
        if self.emotional_context:
            search_tasks.append(
                self._emotional_search(user_id, session_id, self.emotional_context, top_k)
            )
        
        # Graph-based associative search
        if include_associations and self.memory_graph:
            search_tasks.append(
                self._associative_search(query, user_id, top_k)
            )
        
        # Execute all searches in parallel
        search_results = await asyncio.gather(*search_tasks)
        
        # Merge and rank results
        merged_results = await self._merge_search_results(search_results, top_k)
        
        # Update access counts
        for result in merged_results:
            await self._update_memory_access(result.memory.id)
        
        # Update statistics
        self.memory_stats["retrievals"] += 1
        
        return merged_results
    
    async def _semantic_search(
        self,
        query_embedding: np.ndarray,
        user_id: str,
        session_id: Optional[str],
        limit: int
    ) -> List[MemorySearchResult]:
        """Perform semantic similarity search"""
        
        async with self.db_pool.acquire() as conn:
            # Build query
            base_query = """
                SELECT 
                    id, user_id, session_id, content, metadata,
                    timestamp, importance, emotional_valence,
                    access_count, last_accessed, memory_type, associations,
                    1 - (embedding <=> $1::vector) as similarity
                FROM memories
                WHERE user_id = $2
            """
            
            params = [query_embedding.tolist(), user_id]
            
            if session_id:
                base_query += " AND session_id = $3"
                params.append(session_id)
                
            base_query += " ORDER BY similarity DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)
            
            # Execute query
            rows = await conn.fetch(base_query, *params)
            
            # Convert to MemorySearchResult
            results = []
            for row in rows:
                memory = MemoryEntry(
                    id=str(row['id']),
                    user_id=row['user_id'],
                    session_id=row['session_id'],
                    content=row['content'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    timestamp=row['timestamp'],
                    importance=row['importance'],
                    emotional_valence=row['emotional_valence'],
                    access_count=row['access_count'],
                    last_accessed=row['last_accessed'],
                    memory_type=row['memory_type'],
                    associations=row['associations'] or []
                )
                
                results.append(MemorySearchResult(
                    memory=memory,
                    score=row['similarity'],
                    relevance_type="semantic"
                ))
            
            return results
    
    async def _temporal_search(
        self,
        user_id: str,
        session_id: Optional[str],
        time_range: Tuple[datetime, datetime],
        limit: int
    ) -> List[MemorySearchResult]:
        """Search memories within a time range"""
        
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT * FROM memories
                WHERE user_id = $1
                AND timestamp BETWEEN $2 AND $3
            """
            
            params = [user_id, time_range[0], time_range[1]]
            
            if session_id:
                query += " AND session_id = $4"
                params.append(session_id)
            
            query += " ORDER BY timestamp DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            
            results = []
            for row in rows:
                memory = MemoryEntry(
                    id=str(row['id']),
                    user_id=row['user_id'],
                    session_id=row['session_id'],
                    content=row['content'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    timestamp=row['timestamp'],
                    importance=row['importance'],
                    emotional_valence=row['emotional_valence'],
                    access_count=row['access_count'],
                    last_accessed=row['last_accessed'],
                    memory_type=row['memory_type'],
                    associations=row['associations'] or []
                )
                
                # Calculate temporal relevance score
                time_score = 1.0 - (datetime.now() - memory.timestamp).total_seconds() / (7 * 24 * 3600)
                time_score = max(0.0, min(1.0, time_score))
                
                results.append(MemorySearchResult(
                    memory=memory,
                    score=time_score,
                    relevance_type="temporal"
                ))
            
            return results
    
    async def _emotional_search(
        self,
        user_id: str,
        session_id: Optional[str],
        emotional_context: Dict[str, float],
        limit: int
    ) -> List[MemorySearchResult]:
        """Search memories by emotional resonance"""
        
        # Calculate target emotional valence
        target_valence = sum(v for v in emotional_context.values()) / len(emotional_context)
        
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT *,
                ABS(emotional_valence - $1) as valence_distance
                FROM memories
                WHERE user_id = $2
            """
            
            params = [target_valence, user_id]
            
            if session_id:
                query += " AND session_id = $3"
                params.append(session_id)
            
            query += " ORDER BY valence_distance ASC LIMIT $" + str(len(params) + 1)
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            
            results = []
            for row in rows:
                memory = MemoryEntry(
                    id=str(row['id']),
                    user_id=row['user_id'],
                    session_id=row['session_id'],
                    content=row['content'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    timestamp=row['timestamp'],
                    importance=row['importance'],
                    emotional_valence=row['emotional_valence'],
                    access_count=row['access_count'],
                    last_accessed=row['last_accessed'],
                    memory_type=row['memory_type'],
                    associations=row['associations'] or []
                )
                
                # Calculate emotional resonance score
                resonance_score = 1.0 - row['valence_distance']
                
                results.append(MemorySearchResult(
                    memory=memory,
                    score=resonance_score,
                    relevance_type="emotional",
                    context={"target_valence": target_valence}
                ))
            
            return results
    
    async def _associative_search(
        self,
        query: str,
        user_id: str,
        limit: int
    ) -> List[MemorySearchResult]:
        """Search using memory associations graph"""
        
        if not self.memory_graph or self.memory_graph.number_of_nodes() == 0:
            return []
        
        # Find seed memories semantically similar to query
        query_embedding = self.embedding_model.encode(query)
        seed_results = await self._semantic_search(query_embedding, user_id, None, 5)
        
        if not seed_results:
            return []
        
        associative_results = []
        
        # Explore associations from seed memories
        for seed in seed_results[:3]:  # Top 3 seeds
            if seed.memory.id in self.memory_graph:
                # Get associated memories using PageRank-style traversal
                try:
                    associated_nodes = nx.single_source_shortest_path_length(
                        self.memory_graph,
                        seed.memory.id,
                        cutoff=2  # 2-hop associations
                    )
                    
                    for node_id, distance in associated_nodes.items():
                        if node_id != seed.memory.id:
                            # Retrieve full memory
                            memory = await self._get_memory_by_id(node_id)
                            if memory and memory.user_id == user_id:
                                # Calculate association score
                                assoc_score = seed.score * (1.0 / (distance + 1))
                                
                                associative_results.append(MemorySearchResult(
                                    memory=memory,
                                    score=assoc_score,
                                    relevance_type="associative",
                                    context={
                                        "seed_memory": seed.memory.id,
                                        "distance": distance
                                    }
                                ))
                except nx.NetworkXError:
                    continue
        
        # Sort by score and limit
        associative_results.sort(key=lambda x: x.score, reverse=True)
        return associative_results[:limit]
    
    async def _merge_search_results(
        self,
        result_sets: List[List[MemorySearchResult]],
        top_k: int
    ) -> List[MemorySearchResult]:
        """Merge and rank results from different search strategies"""
        
        # Combine all results
        all_results = []
        for results in result_sets:
            all_results.extend(results)
        
        # Group by memory ID to handle duplicates
        grouped_results = defaultdict(list)
        for result in all_results:
            grouped_results[result.memory.id].append(result)
        
        # Merge scores for duplicate memories
        final_results = []
        for memory_id, results in grouped_results.items():
            # Calculate combined score
            scores_by_type = defaultdict(float)
            for r in results:
                scores_by_type[r.relevance_type] = max(scores_by_type[r.relevance_type], r.score)
            
            # Weighted combination
            weights = {
                "semantic": 0.4,
                "temporal": 0.2,
                "emotional": 0.2,
                "associative": 0.2
            }
            
            combined_score = sum(
                scores_by_type[rtype] * weight
                for rtype, weight in weights.items()
            )
            
            # Use the first result as template
            merged_result = MemorySearchResult(
                memory=results[0].memory,
                score=combined_score,
                relevance_type="combined",
                context={
                    "component_scores": dict(scores_by_type),
                    "relevance_types": [r.relevance_type for r in results]
                }
            )
            
            final_results.append(merged_result)
        
        # Sort by combined score
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply consciousness modulation
        if self.consciousness_state:
            final_results = await self._apply_consciousness_modulation(final_results)
        
        return final_results[:top_k]
    
    async def _apply_consciousness_modulation(
        self,
        results: List[MemorySearchResult]
    ) -> List[MemorySearchResult]:
        """Modulate memory retrieval based on consciousness state"""
        
        if not self.consciousness_state:
            return results
        
        # Boost recent memories if high awareness
        awareness_level = self.consciousness_state.get("awareness_level", 0.5)
        
        for result in results:
            # Recency boost
            recency_factor = 1.0
            if awareness_level > 0.7:
                time_diff = (datetime.now() - result.memory.timestamp).total_seconds()
                recency_factor = 1.0 + (0.2 * np.exp(-time_diff / 3600))  # Exponential decay
            
            # Importance boost based on cognitive load
            cognitive_load = self.consciousness_state.get("cognitive_load", 0.5)
            importance_factor = 1.0
            if cognitive_load > 0.7:
                # Under high load, prefer important memories
                importance_factor = 1.0 + (0.3 * result.memory.importance)
            
            # Apply modulation
            result.score *= recency_factor * importance_factor
        
        # Re-sort after modulation
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    async def consolidate_memories(
        self,
        user_id: str,
        session_id: Optional[str] = None
    ) -> int:
        """Consolidate episodic memories into semantic memories"""
        
        consolidation_count = 0
        
        # Get recent episodic memories
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT * FROM memories
                WHERE user_id = $1
                AND memory_type = 'episodic'
                AND timestamp < $2
                ORDER BY importance DESC, timestamp DESC
                LIMIT 100
            """
            
            params = [user_id, datetime.now() - timedelta(seconds=self.consolidation_interval)]
            
            if session_id:
                query = query.replace("WHERE user_id = $1", "WHERE user_id = $1 AND session_id = $2")
                params.insert(1, session_id)
            
            episodic_memories = await conn.fetch(query, *params)
        
        # Group similar memories
        memory_clusters = await self._cluster_memories(episodic_memories)
        
        # Consolidate each cluster
        for cluster in memory_clusters:
            if len(cluster) >= 3:  # Minimum cluster size for consolidation
                semantic_memory = await self._create_semantic_memory(cluster, user_id)
                
                # Store semantic memory
                await self.add_memory(
                    content=semantic_memory["content"],
                    user_id=user_id,
                    session_id=session_id,
                    memory_type="semantic",
                    importance=semantic_memory["importance"],
                    metadata=semantic_memory["metadata"]
                )
                
                # Mark episodic memories as consolidated
                for memory in cluster:
                    await self._mark_as_consolidated(memory['id'])
                
                consolidation_count += 1
        
        # Update statistics
        self.memory_stats["consolidations"] += consolidation_count
        
        logger.info(f"Consolidated {consolidation_count} memory clusters for user {user_id}")
        
        return consolidation_count
    
    async def _cluster_memories(
        self,
        memories: List[asyncpg.Record]
    ) -> List[List[asyncpg.Record]]:
        """Cluster similar memories for consolidation"""
        
        if len(memories) < 2:
            return []
        
        # Extract embeddings
        embeddings = []
        valid_memories = []
        
        for memory in memories:
            if memory['embedding']:
                embeddings.append(memory['embedding'])
                valid_memories.append(memory)
        
        if len(embeddings) < 2:
            return []
        
        # Calculate similarity matrix
        embeddings_array = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        # Simple clustering based on similarity threshold
        clusters = []
        clustered = set()
        threshold = 0.7
        
        for i in range(len(valid_memories)):
            if i in clustered:
                continue
            
            cluster = [valid_memories[i]]
            clustered.add(i)
            
            for j in range(i + 1, len(valid_memories)):
                if j not in clustered and similarity_matrix[i, j] > threshold:
                    cluster.append(valid_memories[j])
                    clustered.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    async def _create_semantic_memory(
        self,
        cluster: List[asyncpg.Record],
        user_id: str
    ) -> Dict[str, Any]:
        """Create a semantic memory from a cluster of episodic memories"""
        
        # Extract common themes
        contents = [m['content'] for m in cluster]
        combined_content = " ".join(contents)
        
        # Calculate aggregate importance
        avg_importance = np.mean([m['importance'] for m in cluster])
        max_importance = max(m['importance'] for m in cluster)
        importance = 0.7 * avg_importance + 0.3 * max_importance
        
        # Create semantic summary
        # In a real implementation, this would use an LLM
        semantic_content = f"Consolidated memory from {len(cluster)} related experiences: {combined_content[:200]}..."
        
        # Aggregate metadata
        metadata = {
            "source_memories": [str(m['id']) for m in cluster],
            "consolidation_time": datetime.now().isoformat(),
            "cluster_size": len(cluster),
            "time_span": (
                max(m['timestamp'] for m in cluster) -
                min(m['timestamp'] for m in cluster)
            ).total_seconds()
        }
        
        return {
            "content": semantic_content,
            "importance": importance,
            "metadata": metadata
        }
    
    async def _consolidation_loop(self):
        """Background task for periodic memory consolidation"""
        
        while True:
            try:
                await asyncio.sleep(self.consolidation_interval)
                
                # Get all active users
                async with self.db_pool.acquire() as conn:
                    users = await conn.fetch(
                        "SELECT DISTINCT user_id FROM memories WHERE timestamp > $1",
                        datetime.now() - timedelta(hours=24)
                    )
                
                # Consolidate for each user
                for user in users:
                    await self.consolidate_memories(user['user_id'])
                
            except Exception as e:
                logger.error(f"Error in consolidation loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def get_memory_summary(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get a summary of user's memories"""
        
        async with self.db_pool.acquire() as conn:
            # Base query conditions
            conditions = ["user_id = $1"]
            params = [user_id]
            
            if session_id:
                conditions.append("session_id = $2")
                params.append(session_id)
            
            if time_window:
                conditions.append(f"timestamp > ${len(params) + 1}")
                params.append(datetime.now() - time_window)
            
            where_clause = " AND ".join(conditions)
            
            # Get memory statistics
            stats_query = f"""
                SELECT 
                    COUNT(*) as total_count,
                    COUNT(DISTINCT session_id) as session_count,
                    AVG(importance) as avg_importance,
                    AVG(emotional_valence) as avg_valence,
                    SUM(access_count) as total_accesses,
                    COUNT(CASE WHEN memory_type = 'episodic' THEN 1 END) as episodic_count,
                    COUNT(CASE WHEN memory_type = 'semantic' THEN 1 END) as semantic_count,
                    MIN(timestamp) as earliest_memory,
                    MAX(timestamp) as latest_memory
                FROM memories
                WHERE {where_clause}
            """
            
            stats = await conn.fetchrow(stats_query, *params)
            
            # Get most important memories
            important_query = f"""
                SELECT content, importance, timestamp
                FROM memories
                WHERE {where_clause}
                ORDER BY importance DESC
                LIMIT 5
            """
            
            important_memories = await conn.fetch(important_query, *params)
            
            # Get emotional profile
            emotional_query = f"""
                SELECT 
                    CASE 
                        WHEN emotional_valence > 0.3 THEN 'positive'
                        WHEN emotional_valence < -0.3 THEN 'negative'
                        ELSE 'neutral'
                    END as emotion_category,
                    COUNT(*) as count
                FROM memories
                WHERE {where_clause}
                GROUP BY emotion_category
            """
            
            emotional_profile = await conn.fetch(emotional_query, *params)
            
        summary = {
            "user_id": user_id,
            "session_id": session_id,
            "statistics": {
                "total_memories": stats['total_count'],
                "unique_sessions": stats['session_count'],
                "average_importance": float(stats['avg_importance']) if stats['avg_importance'] else 0.0,
                "average_emotional_valence": float(stats['avg_valence']) if stats['avg_valence'] else 0.0,
                "total_retrievals": stats['total_accesses'],
                "episodic_memories": stats['episodic_count'],
                "semantic_memories": stats['semantic_count'],
                "memory_span": (stats['latest_memory'] - stats['earliest_memory']).days if stats['earliest_memory'] else 0
            },
            "important_memories": [
                {
                    "content": m['content'][:100] + "..." if len(m['content']) > 100 else m['content'],
                    "importance": float(m['importance']),
                    "timestamp": m['timestamp'].isoformat()
                }
                for m in important_memories
            ],
            "emotional_profile": {
                emotion['emotion_category']: emotion['count']
                for emotion in emotional_profile
            }
        }
        
        return summary
    
    # Helper methods
    
    async def _store_memory(self, memory: MemoryEntry):
        """Store memory in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO memories (
                    id, user_id, session_id, content, embedding, metadata,
                    timestamp, importance, emotional_valence, access_count,
                    last_accessed, memory_type, associations
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
                memory.id, memory.user_id, memory.session_id, memory.content,
                memory.embedding.tolist() if memory.embedding is not None else None,
                json.dumps(memory.metadata), memory.timestamp, memory.importance,
                memory.emotional_valence, memory.access_count, memory.last_accessed,
                memory.memory_type, memory.associations
            )
    
    async def _get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory by ID"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM memories WHERE id = $1",
                memory_id
            )
            
            if not row:
                return None
            
            return MemoryEntry(
                id=str(row['id']),
                user_id=row['user_id'],
                session_id=row['session_id'],
                content=row['content'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                timestamp=row['timestamp'],
                importance=row['importance'],
                emotional_valence=row['emotional_valence'],
                access_count=row['access_count'],
                last_accessed=row['last_accessed'],
                memory_type=row['memory_type'],
                associations=row['associations'] or []
            )
    
    async def _update_memory_access(self, memory_id: str):
        """Update memory access statistics"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE memories
                SET access_count = access_count + 1,
                    last_accessed = NOW()
                WHERE id = $1
            """, memory_id)
    
    async def _mark_as_consolidated(self, memory_id: str):
        """Mark a memory as consolidated"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE memories
                SET metadata = jsonb_set(
                    COALESCE(metadata, '{}'),
                    '{consolidated}',
                    'true'
                )
                WHERE id = $1
            """, memory_id)
    
    async def _calculate_importance(self, content: str, embedding: np.ndarray) -> float:
        """Calculate importance score for a memory"""
        
        # Base importance on content length (normalized)
        length_score = min(1.0, len(content.split()) / 50)
        
        # Check for important markers
        important_markers = [
            "important", "remember", "never forget", "crucial",
            "significant", "milestone", "breakthrough", "realization"
        ]
        
        marker_score = sum(
            0.1 for marker in important_markers
            if marker in content.lower()
        )
        marker_score = min(1.0, marker_score)
        
        # Emotional intensity contributes to importance
        emotion_score = abs(self._extract_emotional_valence(content))
        
        # Combine scores
        importance = (
            0.3 * length_score +
            0.4 * marker_score +
            0.3 * emotion_score
        )
        
        return min(1.0, max(0.0, importance))
    
    def _extract_emotional_valence(self, content: str) -> float:
        """Extract emotional valence from content"""
        
        # Simple keyword-based approach
        positive_keywords = [
            "happy", "joy", "love", "excited", "wonderful", "amazing",
            "great", "fantastic", "beautiful", "grateful", "blessed"
        ]
        
        negative_keywords = [
            "sad", "angry", "frustrated", "disappointed", "hurt", "pain",
            "terrible", "awful", "horrible", "afraid", "worried"
        ]
        
        content_lower = content.lower()
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in content_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in content_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        valence = (positive_count - negative_count) / (positive_count + negative_count)
        
        return valence
    
    async def _update_memory_graph(self, memory: MemoryEntry):
        """Update the memory association graph"""
        
        if not self.memory_graph:
            return
        
        # Add node
        self.memory_graph.add_node(
            memory.id,
            user_id=memory.user_id,
            timestamp=memory.timestamp,
            importance=memory.importance
        )
        
        # Find related memories
        related_memories = await self._semantic_search(
            memory.embedding,
            memory.user_id,
            None,
            5
        )
        
        # Create edges to related memories
        for related in related_memories:
            if related.memory.id != memory.id and related.score > 0.7:
                self.memory_graph.add_edge(
                    memory.id,
                    related.memory.id,
                    weight=related.score,
                    created_at=datetime.now()
                )
                
                # Store association in database
                await self._store_association(
                    memory.id,
                    related.memory.id,
                    "semantic",
                    related.score
                )
    
    async def _store_association(
        self,
        source_id: str,
        target_id: str,
        association_type: str,
        strength: float
    ):
        """Store memory association in database"""
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO memory_associations 
                (source_memory_id, target_memory_id, association_type, strength)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (source_memory_id, target_memory_id) 
                DO UPDATE SET strength = GREATEST(memory_associations.strength, $4)
            """, source_id, target_id, association_type, strength)
    
    # Public utility methods
    
    def set_consciousness_state(self, state: Dict[str, Any]):
        """Update consciousness state for memory modulation"""
        self.consciousness_state = state
    
    def set_emotional_context(self, context: Dict[str, float]):
        """Update emotional context for memory processing"""
        self.emotional_context = context
    
    async def cleanup_old_memories(
        self,
        days_to_keep: int = 90,
        preserve_important: bool = True
    ) -> int:
        """Clean up old memories to manage storage"""
        
        async with self.db_pool.acquire() as conn:
            query = """
                DELETE FROM memories
                WHERE timestamp < $1
            """
            
            params = [datetime.now() - timedelta(days=days_to_keep)]
            
            if preserve_important:
                query += " AND importance < 0.7"
            
            result = await conn.execute(query, *params)
            
            # Extract number of deleted rows
            deleted_count = int(result.split()[-1])
            
            logger.info(f"Cleaned up {deleted_count} old memories")
            
            return deleted_count
    
    async def export_memories(
        self,
        user_id: str,
        output_format: str = "json"
    ) -> Union[str, bytes]:
        """Export user memories in specified format"""
        
        async with self.db_pool.acquire() as conn:
            memories = await conn.fetch(
                "SELECT * FROM memories WHERE user_id = $1 ORDER BY timestamp DESC",
                user_id
            )
        
        if output_format == "json":
            export_data = {
                "user_id": user_id,
                "export_date": datetime.now().isoformat(),
                "memory_count": len(memories),
                "memories": [
                    {
                        "id": str(m['id']),
                        "content": m['content'],
                        "timestamp": m['timestamp'].isoformat(),
                        "importance": float(m['importance']),
                        "emotional_valence": float(m['emotional_valence']),
                        "memory_type": m['memory_type'],
                        "metadata": json.loads(m['metadata']) if m['metadata'] else {}
                    }
                    for m in memories
                ]
            }
            
            return json.dumps(export_data, indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {output_format}")
    
    async def close(self):
        """Clean up resources"""
        
        if self.db_pool:
            await self.db_pool.close()
        
        logger.info("ConsciousnessAwareMemory closed")