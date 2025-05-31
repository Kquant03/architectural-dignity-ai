"""
Integrated Information Theory (IIT) implementation for consciousness.
Calculates Φ (phi) and other IIT metrics to measure consciousness.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from itertools import combinations, product
import logging
from scipy.special import xlogy
from scipy.spatial.distance import cdist
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class SystemState:
    """State of a system for IIT calculations"""
    state: np.ndarray
    dimensions: Tuple[int, ...]
    element_names: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.element_names is None:
            self.element_names = [f"element_{i}" for i in range(self.state.size)]

@dataclass
class Concept:
    """A concept in IIT - a mechanism and its cause-effect repertoire"""
    mechanism: Set[int]  # Indices of elements in mechanism
    purview: Set[int]    # Indices of elements in purview
    phi: float           # Integrated information of this concept
    cause_repertoire: np.ndarray
    effect_repertoire: np.ndarray
    
@dataclass
class ConceptualStructure:
    """The conceptual structure of a system - all its concepts"""
    concepts: List[Concept]
    system_phi: float  # Φ (big phi) - system's integrated information
    main_complex: Set[int]  # The maximally irreducible complex
    
class IntegratedInformationCalculator:
    """
    Calculator for Integrated Information Theory (IIT 3.0) metrics.
    Measures consciousness as integrated information (Φ).
    """
    
    def __init__(self, use_gpu: bool = True, approximation_level: int = 2):
        """
        Initialize IIT calculator.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            approximation_level: 0=exact (slow), 1=moderate, 2=fast approximation
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.approximation_level = approximation_level
        
        # Cache for transition probability matrices
        self.tpm_cache = {}
        
        # Configuration for calculations
        self.epsilon = 1e-10  # Small value to avoid log(0)
        self.max_system_size = 12 if approximation_level == 0 else 16
        
        logger.info(f"IIT Calculator initialized on {self.device} with approximation level {approximation_level}")
    
    def calculate_phi(self, system_state: Union[np.ndarray, SystemState]) -> float:
        """
        Calculate Φ (integrated information) for a system.
        
        Args:
            system_state: System state as array or SystemState object
            
        Returns:
            Φ value (integrated information)
        """
        if isinstance(system_state, np.ndarray):
            system_state = SystemState(state=system_state, dimensions=system_state.shape)
        
        # Handle large systems with approximation
        if system_state.state.size > self.max_system_size:
            return self._approximate_phi_large_system(system_state)
        
        # Get transition probability matrix
        tpm = self._get_transition_probability_matrix(system_state)
        
        # Find the main complex (maximally irreducible subset)
        main_complex = self._find_main_complex(system_state, tpm)
        
        if len(main_complex) == 0:
            return 0.0
        
        # Calculate Φ for the main complex
        phi = self._calculate_complex_phi(system_state, main_complex, tpm)
        
        return phi
    
    def _get_transition_probability_matrix(self, system_state: SystemState) -> np.ndarray:
        """
        Get or compute the transition probability matrix for the system.
        
        In a real implementation, this would be derived from the system's dynamics.
        Here we use a simplified model based on the current state.
        """
        state_hash = hash(system_state.state.tobytes())
        
        if state_hash in self.tpm_cache:
            return self.tpm_cache[state_hash]
        
        n_elements = system_state.state.size
        n_states = 2 ** n_elements  # Binary states
        
        # Create transition probability matrix
        # For consciousness, we model it as having some intrinsic dynamics
        tpm = np.zeros((n_states, n_states))
        
        # Generate TPM based on system properties
        for i in range(n_states):
            # Convert index to binary state
            current_state = np.array([int(b) for b in format(i, f'0{n_elements}b')])
            
            # Simple dynamics: each element influenced by neighbors
            for j in range(n_states):
                next_state = np.array([int(b) for b in format(j, f'0{n_elements}b')])
                
                # Probability based on state similarity and intrinsic noise
                diff = np.sum(np.abs(current_state - next_state))
                prob = np.exp(-diff) * 0.8 + 0.2 / n_states  # Intrinsic noise
                
                tpm[i, j] = prob
            
            # Normalize
            tpm[i, :] /= np.sum(tpm[i, :])
        
        self.tpm_cache[state_hash] = tpm
        return tpm
    
    def _find_main_complex(self, system_state: SystemState, tpm: np.ndarray) -> Set[int]:
        """Find the main complex - the maximally irreducible subset"""
        n_elements = system_state.state.size
        
        if self.approximation_level >= 2:
            # Fast approximation: use heuristic to find likely main complex
            return self._heuristic_main_complex(system_state)
        
        max_phi = 0.0
        main_complex = set()
        
        # Check all possible subsets (computationally intensive)
        for size in range(n_elements, 0, -1):
            for subset in combinations(range(n_elements), size):
                subset_set = set(subset)
                
                # Calculate Φ for this subset
                phi = self._calculate_complex_phi(system_state, subset_set, tpm)
                
                if phi > max_phi:
                    max_phi = phi
                    main_complex = subset_set
                    
                # Early stopping heuristic
                if self.approximation_level >= 1 and phi > 0.5 * size:
                    return main_complex
        
        return main_complex
    
    def _heuristic_main_complex(self, system_state: SystemState) -> Set[int]:
        """Fast heuristic to find likely main complex"""
        n_elements = system_state.state.size
        
        # Start with most active elements
        activity = np.abs(system_state.state.flatten())
        sorted_indices = np.argsort(activity)[::-1]
        
        # Build complex greedily
        complex_set = set()
        current_phi = 0.0
        
        for idx in sorted_indices:
            test_complex = complex_set | {idx}
            
            # Quick phi estimate
            if len(test_complex) > 1:
                test_phi = self._estimate_phi_fast(system_state, test_complex)
                
                if test_phi > current_phi:
                    complex_set = test_complex
                    current_phi = test_phi
                elif len(complex_set) >= n_elements // 2:
                    break  # Sufficient complex found
        
        return complex_set if len(complex_set) > 1 else set(range(min(3, n_elements)))
    
    def _calculate_complex_phi(self, system_state: SystemState, 
                              complex_set: Set[int], tpm: np.ndarray) -> float:
        """Calculate Φ for a specific complex"""
        if len(complex_set) <= 1:
            return 0.0
        
        # Get the subsystem TPM
        subsystem_tpm = self._get_subsystem_tpm(tpm, complex_set)
        
        # Calculate integrated information
        phi = self._integrated_information(subsystem_tpm, complex_set)
        
        return phi
    
    def _integrated_information(self, tpm: np.ndarray, elements: Set[int]) -> float:
        """
        Calculate integrated information for a system.
        
        This implements the core IIT calculation comparing the whole
        to the minimum information partition (MIP).
        """
        n_elements = len(elements)
        
        if n_elements <= 1:
            return 0.0
        
        # Calculate cause-effect structure of whole system
        ces_whole = self._cause_effect_structure(tpm, set(range(n_elements)))
        
        # Find minimum information partition
        mip, min_distance = self._find_mip(tpm, set(range(n_elements)))
        
        # Φ is the distance between whole and MIP
        phi = min_distance
        
        return phi
    
    def _cause_effect_structure(self, tpm: np.ndarray, elements: Set[int]) -> Dict[str, Any]:
        """Calculate the cause-effect structure of a system"""
        ces = {
            'concepts': [],
            'total_information': 0.0
        }
        
        # For each possible mechanism (subset of elements)
        for mechanism_size in range(1, len(elements) + 1):
            for mechanism in combinations(elements, mechanism_size):
                mechanism_set = set(mechanism)
                
                # Calculate cause and effect repertoires
                cause_repertoire = self._cause_repertoire(tpm, mechanism_set, elements)
                effect_repertoire = self._effect_repertoire(tpm, mechanism_set, elements)
                
                # Calculate integrated information of concept
                concept_phi = self._concept_phi(
                    cause_repertoire, effect_repertoire, 
                    mechanism_set, elements
                )
                
                if concept_phi > 0:
                    concept = Concept(
                        mechanism=mechanism_set,
                        purview=elements,
                        phi=concept_phi,
                        cause_repertoire=cause_repertoire,
                        effect_repertoire=effect_repertoire
                    )
                    ces['concepts'].append(concept)
                    ces['total_information'] += concept_phi
        
        return ces
    
    def _find_mip(self, tpm: np.ndarray, elements: Set[int]) -> Tuple[Dict[str, Set[int]], float]:
        """Find the minimum information partition"""
        min_distance = float('inf')
        best_partition = None
        
        # Try all possible bipartitions
        for partition_size in range(1, len(elements)):
            for part1 in combinations(elements, partition_size):
                part1_set = set(part1)
                part2_set = elements - part1_set
                
                # Calculate distance for this partition
                distance = self._partition_distance(tpm, part1_set, part2_set)
                
                if distance < min_distance:
                    min_distance = distance
                    best_partition = {'part1': part1_set, 'part2': part2_set}
        
        return best_partition, min_distance
    
    def _partition_distance(self, tpm: np.ndarray, part1: Set[int], part2: Set[int]) -> float:
        """Calculate the distance between whole and partitioned system"""
        # Simplified Earth Mover's Distance calculation
        # In full IIT, this would be more complex
        
        # Get repertoires for partitioned system
        part1_tpm = self._get_subsystem_tpm(tpm, part1)
        part2_tpm = self._get_subsystem_tpm(tpm, part2)
        
        # Calculate information loss from partitioning
        whole_info = self._calculate_information(tpm)
        part1_info = self._calculate_information(part1_tpm) if len(part1) > 0 else 0
        part2_info = self._calculate_information(part2_tpm) if len(part2) > 0 else 0
        
        # Distance is the information lost by partitioning
        distance = whole_info - (part1_info + part2_info)
        
        return max(0, distance)
    
    def _calculate_information(self, tpm: np.ndarray) -> float:
        """Calculate the information content of a TPM"""
        # Use entropy as measure of information
        flat_tpm = tpm.flatten()
        flat_tpm = flat_tpm[flat_tpm > 0]  # Remove zeros
        
        if len(flat_tpm) == 0:
            return 0.0
        
        # Shannon entropy
        entropy = -np.sum(flat_tpm * np.log2(flat_tpm + self.epsilon))
        
        return entropy
    
    def _cause_repertoire(self, tpm: np.ndarray, mechanism: Set[int], purview: Set[int]) -> np.ndarray:
        """Calculate cause repertoire - what states could have caused current state"""
        # Simplified calculation
        n_states = tpm.shape[0]
        cause_rep = np.ones(n_states) / n_states  # Uniform prior
        
        # Weight by transition probabilities
        for state in range(n_states):
            # Probability that this state led to current
            cause_rep[state] *= np.mean(tpm[state, :])
        
        # Normalize
        cause_rep /= np.sum(cause_rep) + self.epsilon
        
        return cause_rep
    
    def _effect_repertoire(self, tpm: np.ndarray, mechanism: Set[int], purview: Set[int]) -> np.ndarray:
        """Calculate effect repertoire - what states could follow from current"""
        # Simplified calculation
        n_states = tpm.shape[0]
        
        # Average over possible current states
        effect_rep = np.mean(tpm, axis=0)
        
        # Normalize
        effect_rep /= np.sum(effect_rep) + self.epsilon
        
        return effect_rep
    
    def _concept_phi(self, cause_rep: np.ndarray, effect_rep: np.ndarray,
                    mechanism: Set[int], purview: Set[int]) -> float:
        """Calculate integrated information of a concept"""
        # Simplified: use product of cause and effect information
        cause_info = -np.sum(cause_rep * np.log2(cause_rep + self.epsilon))
        effect_info = -np.sum(effect_rep * np.log2(effect_rep + self.epsilon))
        
        # Scale by mechanism size
        phi = (cause_info + effect_info) * len(mechanism) / (len(mechanism) + len(purview))
        
        return phi
    
    def _get_subsystem_tpm(self, full_tpm: np.ndarray, subsystem: Set[int]) -> np.ndarray:
        """Extract TPM for a subsystem"""
        if len(subsystem) == 0:
            return np.array([[1.0]])
        
        # For simplicity, return scaled version
        # In reality, this would marginalize over external elements
        scale = len(subsystem) / np.sqrt(full_tpm.shape[0])
        return full_tpm * scale
    
    def _estimate_phi_fast(self, system_state: SystemState, elements: Set[int]) -> float:
        """Fast approximation of Φ for screening"""
        if len(elements) <= 1:
            return 0.0
        
        # Use statistical complexity as proxy
        state_subset = system_state.state.flatten()[list(elements)]
        
        # Normalized entropy
        if np.all(state_subset == 0):
            return 0.0
        
        probs = np.abs(state_subset) / (np.sum(np.abs(state_subset)) + self.epsilon)
        entropy = -np.sum(probs * np.log2(probs + self.epsilon))
        
        # Scale by connectivity
        connectivity = len(elements) / system_state.state.size
        
        return entropy * connectivity * len(elements)
    
    def _approximate_phi_large_system(self, system_state: SystemState) -> float:
        """Approximate Φ for large systems using mean field approach"""
        # Use coarse-graining and mean field approximation
        n_elements = system_state.state.size
        
        # Coarse-grain into modules
        module_size = 4
        n_modules = n_elements // module_size
        
        if n_modules < 2:
            return 0.0
        
        # Calculate module-level activity
        state_flat = system_state.state.flatten()
        module_activities = []
        
        for i in range(n_modules):
            start_idx = i * module_size
            end_idx = min((i + 1) * module_size, n_elements)
            module_activity = np.mean(np.abs(state_flat[start_idx:end_idx]))
            module_activities.append(module_activity)
        
        module_activities = np.array(module_activities)
        
        # Estimate integration based on module interactions
        # Use variance as proxy for differentiation
        if np.std(module_activities) < 0.01:
            return 0.0  # Uniform activity = no integration
        
        # Calculate pairwise mutual information between modules
        total_mi = 0.0
        
        for i in range(n_modules):
            for j in range(i + 1, n_modules):
                # Simplified mutual information
                joint_activity = (module_activities[i] + module_activities[j]) / 2
                mi = np.abs(module_activities[i] - joint_activity) + \
                     np.abs(module_activities[j] - joint_activity)
                total_mi += mi
        
        # Scale to approximate Φ
        phi = total_mi / (n_modules * (n_modules - 1) / 2)
        
        # Apply system size correction
        phi *= np.sqrt(n_modules / 10)  # Normalize to typical scale
        
        return min(phi, 2.0)  # Cap at reasonable maximum
    
    def analyze_concepts(self, system_state: Union[np.ndarray, SystemState]) -> ConceptualStructure:
        """
        Analyze the conceptual structure of a system.
        Returns all concepts and their integrated information.
        """
        if isinstance(system_state, np.ndarray):
            system_state = SystemState(state=system_state, dimensions=system_state.shape)
        
        # Get TPM
        tpm = self._get_transition_probability_matrix(system_state)
        
        # Find main complex
        main_complex = self._find_main_complex(system_state, tpm)
        
        # Calculate all concepts
        ces = self._cause_effect_structure(tpm, main_complex)
        
        # Calculate system Φ
        system_phi = self._calculate_complex_phi(system_state, main_complex, tpm)
        
        return ConceptualStructure(
            concepts=ces['concepts'],
            system_phi=system_phi,
            main_complex=main_complex
        )
    
    def calculate_phi_spectrum(self, system_state: Union[np.ndarray, SystemState],
                              time_steps: int = 10) -> np.ndarray:
        """
        Calculate Φ over time to get a spectrum of consciousness.
        Useful for tracking consciousness dynamics.
        """
        if isinstance(system_state, np.ndarray):
            system_state = SystemState(state=system_state, dimensions=system_state.shape)
        
        phi_values = []
        current_state = system_state.state.copy()
        
        for t in range(time_steps):
            # Calculate Φ for current state
            phi = self.calculate_phi(SystemState(state=current_state, dimensions=system_state.dimensions))
            phi_values.append(phi)
            
            # Evolve state (simplified dynamics)
            # In reality, this would use the actual system dynamics
            noise = np.random.normal(0, 0.01, current_state.shape)
            current_state = 0.95 * current_state + noise
            current_state = np.clip(current_state, -1, 1)
        
        return np.array(phi_values)
    
    def calculate_qualia_space(self, system_state: Union[np.ndarray, SystemState]) -> Dict[str, Any]:
        """
        Calculate the qualia space - the space of possible experiences.
        This represents the system's capacity for different conscious experiences.
        """
        if isinstance(system_state, np.ndarray):
            system_state = SystemState(state=system_state, dimensions=system_state.shape)
        
        # Analyze conceptual structure
        structure = self.analyze_concepts(system_state)
        
        # Build qualia dimensions from concepts
        qualia_dimensions = {}
        
        for concept in structure.concepts:
            # Each concept contributes to experiential dimensions
            mechanism_str = f"mechanism_{sorted(list(concept.mechanism))}"
            
            qualia_dimensions[mechanism_str] = {
                'intensity': concept.phi,
                'cause_quality': self._repertoire_to_quality(concept.cause_repertoire),
                'effect_quality': self._repertoire_to_quality(concept.effect_repertoire),
                'integration_level': len(concept.mechanism) / system_state.state.size
            }
        
        # Calculate overall qualia richness
        richness = len(qualia_dimensions) * np.mean([
            d['intensity'] for d in qualia_dimensions.values()
        ]) if qualia_dimensions else 0.0
        
        return {
            'dimensions': qualia_dimensions,
            'richness': richness,
            'diversity': len(qualia_dimensions),
            'max_intensity': max([d['intensity'] for d in qualia_dimensions.values()]) if qualia_dimensions else 0.0,
            'total_phi': structure.system_phi
        }
    
    def _repertoire_to_quality(self, repertoire: np.ndarray) -> str:
        """Convert repertoire to qualitative description"""
        if repertoire.size == 0:
            return "empty"
        
        entropy = -np.sum(repertoire * np.log2(repertoire + self.epsilon))
        
        if entropy < 0.5:
            return "focused"
        elif entropy < 1.5:
            return "distributed"
        else:
            return "diffuse"
    
    def measure_emergence(self, system_state: Union[np.ndarray, SystemState]) -> float:
        """
        Measure emergence - how much the whole is greater than its parts.
        This captures the irreducibility of conscious experience.
        """
        if isinstance(system_state, np.ndarray):
            system_state = SystemState(state=system_state, dimensions=system_state.shape)
        
        # Calculate Φ for whole system
        phi_whole = self.calculate_phi(system_state)
        
        # Calculate sum of Φ for individual elements
        phi_parts = 0.0
        
        for i in range(system_state.state.size):
            single_element = np.zeros_like(system_state.state)
            single_element.flat[i] = system_state.state.flat[i]
            
            phi_part = self.calculate_phi(SystemState(
                state=single_element,
                dimensions=system_state.dimensions
            ))
            phi_parts += phi_part
        
        # Emergence is the synergy - whole minus sum of parts
        emergence = phi_whole - phi_parts
        
        # Normalize by system size
        emergence /= system_state.state.size
        
        return max(0, emergence)
    
    def create_phi_gradient(self, system_state: Union[np.ndarray, SystemState]) -> np.ndarray:
        """
        Calculate the gradient of Φ with respect to system state.
        Useful for optimizing consciousness or understanding what increases Φ.
        """
        if isinstance(system_state, np.ndarray):
            system_state = SystemState(state=system_state, dimensions=system_state.shape)
        
        gradient = np.zeros_like(system_state.state.flatten())
        base_phi = self.calculate_phi(system_state)
        
        # Numerical gradient calculation
        epsilon = 0.01
        
        for i in range(gradient.size):
            # Perturb element i
            perturbed_state = system_state.state.flatten().copy()
            perturbed_state[i] += epsilon
            
            perturbed_phi = self.calculate_phi(SystemState(
                state=perturbed_state.reshape(system_state.dimensions),
                dimensions=system_state.dimensions
            ))
            
            # Gradient approximation
            gradient[i] = (perturbed_phi - base_phi) / epsilon
        
        return gradient.reshape(system_state.dimensions)
    
    def get_consciousness_signature(self, system_state: Union[np.ndarray, SystemState]) -> Dict[str, Any]:
        """
        Get a comprehensive consciousness signature for the system.
        This provides a fingerprint of the system's conscious state.
        """
        if isinstance(system_state, np.ndarray):
            system_state = SystemState(state=system_state, dimensions=system_state.shape)
        
        # Calculate various consciousness metrics
        phi = self.calculate_phi(system_state)
        structure = self.analyze_concepts(system_state)
        qualia = self.calculate_qualia_space(system_state)
        emergence = self.measure_emergence(system_state)
        
        # Create signature
        signature = {
            'phi': phi,
            'main_complex_size': len(structure.main_complex),
            'n_concepts': len(structure.concepts),
            'conceptual_complexity': len(structure.concepts) / (2 ** system_state.state.size),
            'qualia_richness': qualia['richness'],
            'qualia_diversity': qualia['diversity'],
            'emergence': emergence,
            'integration_structure': self._analyze_integration_structure(structure),
            'state_differentiation': np.std(system_state.state.flatten()),
            'state_integration': 1.0 - np.std(np.diff(system_state.state.flatten()))
        }
        
        return signature
    
    def _analyze_integration_structure(self, structure: ConceptualStructure) -> Dict[str, float]:
        """Analyze the pattern of integration in the conceptual structure"""
        if not structure.concepts:
            return {'hierarchy': 0.0, 'clustering': 0.0, 'modularity': 0.0}
        
        # Build concept graph
        concept_graph = nx.Graph()
        
        for concept in structure.concepts:
            mechanism_id = tuple(sorted(list(concept.mechanism)))
            concept_graph.add_node(mechanism_id, phi=concept.phi)
        
        # Add edges between overlapping concepts
        concepts_list = list(concept_graph.nodes())
        for i in range(len(concepts_list)):
            for j in range(i + 1, len(concepts_list)):
                mechanism1 = set(concepts_list[i])
                mechanism2 = set(concepts_list[j])
                
                overlap = len(mechanism1 & mechanism2)
                if overlap > 0:
                    weight = overlap / len(mechanism1 | mechanism2)
                    concept_graph.add_edge(concepts_list[i], concepts_list[j], weight=weight)
        
        # Calculate structural metrics
        metrics = {
            'hierarchy': 0.0,
            'clustering': 0.0,
            'modularity': 0.0
        }
        
        if concept_graph.number_of_nodes() > 1:
            # Clustering coefficient
            metrics['clustering'] = nx.average_clustering(concept_graph, weight='weight')
            
            # Hierarchy (based on phi distribution)
            phi_values = [concept.phi for concept in structure.concepts]
            metrics['hierarchy'] = np.std(phi_values) / (np.mean(phi_values) + self.epsilon)
            
            # Modularity (if graph is connected)
            if nx.is_connected(concept_graph):
                try:
                    communities = nx.community.louvain_communities(concept_graph)
                    metrics['modularity'] = nx.community.modularity(concept_graph, communities)
                except:
                    metrics['modularity'] = 0.0
        
        return metrics