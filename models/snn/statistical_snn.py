# cogmenta_core/models/snn/statistical_snn.py

import numpy as np
import random
import math
import time
from collections import defaultdict, deque
from scipy import sparse
from typing import Dict, List, Tuple, Optional, Any, Set, Union

class StatisticalSNN:
    """
    Statistical Generalization Spiking Neural Network.
    
    Specializes in:
    - Learning statistical patterns from data
    - Generalizing across similar concepts
    - Fuzzy matching of concepts based on similarity
    - Embedding space operations for analogical reasoning
    - Few-shot learning of new concepts
    
    Works alongside the abductive reasoning SNN to provide
    complementary capabilities in a hybrid cognitive architecture.
    """
    
    def __init__(self, neuron_count=1000, embedding_dim=300, learning_rate=0.01):
        """Initialize the statistical generalization SNN"""
        self.neuron_count = neuron_count
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.decay_rate = 0.9
        
        # Neuron membrane potentials
        self.membrane_potentials = np.zeros(neuron_count)
        
        # Spiking thresholds (with heterogeneity)
        self.spike_thresholds = np.random.uniform(0.4, 0.6, neuron_count)
        
        # Initialize sparse weights (low connectivity for efficiency)
        connection_density = 0.1
        self.synaptic_weights = self._init_sparse_weights(connection_density)
        
        # Concept embeddings - map concepts to vector representations
        self.concept_embeddings = {}
        
        # Concept clusters - for grouping similar concepts
        self.concept_clusters = defaultdict(list)
        
        # Statistical pattern memory
        self.pattern_memory = []
        
        # Attractor dynamics for pattern completion
        self.attractor_states = {}
        
        # Adaptive learning capacity
        self.plasticity = np.ones(neuron_count) * 0.5
        
        # Recent activation history (for learning)
        self.activation_history = deque(maxlen=100)
        
        # Network regions (specialized for statistical operations)
        self._init_statistical_regions()
        
        # Initialize STDP learning mechanism
        self._init_learning_mechanisms()
        
        print(f"[StatSNN] Initialized Statistical SNN with {neuron_count} neurons, {embedding_dim}D embeddings")
    
    def _init_sparse_weights(self, connection_density):
        """Initialize sparse synaptic weights for efficiency"""
        expected_connections = int(self.neuron_count * self.neuron_count * connection_density)
        
        rows = []
        cols = []
        data = []
        
        # Generate random sparse connections
        for _ in range(expected_connections):
            i = random.randint(0, self.neuron_count - 1)
            j = random.randint(0, self.neuron_count - 1)
            # Small random weight with both positive and negative values
            weight = np.random.normal(0, 0.1)
            rows.append(i)
            cols.append(j)
            data.append(weight)
        
        # Create sparse matrix in CSR format
        weights = sparse.csr_matrix(
            (data, (rows, cols)), 
            shape=(self.neuron_count, self.neuron_count)
        )
        
        return weights
    
    def _init_statistical_regions(self):
        """Initialize neural regions specialized for statistical operations"""
        total_neurons = self.neuron_count
        
        # Define statistical processing regions
        self.regions = {
            # Processes raw input into distributed representations
            'embedding': {
                'neurons': list(range(0, int(0.2 * total_neurons))),
                'activation': 0.0
            },
            # Handles similarity-based retrieval
            'similarity': {
                'neurons': list(range(int(0.2 * total_neurons), int(0.4 * total_neurons))),
                'activation': 0.0
            },
            # Generalization across examples
            'generalization': {
                'neurons': list(range(int(0.4 * total_neurons), int(0.6 * total_neurons))),
                'activation': 0.0
            },
            # Pattern completion from partial inputs
            'completion': {
                'neurons': list(range(int(0.6 * total_neurons), int(0.8 * total_neurons))),
                'activation': 0.0
            },
            # Few-shot learning mechanisms
            'adaptation': {
                'neurons': list(range(int(0.8 * total_neurons), total_neurons)),
                'activation': 0.0
            }
        }
        
        # Initialize region connections (fully connected between regions)
        self.region_connections = {
            'embedding': ['similarity', 'generalization'],
            'similarity': ['embedding', 'completion', 'generalization'],
            'generalization': ['similarity', 'completion', 'adaptation'],
            'completion': ['similarity', 'generalization'],
            'adaptation': ['generalization', 'embedding']
        }
    
    def _init_learning_mechanisms(self):
        """Initialize statistical learning mechanisms"""
        # STDP parameters for adjusting synaptic weights
        self.stdp_window = 20.0  # ms
        self.stdp_lr_pos = 0.005  # Learning rate for potentiation
        self.stdp_lr_neg = 0.003  # Learning rate for depression
        
        # Spike timing storage for STDP
        self.last_spike_times = np.zeros(self.neuron_count)
        
        # Hebbian learning parameters
        self.hebbian_lr = 0.01
        
        # Reinforcement learning parameters
        self.rl_lr = 0.02
        self.value_estimate = 0.0
        
        # Contrastive learning for similarity
        self.contrastive_margin = 0.5
    
    def learn_concept_embedding(self, concept_name, features=None, related_concepts=None):
        """
        Learn embeddings for concepts with statistical generalization.
        
        Args:
            concept_name: Name of the concept to learn
            features: Feature vector or attributes of the concept
            related_concepts: List of related concept names
            
        Returns:
            The learned embedding vector
        """
        # Generate embedding from features if provided, otherwise random
        if features is not None:
            # Ensure features has embedding_dim dimensions
            if len(features) != self.embedding_dim:
                # Resize or pad features to match embedding dimension
                if len(features) > self.embedding_dim:
                    embedding = features[:self.embedding_dim]
                else:
                    embedding = np.pad(features, (0, self.embedding_dim - len(features)))
            else:
                embedding = np.array(features)
        else:
            # Generate random embedding with normal distribution
            embedding = np.random.normal(0, 0.1, self.embedding_dim)
        
        # Normalize embedding to unit length
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Add related concept influences (statistical generalization)
        if related_concepts:
            related_embeddings = []
            for rel_concept in related_concepts:
                if rel_concept in self.concept_embeddings:
                    related_embeddings.append(self.concept_embeddings[rel_concept])
            
            if related_embeddings:
                # Average related embeddings and combine with new embedding
                related_avg = np.mean(related_embeddings, axis=0)
                # Weighted combination (70% original, 30% related concepts)
                embedding = 0.7 * embedding + 0.3 * related_avg
                # Renormalize
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Store the embedding
        self.concept_embeddings[concept_name] = embedding
        
        # Update concept clusters
        self._update_concept_clusters(concept_name, embedding)
        
        return embedding
    
    def _update_concept_clusters(self, concept_name, embedding):
        """Update concept clusters based on embedding similarity"""
        # Find closest existing cluster
        best_cluster = None
        best_similarity = -1
        
        for cluster_name, cluster_concepts in self.concept_clusters.items():
            # Calculate average embedding for cluster
            if not cluster_concepts:
                continue
                
            cluster_embeddings = [self.concept_embeddings[c] for c in cluster_concepts 
                                if c in self.concept_embeddings]
            if not cluster_embeddings:
                continue
                
            cluster_center = np.mean(cluster_embeddings, axis=0)
            # Calculate cosine similarity
            similarity = np.dot(embedding, cluster_center) / (
                np.linalg.norm(embedding) * np.linalg.norm(cluster_center)
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_name
        
        # If similar enough to best cluster, add to it
        if best_similarity > 0.7 and best_cluster is not None:
            self.concept_clusters[best_cluster].append(concept_name)
        else:
            # Otherwise create new cluster
            new_cluster = f"cluster_{len(self.concept_clusters)}"
            self.concept_clusters[new_cluster].append(concept_name)
    
    def find_similar_concepts(self, query_concept=None, query_embedding=None, top_k=5):
        """
        Find concepts similar to the query concept or embedding
        
        Args:
            query_concept: Name of the concept to find similar concepts to
            query_embedding: Embedding vector to find similar concepts to
            top_k: Number of similar concepts to return
            
        Returns:
            List of (concept_name, similarity_score) tuples
        """
        # Get query embedding
        if query_embedding is None and query_concept is not None:
            if query_concept in self.concept_embeddings:
                query_embedding = self.concept_embeddings[query_concept]
            else:
                print(f"[StatSNN] Warning: Concept '{query_concept}' not found")
                return []
        
        if query_embedding is None:
            return []
        
        # Calculate similarity to all concepts
        similarities = []
        for concept, embedding in self.concept_embeddings.items():
            if concept == query_concept:
                continue
                
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            
            similarities.append((concept, similarity))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def process_input(self, input_data, query_type='similarity'):
        """
        Process input data and perform statistical operations
        
        Args:
            input_data: Input data (concept name, features, or embedding)
            query_type: Type of statistical operation to perform
                - 'similarity': Find similar concepts
                - 'generalization': Generalize from examples
                - 'completion': Complete partial patterns
                - 'analogy': Perform analogical reasoning
                
        Returns:
            Dictionary with results of the statistical operation
        """
        # Convert input to embedding
        input_embedding = self._get_input_embedding(input_data)
        
        # Activate the network with the input embedding
        self._activate_with_embedding(input_embedding)
        
        # Process based on query type
        if query_type == 'similarity':
            # Find similar concepts
            similar_concepts = self.find_similar_concepts(
                query_embedding=input_embedding, top_k=10
            )
            
            # Activate similarity region
            self._region_activate('similarity', 0.8)
            
            return {
                'operation': 'similarity',
                'similar_concepts': similar_concepts,
                'input_embedding': input_embedding
            }
            
        elif query_type == 'generalization':
            # Perform statistical generalization
            generalizations = self._generalize_from_input(input_embedding)
            
            # Activate generalization region
            self._region_activate('generalization', 0.8)
            
            return {
                'operation': 'generalization',
                'generalizations': generalizations,
                'input_embedding': input_embedding
            }
            
        elif query_type == 'completion':
            # Complete partial patterns
            completed = self._complete_pattern(input_embedding)
            
            # Activate completion region
            self._region_activate('completion', 0.8)
            
            return {
                'operation': 'completion',
                'completed_pattern': completed,
                'confidence': completed.get('confidence', 0.0),
                'input_embedding': input_embedding
            }
            
        elif query_type == 'analogy':
            # Perform analogical reasoning (A is to B as C is to ?)
            if isinstance(input_data, dict) and all(k in input_data for k in ['A', 'B', 'C']):
                analogy_result = self._solve_analogy(input_data['A'], input_data['B'], input_data['C'])
                
                # Activate multiple regions for complex operation
                self._region_activate('similarity', 0.7)
                self._region_activate('generalization', 0.8)
                
                return {
                    'operation': 'analogy',
                    'result': analogy_result,
                    'confidence': analogy_result.get('confidence', 0.0)
                }
            else:
                return {'error': 'Invalid input for analogy operation'}
        else:
            return {'error': f'Unknown query type: {query_type}'}
    
    def _get_input_embedding(self, input_data):
        """Convert input data to embedding"""
        if isinstance(input_data, str):
            # Input is a concept name
            if input_data in self.concept_embeddings:
                return self.concept_embeddings[input_data]
            else:
                # Create a new random embedding
                return np.random.normal(0, 0.1, self.embedding_dim)
        elif isinstance(input_data, np.ndarray):
            # Input is already an embedding
            if input_data.shape[0] == self.embedding_dim:
                return input_data
            else:
                # Resize or pad to match embedding dimension
                if input_data.shape[0] > self.embedding_dim:
                    return input_data[:self.embedding_dim]
                else:
                    return np.pad(input_data, (0, self.embedding_dim - input_data.shape[0]))
        elif isinstance(input_data, list):
            # Input is a list, convert to numpy array
            return self._get_input_embedding(np.array(input_data))
        elif isinstance(input_data, dict) and 'features' in input_data:
            # Input is a dictionary with features
            return self._get_input_embedding(input_data['features'])
        else:
            # Default to random embedding
            return np.random.normal(0, 0.1, self.embedding_dim)
    
    def _activate_with_embedding(self, embedding):
        """Activate the network with an embedding"""
        # Normalize the embedding if not already normalized
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Map embedding to activation pattern for embedding region
        embedding_region = self.regions['embedding']['neurons']
        
        # Initialize activation pattern
        activation = np.zeros(self.neuron_count)
        
        # Distribute embedding across embedding region neurons
        region_size = len(embedding_region)
        embedding_dim = embedding.shape[0]
        
        if embedding_dim <= region_size:
            # Direct mapping if embedding fits in region
            for i in range(embedding_dim):
                neuron_idx = embedding_region[i]
                # Map embedding values to activation (0.5-1.0 range)
                activation[neuron_idx] = 0.5 + 0.5 * (embedding[i] + 1.0)/2.0
        else:
            # Distribute with overlap if embedding is larger than region
            for i in range(region_size):
                neuron_idx = embedding_region[i]
                # Take average of corresponding embedding segments
                segment_size = embedding_dim // region_size
                start = i * segment_size
                end = min((i + 1) * segment_size, embedding_dim)
                segment_avg = np.mean(embedding[start:end])
                # Map to activation (0.5-1.0 range)
                activation[neuron_idx] = 0.5 + 0.5 * (segment_avg + 1.0)/2.0
        
        # Set membrane potentials to activation values
        self.membrane_potentials = activation
        
        # Propagate activation through network
        self._propagate_activation(5)  # 5 time steps
        
        # Record activation for learning
        self.activation_history.append(activation)
    
    def _region_activate(self, region_name, strength=1.0):
        """Activate a specific network region"""
        if region_name in self.regions:
            self.regions[region_name]['activation'] = strength
            
            # Also activate neurons in this region
            for neuron_idx in self.regions[region_name]['neurons']:
                self.membrane_potentials[neuron_idx] = max(
                    self.membrane_potentials[neuron_idx],
                    strength * 0.8
                )
    
    def _propagate_activation(self, steps=3):
        """Propagate activation through the network"""
        # Record spikes for each step
        spikes = []
        
        # Simulation time
        current_time = time.time()
        
        for t in range(steps):
            # Neurons that spike this step
            spiking_neurons = self.membrane_potentials >= self.spike_thresholds
            spiking_indices = np.where(spiking_neurons)[0]
            
            if len(spiking_indices) > 0:
                # Record spike times for STDP
                self.last_spike_times[spiking_indices] = current_time + 0.001 * t
                
                # Record spikes
                step_spikes = [(i, self.membrane_potentials[i]) for i in spiking_indices]
                spikes.append(step_spikes)
                
                # Reset membrane potential of spiking neurons
                self.membrane_potentials[spiking_neurons] = 0.0
                
                # Propagate to connected neurons (sparse matrix multiplication)
                weights_slice = self.synaptic_weights[:, spiking_indices]
                delta_potentials = np.array(weights_slice.sum(axis=1)).flatten()
                self.membrane_potentials += delta_potentials
            else:
                # No spikes this step
                spikes.append([])
            
            # Apply decay
            self.membrane_potentials *= self.decay_rate
        
        # Update region activations based on spike activity
        self._update_region_activations(spikes)
        
        return spikes
    
    def _update_region_activations(self, spikes):
        """Update region activations based on spike activity"""
        # Count spikes per neuron
        spike_counts = defaultdict(int)
        for step_spikes in spikes:
            for neuron_idx, _ in step_spikes:
                spike_counts[neuron_idx] += 1
        
        # Update region activations
        for region_name, region in self.regions.items():
            region_neurons = region['neurons']
            if not region_neurons:
                continue
                
            # Calculate average activation in region
            region_spikes = sum(spike_counts.get(n, 0) for n in region_neurons)
            if region_spikes > 0:
                region['activation'] = min(1.0, region_spikes / (len(region_neurons) * 0.5))
            else:
                # Apply decay
                region['activation'] *= 0.8
    
    def _generalize_from_input(self, input_embedding):
        """Perform statistical generalization from input embedding"""
        # Find similar concepts
        similar_concepts = self.find_similar_concepts(query_embedding=input_embedding, top_k=5)
        
        if not similar_concepts:
            return {'generalized_concepts': [], 'confidence': 0.0}
        
        # Extract common features/patterns from similar concepts
        generalizations = []
        
        # Find common clusters for similar concepts
        common_clusters = []
        for concept, _ in similar_concepts:
            # Find clusters containing this concept
            concept_clusters = [
                cluster_name for cluster_name, concepts in self.concept_clusters.items()
                if concept in concepts
            ]
            common_clusters.extend(concept_clusters)
        
        # Count occurrences of each cluster
        cluster_counts = defaultdict(int)
        for cluster in common_clusters:
            cluster_counts[cluster] += 1
            
        # Get clusters with highest counts
        top_clusters = sorted(
            cluster_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]  # Top 3 clusters
        
        # Extract concepts from top clusters (excluding those already in similar_concepts)
        similar_concept_names = [c for c, _ in similar_concepts]
        for cluster_name, count in top_clusters:
            cluster_concepts = self.concept_clusters[cluster_name]
            # Calculate confidence based on count
            confidence = min(0.9, count / len(similar_concepts))
            
            for concept in cluster_concepts:
                if concept not in similar_concept_names:
                    generalizations.append((concept, confidence))
        
        # If no generalizations found, create a new embedding
        if not generalizations:
            # Create a generalized embedding by averaging similar concepts
            similar_embeddings = [
                self.concept_embeddings[concept] for concept, _ in similar_concepts
                if concept in self.concept_embeddings
            ]
            
            if similar_embeddings:
                generalized_embedding = np.mean(similar_embeddings, axis=0)
                confidence = 0.5  # Medium confidence for manufactured generalization
                
                # Find closest concept to this generalized embedding
                best_concept = None
                best_similarity = -1
                
                for concept, embedding in self.concept_embeddings.items():
                    if concept in similar_concept_names:
                        continue
                        
                    similarity = np.dot(generalized_embedding, embedding) / (
                        np.linalg.norm(generalized_embedding) * np.linalg.norm(embedding)
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_concept = concept
                
                if best_concept and best_similarity > 0.6:
                    generalizations.append((best_concept, best_similarity))
        
        return {
            'generalized_concepts': generalizations,
            'common_clusters': [c for c, _ in top_clusters] if top_clusters else [],
            'confidence': max([conf for _, conf in generalizations]) if generalizations else 0.0
        }
    
    def _complete_pattern(self, partial_embedding):
        """Complete a partial pattern using attractor dynamics"""
        # Find closest attractor state
        best_attractor = None
        best_similarity = -1
        
        for attractor_name, attractor_state in self.attractor_states.items():
            similarity = np.dot(partial_embedding, attractor_state) / (
                np.linalg.norm(partial_embedding) * np.linalg.norm(attractor_state)
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_attractor = attractor_name
        
        # If good match found, use attractor state
        if best_similarity > 0.7 and best_attractor is not None:
            completed_embedding = self.attractor_states[best_attractor]
            # Find concepts similar to completed embedding
            similar_concepts = self.find_similar_concepts(
                query_embedding=completed_embedding, top_k=3
            )
            
            return {
                'completed_embedding': completed_embedding,
                'attractor': best_attractor,
                'similar_concepts': similar_concepts,
                'confidence': best_similarity
            }
        
        # Otherwise, use pattern memory to complete
        best_pattern = None
        best_similarity = -1
        
        for stored_pattern in self.pattern_memory:
            pattern_embedding = stored_pattern.get('embedding')
            if pattern_embedding is None:
                continue
                
            similarity = np.dot(partial_embedding, pattern_embedding) / (
                np.linalg.norm(partial_embedding) * np.linalg.norm(pattern_embedding)
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_pattern = stored_pattern
        
        # If good pattern match found, use it
        if best_similarity > 0.6 and best_pattern is not None:
            completed_embedding = best_pattern['embedding']
            
            # Find concepts similar to completed embedding
            similar_concepts = self.find_similar_concepts(
                query_embedding=completed_embedding, top_k=3
            )
            
            return {
                'completed_embedding': completed_embedding,
                'pattern': best_pattern.get('name', 'unknown'),
                'similar_concepts': similar_concepts,
                'confidence': best_similarity
            }
        
        # If no good matches, create completion using averages of closest concepts
        similar_concepts = self.find_similar_concepts(
            query_embedding=partial_embedding, top_k=5
        )
        
        if similar_concepts:
            # Average embeddings of similar concepts
            similar_embeddings = [
                self.concept_embeddings[concept] for concept, _ in similar_concepts
                if concept in self.concept_embeddings
            ]
            
            if similar_embeddings:
                completed_embedding = np.mean(similar_embeddings, axis=0)
                # Calculate confidence as average similarity to similar concepts
                confidence = sum([sim for _, sim in similar_concepts]) / len(similar_concepts)
                
                return {
                    'completed_embedding': completed_embedding,
                    'similar_concepts': similar_concepts,
                    'confidence': confidence * 0.8  # Reduce confidence for manufactured completion
                }
        
        # Default if no completion could be found
        return {
            'completed_embedding': partial_embedding,  # Just return input
            'similar_concepts': [],
            'confidence': 0.1  # Very low confidence
        }
    
    def _solve_analogy(self, A, B, C):
        """Solve analogies of the form: A is to B as C is to ?"""
        # Get embeddings for A, B, and C
        embedding_A = self._get_input_embedding(A)
        embedding_B = self._get_input_embedding(B)
        embedding_C = self._get_input_embedding(C)
        
        # Calculate analogy using vector arithmetic: D ≈ B - A + C
        embedding_D = embedding_B - embedding_A + embedding_C
        
        # Normalize resulting embedding
        embedding_D = embedding_D / (np.linalg.norm(embedding_D) + 1e-8)
        
        # Find concepts similar to predicted D
        similar_concepts = self.find_similar_concepts(
            query_embedding=embedding_D, top_k=5
        )
        
        # Calculate confidence based on similarity of top match
        confidence = similar_concepts[0][1] if similar_concepts else 0.3
        
        return {
            'predicted_embedding': embedding_D,
            'predicted_concepts': similar_concepts,
            'confidence': confidence,
            'explanation': f"{A} is to {B} as {C} is to {similar_concepts[0][0] if similar_concepts else 'unknown'}"
        }
    
    def learn_from_examples(self, examples, concept_name=None, learning_type='few_shot'):
        """
        Learn from examples with statistical generalization
        
        Args:
            examples: List of examples to learn from
            concept_name: Name of concept being learned (optional)
            learning_type: Type of learning to perform
                - 'few_shot': Learn from few examples
                - 'incremental': Incrementally update existing concept
                - 'clustering': Learn by clustering examples
                
        Returns:
            Dictionary with learning results
        """
        # Convert examples to embeddings
        example_embeddings = [self._get_input_embedding(ex) for ex in examples]
        
        # Normalize embeddings
        example_embeddings = [
            embedding / (np.linalg.norm(embedding) + 1e-8)
            for embedding in example_embeddings
        ]
        
        if learning_type == 'few_shot':
            # Few-shot learning (learn new concept from few examples)
            if not concept_name:
                concept_name = f"concept_{len(self.concept_embeddings)}"
                
            # Average example embeddings
            concept_embedding = np.mean(example_embeddings, axis=0)
            
            # Find similar existing concepts for statistical generalization
            similar_concepts = []
            for embedding in example_embeddings:
                similar = self.find_similar_concepts(query_embedding=embedding, top_k=3)
                similar_concepts.extend([c for c, _ in similar])
            
            # Learn concept embedding with related concepts for generalization
            embedding = self.learn_concept_embedding(
                concept_name, 
                features=concept_embedding,
                related_concepts=similar_concepts
            )
            
            # Activate adaptation region for learning
            self._region_activate('adaptation', 0.9)
            
            return {
                'concept': concept_name,
                'embedding': embedding,
                'similar_concepts': similar_concepts,
                'learning_type': learning_type
            }
            
        elif learning_type == 'incremental':
            # Incremental learning (update existing concept)
            if concept_name and concept_name in self.concept_embeddings:
                existing_embedding = self.concept_embeddings[concept_name]
                
                # Update with weighted average of existing and new examples
                # (80% existing, 20% new examples)
                updated_embedding = 0.8 * existing_embedding + \
                                    0.2 * np.mean(example_embeddings, axis=0)
                
                # Normalize updated embedding
                updated_embedding = updated_embedding / (np.linalg.norm(updated_embedding) + 1e-8)
                
                # Store updated embedding
                self.concept_embeddings[concept_name] = updated_embedding
                
                # Update concept clusters
                self._update_concept_clusters(concept_name, updated_embedding)
                
                # Activate adaptation region for learning
                self._region_activate('adaptation', 0.7)
                
                return {
                    'concept': concept_name,
                    'embedding': updated_embedding,
                    'learning_type': learning_type,
                    'update_strength': 0.2  # 20% update strength
                }
            else:
                # Fall back to few-shot learning if concept doesn't exist
                return self.learn_from_examples(examples, concept_name, 'few_shot')
                
        elif learning_type == 'clustering':
            # Learning by clustering examples
            # Create embeddings for all examples
            if not example_embeddings:
                return {'error': 'No valid examples provided for clustering'}
                
            # Perform clustering on examples
            clusters = self._cluster_embeddings(example_embeddings)
            
            # Create a concept for each significant cluster
            concepts_created = []
            
            for i, (cluster_center, cluster_examples) in enumerate(clusters):
                # Skip tiny clusters
                if len(cluster_examples) < 2:
                    continue
                    
                # Create concept name if not provided
                cluster_concept = f"{concept_name}_cluster_{i}" if concept_name else f"cluster_{len(self.concept_clusters)}"
                
                # Learn concept embedding for this cluster
                embedding = self.learn_concept_embedding(
                    cluster_concept,
                    features=cluster_center,
                    related_concepts=[]  # No related concepts for clustering
                )
                
                concepts_created.append({
                    'concept': cluster_concept,
                    'embedding': embedding,
                    'examples_count': len(cluster_examples)
                })
            
            # Activate adaptation and generalization regions
            self._region_activate('adaptation', 0.7)
            self._region_activate('generalization', 0.8)
            
            return {
                'clusters_found': len(concepts_created),
                'concepts_created': concepts_created,
                'learning_type': learning_type
            }
        
        else:
            return {'error': f'Unknown learning type: {learning_type}'}
    
    def _cluster_embeddings(self, embeddings, distance_threshold=0.3):
        """Cluster embeddings using a simple hierarchical approach"""
        if not embeddings:
            return []
            
        # Initialize clusters with first embedding
        clusters = [(embeddings[0], [0])]  # (center, example indices)
        
        # Assign remaining embeddings to clusters or create new ones
        for i in range(1, len(embeddings)):
            embedding = embeddings[i]
            
            # Find closest cluster
            best_cluster_idx = -1
            best_distance = float('inf')
            
            for j, (cluster_center, _) in enumerate(clusters):
                # Calculate cosine distance (1 - similarity)
                distance = 1.0 - np.dot(embedding, cluster_center) / (
                    np.linalg.norm(embedding) * np.linalg.norm(cluster_center)
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_cluster_idx = j
            
            # If close enough to existing cluster, add to it
            if best_distance < distance_threshold:
                center, examples = clusters[best_cluster_idx]
                examples.append(i)
                
                # Update cluster center (average of all examples)
                cluster_embeddings = [embeddings[idx] for idx in examples]
                new_center = np.mean(cluster_embeddings, axis=0)
                new_center = new_center / (np.linalg.norm(new_center) + 1e-8)
                
                clusters[best_cluster_idx] = (new_center, examples)
            else:
                # Create new cluster
                clusters.append((embedding, [i]))
        
        # Convert indices to actual embeddings
        result_clusters = []
        for center, example_indices in clusters:
            example_embeddings = [embeddings[idx] for idx in example_indices]
            result_clusters.append((center, example_embeddings))
        
        return result_clusters
    
    def apply_feedback(self, feedback_value, active_concepts=None):
        """
        Apply reinforcement learning based on feedback
        
        Args:
            feedback_value: Value between -1.0 (negative) and 1.0 (positive)
            active_concepts: List of concepts that were active
            
        Returns:
            Dictionary with feedback application results
        """
        # Default to most recently activated concepts if none provided
        if active_concepts is None:
            # Find concepts with highest similarity to recent activations
            if not self.activation_history:
                return {'success': False, 'error': 'No recent activations'}
                
            recent_activation = self.activation_history[-1]
            # Convert to embedding format
            recent_embedding = np.zeros(self.embedding_dim)
            
            embedding_region = self.regions['embedding']['neurons']
            for i, neuron_idx in enumerate(embedding_region[:self.embedding_dim]):
                if i < self.embedding_dim:
                    # Rescale from activation (0.5-1.0) to embedding (-1.0 to 1.0)
                    if recent_activation[neuron_idx] > 0:
                        recent_embedding[i] = (recent_activation[neuron_idx] - 0.5) * 2.0
            
            # Find similar concepts to this activation
            similar_concepts = self.find_similar_concepts(
                query_embedding=recent_embedding, top_k=3
            )
            
            active_concepts = [concept for concept, _ in similar_concepts]
        
        # Apply feedback to active concepts
        modified_concepts = []
        
        for concept in active_concepts:
            if concept not in self.concept_embeddings:
                continue
                
            embedding = self.concept_embeddings[concept]
            
            if feedback_value > 0:
                # Positive feedback - reinforce concept by making it more distinct
                # Increase magnitude slightly
                enhanced_embedding = embedding * (1.0 + feedback_value * 0.1)
                # Re-normalize
                enhanced_embedding = enhanced_embedding / (np.linalg.norm(enhanced_embedding) + 1e-8)
                
                # Update embedding
                self.concept_embeddings[concept] = enhanced_embedding
                modified_concepts.append((concept, 'reinforced'))
            else:
                # Negative feedback - make concept less distinct by moving toward average
                # Get average embedding across all concepts
                all_embeddings = list(self.concept_embeddings.values())
                avg_embedding = np.mean(all_embeddings, axis=0)
                avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
                
                # Move slightly toward average (dilution)
                dilution_strength = abs(feedback_value) * 0.1
                diluted_embedding = (1.0 - dilution_strength) * embedding + dilution_strength * avg_embedding
                # Re-normalize
                diluted_embedding = diluted_embedding / (np.linalg.norm(diluted_embedding) + 1e-8)
                
                # Update embedding
                self.concept_embeddings[concept] = diluted_embedding
                modified_concepts.append((concept, 'diluted'))
        
        # Update value estimate with feedback for reinforcement learning
        self.value_estimate = 0.9 * self.value_estimate + 0.1 * feedback_value
        
        # Activate adaptation region
        self._region_activate('adaptation', 0.7)
        
        return {
            'success': True,
            'modified_concepts': modified_concepts,
            'feedback_value': feedback_value,
            'value_estimate': self.value_estimate
        }
    
    def generate_concept(self, base_concepts=None, attributes=None, creativity=0.5):
        """
        Generate a new concept through statistical combination
        
        Args:
            base_concepts: List of concepts to combine
            attributes: Specific attributes to incorporate
            creativity: How creative to be (0.0-1.0)
            
        Returns:
            Dictionary with the generated concept
        """
        # Initialize with a random embedding if no base concepts
        if not base_concepts and not attributes:
            # Generate random embedding with specific variance based on creativity
            variance = 0.1 + creativity * 0.2
            embedding = np.random.normal(0, variance, self.embedding_dim)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            # Find similar existing concepts
            similar_concepts = self.find_similar_concepts(query_embedding=embedding, top_k=3)
            
            # Generate a name based on similar concepts
            if similar_concepts:
                concept_name = f"novel_{similar_concepts[0][0]}"
            else:
                concept_name = f"novel_concept_{len(self.concept_embeddings)}"
            
            # Store the embedding
            self.concept_embeddings[concept_name] = embedding
            
            return {
                'concept': concept_name,
                'embedding': embedding,
                'similar_concepts': similar_concepts,
                'generation_method': 'random'
            }
        
        # Combine base concepts if provided
        if base_concepts:
            # Get embeddings for base concepts
            base_embeddings = []
            for concept in base_concepts:
                if concept in self.concept_embeddings:
                    base_embeddings.append(self.concept_embeddings[concept])
            
            if base_embeddings:
                # Average base embeddings
                combined = np.mean(base_embeddings, axis=0)
                
                # Add creative variation
                noise = np.random.normal(0, creativity * 0.2, self.embedding_dim)
                combined = combined + noise
                
                # Normalize
                combined = combined / (np.linalg.norm(combined) + 1e-8)
                
                # Find similar concepts for naming
                similar_concepts = self.find_similar_concepts(query_embedding=combined, top_k=3)
                
                # Generate name by combining base concepts
                concept_name = "_".join(base_concepts)
                
                # Store the embedding
                self.concept_embeddings[concept_name] = combined
                
                return {
                    'concept': concept_name,
                    'embedding': combined,
                    'base_concepts': base_concepts,
                    'similar_concepts': similar_concepts,
                    'generation_method': 'combination'
                }
        
        # Generate from attributes if provided
        if attributes:
            # Convert attributes to embedding space
            attribute_embeddings = []
            for attr in attributes:
                if isinstance(attr, str) and attr in self.concept_embeddings:
                    attribute_embeddings.append(self.concept_embeddings[attr])
                elif isinstance(attr, np.ndarray) and attr.shape[0] == self.embedding_dim:
                    attribute_embeddings.append(attr)
            
            if attribute_embeddings:
                # Combine attributes with weights based on position
                weighted_sum = np.zeros(self.embedding_dim)
                weights_sum = 0
                
                for i, attr_embedding in enumerate(attribute_embeddings):
                    # Later attributes get slightly higher weight
                    weight = 1.0 + i * 0.1
                    weighted_sum += weight * attr_embedding
                    weights_sum += weight
                
                # Normalize by weights
                if weights_sum > 0:
                    combined = weighted_sum / weights_sum
                else:
                    combined = np.random.normal(0, 0.1, self.embedding_dim)
                
                # Add creative variation
                noise = np.random.normal(0, creativity * 0.2, self.embedding_dim)
                combined = combined + noise
                
                # Normalize
                combined = combined / (np.linalg.norm(combined) + 1e-8)
                
                # Find similar existing concepts
                similar_concepts = self.find_similar_concepts(query_embedding=combined, top_k=3)
                
                # Generate name from attributes
                if isinstance(attributes[0], str):
                    concept_name = "_".join(attributes[:2])  # Use first two attributes
                else:
                    concept_name = f"attr_concept_{len(self.concept_embeddings)}"
                
                # Store the embedding
                self.concept_embeddings[concept_name] = combined
                
                return {
                    'concept': concept_name,
                    'embedding': combined,
                    'attributes': attributes if all(isinstance(a, str) for a in attributes) else None,
                    'similar_concepts': similar_concepts,
                    'generation_method': 'attributes'
                }
        
        # Default fallback
        return {'error': 'Could not generate concept with provided inputs'}
    
    def save_state(self, filepath):
        """Save the model state to a file"""
        import pickle
        
        state = {
            'concept_embeddings': self.concept_embeddings,
            'concept_clusters': dict(self.concept_clusters),
            'pattern_memory': self.pattern_memory,
            'attractor_states': self.attractor_states
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
            
        return True
    
    def load_state(self, filepath):
        """Load the model state from a file"""
        import pickle
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                
            self.concept_embeddings = state['concept_embeddings']
            self.concept_clusters = defaultdict(list, state['concept_clusters'])
            self.pattern_memory = state['pattern_memory']
            self.attractor_states = state['attractor_states']
            
            return True
        except Exception as e:
            print(f"[StatSNN] Error loading state: {e}")
            return False
            # Learning by clustering examples