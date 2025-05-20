"""
Enhanced concept embedding system for the Cogmenta Core.
Provides concept-based rather than token-based representation of meaning.
Supports concept hierarchies, relationships, and associative networks.
"""

import numpy as np
import re
import json
from collections import defaultdict
import time
from scipy.spatial.distance import cosine

class ConceptEmbeddingSystem:
    """
    Enhanced concept representation system that provides deeper semantic understanding
    than token-based approaches by representing meanings as structured conceptual entities.
    """
    
    def __init__(self, embedding_dim=300, use_pretrained=False, pretrained_path=None):
        """
        Initialize the concept embedding system.
        
        Args:
            embedding_dim: Dimension of concept embeddings
            use_pretrained: Whether to load pretrained embeddings
            pretrained_path: Path to pretrained embeddings
        """
        self.embedding_dim = embedding_dim
        self.concepts = {}  # name -> vector
        self.concept_metadata = {}  # name -> metadata dict
        self.concept_hierarchy = defaultdict(list)  # concept -> list of (parent, weight) tuples
        self.child_concepts = defaultdict(list)  # parent -> list of (child, weight) tuples
        self.concept_relations = defaultdict(list)  # concept -> list of (related_concept, relation_type, weight)
        self.relation_types = set(["is_a", "has_part", "similar_to", "opposite_of", "used_for", "causes", "located_at"])
        
        # Maps from text to concepts
        self.text_to_concept_map = {}  # Exact matches
        self.text_to_concept_patterns = []  # (regex, concept_name) tuples
        
        # Concept clusters for related meanings
        self.concept_clusters = {}  # cluster_name -> list of concept names
        
        # For tracking conceptual activations
        self.active_concepts = set()
        self.activation_levels = {}  # concept_name -> activation (0-1)
        self.activation_history = defaultdict(list)  # concept_name -> list of (time, activation) tuples
        
        # For similarity caching
        self.similarity_cache = {}
        
        # Load pretrained embeddings if specified
        if use_pretrained and pretrained_path:
            self._load_pretrained_embeddings(pretrained_path)
    
    def create_concept(self, concept_name, definition=None, vector=None, category=None, attributes=None):
        """
        Create a new concept with embedding vector and metadata.
        
        Args:
            concept_name: Name of the concept
            definition: Text definition of the concept
            vector: Embedding vector (optional)
            category: Category of the concept 
            attributes: Dictionary of concept attributes
            
        Returns:
            The concept name
        """
        concept_name = concept_name.lower()
        
        # Generate vector if not provided
        if vector is None:
            # Generate a random normalized vector
            vector = np.random.randn(self.embedding_dim)
            vector = vector / np.linalg.norm(vector)
        else:
            # Ensure vector has the right dimension
            if len(vector) != self.embedding_dim:
                raise ValueError(f"Vector dimension mismatch. Expected {self.embedding_dim}, got {len(vector)}")
            
            # Ensure vector is normalized
            vector = vector / np.linalg.norm(vector)
            
        # Store concept vector
        self.concepts[concept_name] = vector
        
        # Create and store metadata
        metadata = {
            "name": concept_name,
            "definition": definition or "",
            "category": category or "general",
            "attributes": attributes or {},
            "creation_time": time.time()
        }
        self.concept_metadata[concept_name] = metadata
        
        # Map the concept name to itself
        self.text_to_concept_map[concept_name] = concept_name
        
        return concept_name
    
    def add_concept_text_mapping(self, text, concept_name):
        """
        Map a text phrase to a concept.
        
        Args:
            text: Text phrase to map
            concept_name: Target concept
        """
        if concept_name not in self.concepts:
            raise ValueError(f"Concept '{concept_name}' does not exist")
            
        # Add to exact match mapping
        self.text_to_concept_map[text.lower()] = concept_name
        
        # Add variations with simple plurals/singulars
        text_lower = text.lower()
        if text_lower.endswith('s'):
            self.text_to_concept_map[text_lower[:-1]] = concept_name
        else:
            self.text_to_concept_map[text_lower + 's'] = concept_name
    
    def add_concept_pattern(self, pattern, concept_name):
        """
        Add a regex pattern that maps to a concept.
        
        Args:
            pattern: Regex pattern as string
            concept_name: Target concept
        """
        if concept_name not in self.concepts:
            raise ValueError(f"Concept '{concept_name}' does not exist")
            
        # Compile regex and add to patterns
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            self.text_to_concept_patterns.append((regex, concept_name))
        except re.error:
            raise ValueError(f"Invalid regex pattern: {pattern}")
    
    def add_concept_hierarchy_relation(self, concept_name, parent_concept, weight=1.0):
        """
        Add a hierarchical relationship between concepts.
        
        Args:
            concept_name: Child concept
            parent_concept: Parent concept
            weight: Strength of relation (0-1)
        """
        concept_name = concept_name.lower()
        parent_concept = parent_concept.lower()
        
        # Ensure both concepts exist
        if concept_name not in self.concepts:
            raise ValueError(f"Concept '{concept_name}' does not exist")
        if parent_concept not in self.concepts:
            raise ValueError(f"Concept '{parent_concept}' does not exist")
        
        # Prevent circular hierarchies
        if self._is_ancestor(concept_name, parent_concept):
            raise ValueError(f"Circular hierarchy detected: {parent_concept} is already a descendant of {concept_name}")
        
        # Add to hierarchy
        self.concept_hierarchy[concept_name].append((parent_concept, weight))
        self.child_concepts[parent_concept].append((concept_name, weight))
        
        # Also add as relation for completeness
        self.add_concept_relation(concept_name, parent_concept, "is_a", weight)
    
    def add_concept_relation(self, concept1, concept2, relation_type, weight=1.0):
        """
        Add a relation between concepts.
        
        Args:
            concept1: First concept
            concept2: Second concept
            relation_type: Type of relation
            weight: Strength of relation (0-1)
        """
        concept1 = concept1.lower()
        concept2 = concept2.lower()
        
        # Ensure both concepts exist
        if concept1 not in self.concepts:
            self.create_concept(concept1)
        if concept2 not in self.concepts:
            self.create_concept(concept2)
        
        # Add relation
        self.concept_relations[concept1].append((concept2, relation_type, weight))
        
        # Add inverse relation if appropriate
        if relation_type == "similar_to":
            self.concept_relations[concept2].append((concept1, relation_type, weight))
        elif relation_type == "opposite_of":
            self.concept_relations[concept2].append((concept1, relation_type, weight))
        elif relation_type == "is_a":
            self.concept_relations[concept2].append((concept1, "has_instance", weight))
        elif relation_type == "has_part":
            self.concept_relations[concept2].append((concept1, "part_of", weight))
        elif relation_type == "causes":
            self.concept_relations[concept2].append((concept1, "caused_by", weight))
    
    def create_concept_cluster(self, cluster_name, concepts):
        """
        Create a cluster of related concepts.
        
        Args:
            cluster_name: Name of the cluster
            concepts: List of concept names to include
        """
        # Verify all concepts exist
        for concept in concepts:
            if concept not in self.concepts:
                raise ValueError(f"Concept '{concept}' does not exist")
        
        # Create the cluster
        self.concept_clusters[cluster_name] = list(concepts)
        
        # Add mutual similar_to relations with medium weight
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                self.add_concept_relation(concept1, concept2, "similar_to", 0.7)
    
    def get_concept_vector(self, concept_name):
        """
        Get the embedding vector for a concept.
        
        Args:
            concept_name: Name of the concept
            
        Returns:
            Numpy array with concept vector
        """
        concept_name = concept_name.lower()
        if concept_name not in self.concepts:
            raise ValueError(f"Concept '{concept_name}' does not exist")
            
        return self.concepts[concept_name]
    
    def get_concept_from_text(self, text, min_similarity=0.7):
        """
        Get the most relevant concept for a text phrase.
        
        Args:
            text: Text to find concept for
            min_similarity: Minimum similarity threshold
            
        Returns:
            Tuple of (concept_name, similarity_score)
        """
        text_lower = text.lower()
        
        # Check for exact match
        if text_lower in self.text_to_concept_map:
            return (self.text_to_concept_map[text_lower], 1.0)
        
        # Check regex patterns
        for pattern, concept_name in self.text_to_concept_patterns:
            if pattern.search(text_lower):
                return (concept_name, 0.9)  # High confidence for regex matches
        
        # Try fuzzy matching
        best_match = None
        best_score = 0
        
        # First try word overlap for multi-word phrases
        if ' ' in text_lower:
            words = set(text_lower.split())
            for phrase, concept in self.text_to_concept_map.items():
                if ' ' in phrase:
                    phrase_words = set(phrase.split())
                    overlap = len(words.intersection(phrase_words)) / max(len(words), len(phrase_words))
                    if overlap > best_score:
                        best_score = overlap
                        best_match = concept
        
        # If good enough match found
        if best_score >= min_similarity:
            return (best_match, best_score)
        
        # Try substring matching
        for phrase, concept in self.text_to_concept_map.items():
            if phrase in text_lower or text_lower in phrase:
                sim_score = len(min(phrase, text_lower)) / len(max(phrase, text_lower))
                if sim_score > best_score:
                    best_score = sim_score
                    best_match = concept
        
        # Return best match if above threshold
        if best_score >= min_similarity:
            return (best_match, best_score)
            
        # No good match found
        return (None, 0)
    
    def extract_concepts_from_text(self, text, min_confidence=0.6):
        """
        Extract all concepts from a text passage.
        
        Args:
            text: Text to analyze
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of (concept_name, confidence, text_span) tuples
        """
        concepts = []
        text_lower = text.lower()
        
        # Check for exact matches of multi-word phrases first
        for phrase, concept in self.text_to_concept_map.items():
            if ' ' in phrase and phrase in text_lower:
                for match in re.finditer(re.escape(phrase), text_lower):
                    start, end = match.span()
                    concepts.append((concept, 1.0, text[start:end]))
        
        # Check regex patterns
        for pattern, concept_name in self.text_to_concept_patterns:
            for match in pattern.finditer(text):
                start, end = match.span()
                concepts.append((concept_name, 0.9, text[start:end]))
        
        # Check single words against the concept map
        words = re.findall(r'\b\w+\b', text_lower)
        for word in words:
            if word in self.text_to_concept_map:
                # Find the position of the word in the original text
                for match in re.finditer(r'\b' + re.escape(word) + r'\b', text_lower):
                    start, end = match.span()
                    concepts.append((self.text_to_concept_map[word], 1.0, text[start:end]))
        
        # Filter and sort by confidence
        filtered_concepts = [c for c in concepts if c[1] >= min_confidence]
        filtered_concepts.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_concepts
        
    def calculate_concept_similarity(self, concept1, concept2):
        """
        Calculate semantic similarity between two concepts.
        
        Args:
            concept1: First concept name
            concept2: Second concept name
            
        Returns:
            Similarity score (0-1)
        """
        concept1 = concept1.lower()
        concept2 = concept2.lower()
        
        # Check cache
        cache_key = tuple(sorted([concept1, concept2]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # If either concept doesn't exist, return 0
        if concept1 not in self.concepts or concept2 not in self.concepts:
            return 0.0
            
        # Same concept has similarity 1
        if concept1 == concept2:
            return 1.0
            
        # Calculate vector similarity (cosine similarity)
        vec1 = self.concepts[concept1]
        vec2 = self.concepts[concept2]
        vec_similarity = 1 - cosine(vec1, vec2)
        
        # Check for direct similar_to relations
        relation_similarity = 0
        for related, rel_type, weight in self.concept_relations[concept1]:
            if related == concept2:
                if rel_type == "similar_to":
                    relation_similarity = max(relation_similarity, weight)
                elif rel_type == "is_a" or rel_type == "has_part":
                    relation_similarity = max(relation_similarity, weight * 0.8)
                    
        # Check hierarchy (parent/child relationships boost similarity)
        hierarchy_similarity = 0
        
        # Check if concept1 is a descendant of concept2
        for parent, weight in self.concept_hierarchy[concept1]:
            if parent == concept2:
                hierarchy_similarity = max(hierarchy_similarity, weight * 0.8)
            else:
                # Check one level more of ancestry
                for grandparent, gp_weight in self.concept_hierarchy[parent]:
                    if grandparent == concept2:
                        hierarchy_similarity = max(hierarchy_similarity, weight * gp_weight * 0.6)
        
        # Check if concept2 is a descendant of concept1
        for parent, weight in self.concept_hierarchy[concept2]:
            if parent == concept1:
                hierarchy_similarity = max(hierarchy_similarity, weight * 0.8)
            else:
                # Check one level more of ancestry
                for grandparent, gp_weight in self.concept_hierarchy[parent]:
                    if grandparent == concept1:
                        hierarchy_similarity = max(hierarchy_similarity, weight * gp_weight * 0.6)
        
        # Check if they share parents
        concept1_parents = {parent for parent, _ in self.concept_hierarchy[concept1]}
        concept2_parents = {parent for parent, _ in self.concept_hierarchy[concept2]}
        common_parents = concept1_parents.intersection(concept2_parents)
        
        shared_parent_similarity = 0
        if common_parents:
            # Calculate average similarity based on common parents
            for parent in common_parents:
                # Find the weights
                weight1 = next((w for p, w in self.concept_hierarchy[concept1] if p == parent), 0)
                weight2 = next((w for p, w in self.concept_hierarchy[concept2] if p == parent), 0)
                shared_parent_similarity = max(shared_parent_similarity, (weight1 + weight2) / 2 * 0.7)
        
        # Calculate final similarity as a weighted combination
        similarity = (
            0.5 * vec_similarity +
            0.3 * relation_similarity +
            0.1 * hierarchy_similarity +
            0.1 * shared_parent_similarity
        )
        
        # Cache the result
        self.similarity_cache[cache_key] = similarity
        
        return similarity
    
    def activate_concept(self, concept_name, activation_level=1.0, spread_activation=True):
        """
        Activate a concept and optionally spread activation to related concepts.
        
        Args:
            concept_name: Name of the concept to activate
            activation_level: Level of activation (0-1)
            spread_activation: Whether to spread activation to related concepts
            
        Returns:
            Dictionary of activated concepts and their levels
        """
        concept_name = concept_name.lower()
        
        # Check if concept exists
        if concept_name not in self.concepts:
            return {}
            
        # Activate the concept
        self.active_concepts.add(concept_name)
        self.activation_levels[concept_name] = activation_level
        
        # Record activation in history
        self.activation_history[concept_name].append((time.time(), activation_level))
        
        # If not spreading activation, just return this concept
        if not spread_activation:
            return {concept_name: activation_level}
            
        # Spread activation to related concepts
        activated = {concept_name: activation_level}
        self._spread_activation(concept_name, activation_level, activated, depth=0, max_depth=2)
        
        return activated
    
    def _spread_activation(self, concept_name, activation_level, activated, depth, max_depth):
        """
        Recursively spread activation to related concepts.
        
        Args:
            concept_name: Current concept
            activation_level: Current activation level
            activated: Dictionary to track all activations
            depth: Current depth in spreading
            max_depth: Maximum depth to spread
            
        Returns:
            Updated activated dictionary
        """
        # Stop if we've reached max depth or the activation is too low
        if depth >= max_depth or activation_level < 0.2:
            return activated
            
        # Determine spread factor based on relation type
        relation_spread_factors = {
            "is_a": 0.7,
            "has_instance": 0.5,
            "has_part": 0.6,
            "part_of": 0.6,
            "similar_to": 0.8,
            "opposite_of": 0.4,
            "used_for": 0.5,
            "causes": 0.6,
            "caused_by": 0.6
        }
        
        # Spread to hierarchically related concepts
        for parent, weight in self.concept_hierarchy[concept_name]:
            spread_level = activation_level * weight * 0.7  # Reduced spreading to parents
            
            # Only spread if it would increase the current activation
            if spread_level > activated.get(parent, 0):
                activated[parent] = spread_level
                self.active_concepts.add(parent)
                self.activation_levels[parent] = max(spread_level, self.activation_levels.get(parent, 0))
                self.activation_history[parent].append((time.time(), spread_level))
                
                # Continue spreading from this concept
                self._spread_activation(parent, spread_level, activated, depth+1, max_depth)
        
        # Spread to child concepts
        for child, weight in self.child_concepts[concept_name]:
            spread_level = activation_level * weight * 0.8  # Stronger spreading to children
            
            # Only spread if it would increase the current activation
            if spread_level > activated.get(child, 0):
                activated[child] = spread_level
                self.active_concepts.add(child)
                self.activation_levels[child] = max(spread_level, self.activation_levels.get(child, 0))
                self.activation_history[child].append((time.time(), spread_level))
                
                # Continue spreading from this concept
                self._spread_activation(child, spread_level, activated, depth+1, max_depth)
        
        # Spread to related concepts
        for related, rel_type, weight in self.concept_relations[concept_name]:
            # Skip hierarchical relations as they're handled above
            if rel_type in ["is_a", "has_instance"]:
                continue
                
            # Get spread factor based on relation type
            spread_factor = relation_spread_factors.get(rel_type, 0.5)
            spread_level = activation_level * weight * spread_factor
            
            # Only spread if it would increase the current activation
            if spread_level > activated.get(related, 0):
                activated[related] = spread_level
                self.active_concepts.add(related)
                self.activation_levels[related] = max(spread_level, self.activation_levels.get(related, 0))
                self.activation_history[related].append((time.time(), spread_level))
                
                # Continue spreading from this concept
                self._spread_activation(related, spread_level, activated, depth+1, max_depth)
        
        return activated
    
    def get_most_activated_concepts(self, top_n=5, min_activation=0.2):
        """
        Get the most activated concepts.
        
        Args:
            top_n: Number of top concepts to return
            min_activation: Minimum activation threshold
            
        Returns:
            List of (concept_name, activation_level) tuples
        """
        # Filter concepts with minimum activation
        activated = [(c, a) for c, a in self.activation_levels.items() if a >= min_activation]
        
        # Sort by activation level (descending)
        activated.sort(key=lambda x: x[1], reverse=True)
        
        return activated[:top_n]
    
    def decay_activations(self, decay_factor=0.9):
        """
        Decay all concept activations.
        
        Args:
            decay_factor: Multiplier for current activations (0-1)
            
        Returns:
            Number of concepts still active after decay
        """
        # Apply decay to all activations
        for concept in list(self.activation_levels.keys()):
            new_level = self.activation_levels[concept] * decay_factor
            
            # Remove if below threshold
            if new_level < 0.1:
                self.activation_levels.pop(concept)
                if concept in self.active_concepts:
                    self.active_concepts.remove(concept)
            else:
                self.activation_levels[concept] = new_level
        
        return len(self.active_concepts)
    
    def build_meaning_graph(self, text):
        """
        Build a structured meaning graph from text.
        
        Args:
            text: Input text
            
        Returns:
            Graph structure with nodes and edges
        """
        # Extract concepts from text
        extracted_concepts = self.extract_concepts_from_text(text)
        
        # Create graph structure
        graph = {
            'nodes': [],
            'edges': []
        }
        
        # Track processed concepts to avoid duplicates
        processed_concepts = set()
        
        # Add extracted concepts as nodes
        for concept_name, confidence, text_span in extracted_concepts:
            if concept_name not in processed_concepts:
                # Get concept metadata
                metadata = self.concept_metadata.get(concept_name, {})
                
                # Create node
                node = {
                    'id': concept_name,
                    'label': concept_name,
                    'type': 'concept',
                    'category': metadata.get('category', 'general'),
                    'confidence': confidence,
                    'mentions': [text_span],
                    'definition': metadata.get('definition', '')
                }
                
                graph['nodes'].append(node)
                processed_concepts.add(concept_name)
                
                # Activate this concept
                self.activate_concept(concept_name, confidence)
            else:
                # Add mention to existing node
                for node in graph['nodes']:
                    if node['id'] == concept_name and text_span not in node['mentions']:
                        node['mentions'].append(text_span)
                        # Update confidence if higher
                        node['confidence'] = max(node['confidence'], confidence)
                        break
        
        # Get most activated concepts and add if not already in graph
        top_activated = self.get_most_activated_concepts(top_n=10)
        for concept_name, activation in top_activated:
            if concept_name not in processed_concepts and activation >= 0.4:
                # Get concept metadata
                metadata = self.concept_metadata.get(concept_name, {})
                
                # Create node
                node = {
                    'id': concept_name,
                    'label': concept_name,
                    'type': 'activated_concept',
                    'category': metadata.get('category', 'general'),
                    'confidence': activation,
                    'mentions': [],
                    'definition': metadata.get('definition', '')
                }
                
                graph['nodes'].append(node)
                processed_concepts.add(concept_name)
        
        # Add edges from concept relations
        edges_added = set()  # Track which edges have been added
        
        for concept_name in processed_concepts:
            # Add hierarchical relations
            for parent, weight in self.concept_hierarchy.get(concept_name, []):
                if parent in processed_concepts:
                    edge_id = f"{concept_name}_is_a_{parent}"
                    if edge_id not in edges_added:
                        graph['edges'].append({
                            'id': edge_id,
                            'source': concept_name,
                            'target': parent,
                            'label': 'is_a',
                            'weight': weight
                        })
                        edges_added.add(edge_id)
            
            # Add other concept relations
            for related, rel_type, weight in self.concept_relations.get(concept_name, []):
                if related in processed_concepts:
                    edge_id = f"{concept_name}_{rel_type}_{related}"
                    if edge_id not in edges_added:
                        graph['edges'].append({
                            'id': edge_id,
                            'source': concept_name,
                            'target': related,
                            'label': rel_type,
                            'weight': weight
                        })
                        edges_added.add(edge_id)
        
        return graph
    
    def get_concept_hierarchy_paths(self, concept_name, max_depth=3):
        """
        Get all paths up the concept hierarchy.
        
        Args:
            concept_name: Starting concept
            max_depth: Maximum depth to traverse
            
        Returns:
            List of paths, where each path is a list of concept names
        """
        concept_name = concept_name.lower()
        
        if concept_name not in self.concepts:
            return []
            
        paths = []
        self._build_hierarchy_paths(concept_name, [concept_name], paths, 0, max_depth)
        
        return paths
    
    def _build_hierarchy_paths(self, concept_name, current_path, all_paths, depth, max_depth):
        """
        Recursive helper to build hierarchy paths.
        
        Args:
            concept_name: Current concept
            current_path: Path so far
            all_paths: List to collect all paths
            depth: Current depth
            max_depth: Maximum depth
        """
        # Stop if max depth reached
        if depth >= max_depth:
            all_paths.append(current_path.copy())
            return
            
        # Get parents
        parents = self.concept_hierarchy.get(concept_name, [])
        
        if not parents:
            # This is a root concept
            all_paths.append(current_path.copy())
            return
            
        for parent, _ in parents:
            # Skip if this would create a cycle
            if parent in current_path:
                all_paths.append(current_path.copy())
                continue
                
            # Add this parent to the path and continue
            new_path = current_path.copy()
            new_path.append(parent)
            self._build_hierarchy_paths(parent, new_path, all_paths, depth+1, max_depth)
    
    def _is_ancestor(self, concept_name, potential_ancestor, visited=None):
        """
        Check if a concept is an ancestor of another in the hierarchy.
        
        Args:
            concept_name: Child concept
            potential_ancestor: Potential ancestor
            visited: Set of visited concepts (for cycle detection)
            
        Returns:
            Boolean indicating if potential_ancestor is an ancestor
        """
        if visited is None:
            visited = set()
            
        # Prevent infinite recursion
        if concept_name in visited:
            return False
            
        visited.add(concept_name)
        
        # Check direct parents
        for parent, _ in self.concept_hierarchy.get(concept_name, []):
            if parent == potential_ancestor:
                return True
            
            # Check parent's ancestors
            if self._is_ancestor(parent, potential_ancestor, visited):
                return True
                
        return False
    
    def find_related_concepts(self, concept_name, relation_types=None, min_weight=0.5):
        """
        Find concepts related to a given concept by specified relation types.
        
        Args:
            concept_name: Target concept name
            relation_types: List of relation types to include (None for all)
            min_weight: Minimum relation weight to include
            
        Returns:
            Dictionary mapping relation types to lists of (concept, weight) tuples
        """
        concept_name = concept_name.lower()
        
        if concept_name not in self.concepts:
            return {}
            
        # Filter relation types if specified
        if relation_types is None:
            relation_types = list(self.relation_types) + ["has_instance", "part_of"]
            
        # Get all relations for this concept
        related = defaultdict(list)
        
        # Add direct relations
        for related_concept, rel_type, weight in self.concept_relations.get(concept_name, []):
            if rel_type in relation_types and weight >= min_weight:
                related[rel_type].append((related_concept, weight))
                
        # Add hierarchical relations
        for parent, weight in self.concept_hierarchy.get(concept_name, []):
            if "is_a" in relation_types and weight >= min_weight:
                related["is_a"].append((parent, weight))
                
        for child, weight in self.child_concepts.get(concept_name, []):
            if "has_instance" in relation_types and weight >= min_weight:
                related["has_instance"].append((child, weight))
                
        # Sort each relation type by weight
        for rel_type in related:
            related[rel_type].sort(key=lambda x: x[1], reverse=True)
            
        return dict(related)
    
    def merge_concept_vectors(self, concept_names, weights=None):
        """
        Create a merged vector from multiple concepts.
        
        Args:
            concept_names: List of concept names to merge
            weights: Optional weights for each concept
            
        Returns:
            Merged normalized vector
        """
        if not concept_names:
            return np.zeros(self.embedding_dim)
            
        # Use equal weights if not specified
        if weights is None:
            weights = [1.0] * len(concept_names)
        elif len(weights) != len(concept_names):
            raise ValueError("Number of weights must match number of concepts")
            
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return np.zeros(self.embedding_dim)
            
        norm_weights = [w/total_weight for w in weights]
        
        # Compute weighted sum
        merged = np.zeros(self.embedding_dim)
        for concept, weight in zip(concept_names, norm_weights):
            if concept in self.concepts:
                merged += self.concepts[concept] * weight
                
        # Normalize the result
        norm = np.linalg.norm(merged)
        if norm > 0:
            merged = merged / norm
            
        return merged
    
    def find_concept_path(self, start_concept, end_concept, max_depth=4):
        """
        Find a path between two concepts through relations.
        
        Args:
            start_concept: Starting concept
            end_concept: Target concept
            max_depth: Maximum path length
            
        Returns:
            List of (concept, relation_type) tuples forming the path, or empty if no path
        """
        start_concept = start_concept.lower()
        end_concept = end_concept.lower()
        
        # Check if concepts exist
        if start_concept not in self.concepts or end_concept not in self.concepts:
            return []
            
        # Breadth-first search
        queue = [(start_concept, [])]  # (concept, path_so_far)
        visited = {start_concept}
        
        while queue:
            current, path = queue.pop(0)
            
            # Check if we've reached the target
            if current == end_concept:
                return path
                
            # Stop if max depth reached
            if len(path) >= max_depth:
                continue
                
            # Try hierarchical relations
            for parent, weight in self.concept_hierarchy.get(current, []):
                if parent not in visited:
                    visited.add(parent)
                    new_path = path + [(current, "is_a", parent)]
                    queue.append((parent, new_path))
                    
            for child, weight in self.child_concepts.get(current, []):
                if child not in visited:
                    visited.add(child)
                    new_path = path + [(current, "has_instance", child)]
                    queue.append((child, new_path))
                    
            # Try other relations
            for related, rel_type, weight in self.concept_relations.get(current, []):
                if related not in visited:
                    visited.add(related)
                    new_path = path + [(current, rel_type, related)]
                    queue.append((related, new_path))
                    
        # No path found
        return []
    
    def save_to_file(self, filename):
        """
        Save the concept embedding system to a file.
        
        Args:
            filename: Target filename
            
        Returns:
            Success flag
        """
        try:
            # Prepare data dictionary
            data = {
                "embedding_dim": self.embedding_dim,
                "concepts": {name: vec.tolist() for name, vec in self.concepts.items()},
                "concept_metadata": self.concept_metadata,
                "concept_hierarchy": {c: h for c, h in self.concept_hierarchy.items()},
                "concept_relations": {
                    c: [(r, t, w) for r, t, w in rels] 
                    for c, rels in self.concept_relations.items()
                },
                "text_to_concept_map": self.text_to_concept_map,
                "concept_clusters": self.concept_clusters
            }
            
            # Save as JSON
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            return True
            
        except Exception as e:
            print(f"Error saving concept system: {e}")
            return False
    
    def load_from_file(self, filename):
        """
        Load the concept embedding system from a file.
        
        Args:
            filename: Source filename
            
        Returns:
            Success flag
        """
        try:
            # Load JSON data
            with open(filename, 'r') as f:
                data = json.load(f)
                
            # Reset current state
            self.concepts = {}
            self.concept_metadata = {}
            self.concept_hierarchy = defaultdict(list)
            self.child_concepts = defaultdict(list)
            self.concept_relations = defaultdict(list)
            self.text_to_concept_map = {}
            self.concept_clusters = {}
            
            # Load embedding dimension
            self.embedding_dim = data.get("embedding_dim", 300)
            
            # Load concepts
            for name, vec_list in data.get("concepts", {}).items():
                self.concepts[name] = np.array(vec_list)
                
            # Load metadata
            self.concept_metadata = data.get("concept_metadata", {})
            
            # Load hierarchy
            for concept, parents in data.get("concept_hierarchy", {}).items():
                self.concept_hierarchy[concept] = parents
                # Also update child_concepts
                for parent, weight in parents:
                    self.child_concepts[parent].append((concept, weight))
                    
            # Load relations
            for concept, relations in data.get("concept_relations", {}).items():
                self.concept_relations[concept] = relations
                
            # Load text mappings
            self.text_to_concept_map = data.get("text_to_concept_map", {})
            
            # Load clusters
            self.concept_clusters = data.get("concept_clusters", {})
            
            return True
            
        except Exception as e:
            print(f"Error loading concept system: {e}")
            return False
    
    def _load_pretrained_embeddings(self, path):
        """
        Load pretrained word embeddings from a file.
        
        Args:
            path: Path to embeddings file
            
        Returns:
            Number of embeddings loaded
        """
        try:
            count = 0
            
            # Try to detect file format based on extension
            if path.endswith('.txt') or path.endswith('.vec'):
                # Assume word2vec text format
                with open(path, 'r', encoding='utf-8') as f:
                    # First line might have vocabulary size and dimension
                    header = f.readline().strip().split()
                    if len(header) == 2:
                        # This is a proper word2vec file with header
                        vocab_size, dim = map(int, header)
                        if dim != self.embedding_dim:
                            print(f"Warning: Embedding dimension mismatch. Expected {self.embedding_dim}, got {dim}")
                            self.embedding_dim = dim
                    else:
                        # No header, rewind to start
                        f.seek(0)
                        
                    # Process each line
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) <= self.embedding_dim:
                            continue  # Skip invalid lines
                            
                        word = parts[0].lower()
                        try:
                            vector = np.array([float(val) for val in parts[1:self.embedding_dim+1]])
                            # Normalize the vector
                            vector = vector / np.linalg.norm(vector)
                            
                            # Create concept
                            self.create_concept(word, vector=vector)
                            
                            # Map the word to itself
                            self.text_to_concept_map[word] = word
                            
                            count += 1
                        except ValueError:
                            continue  # Skip invalid vectors
            elif path.endswith('.json'):
                # Assume JSON format
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Check format
                    if isinstance(data, dict) and 'embeddings' in data:
                        for word, vec in data['embeddings'].items():
                            word = word.lower()
                            try:
                                vector = np.array(vec)
                                if len(vector) != self.embedding_dim:
                                    # Resize if needed
                                    if len(vector) > self.embedding_dim:
                                        vector = vector[:self.embedding_dim]
                                    else:
                                        # Pad with zeros
                                        padded = np.zeros(self.embedding_dim)
                                        padded[:len(vector)] = vector
                                        vector = padded
                                        
                                # Normalize
                                vector = vector / np.linalg.norm(vector)
                                
                                # Create concept
                                self.create_concept(word, vector=vector)
                                
                                # Map the word to itself
                                self.text_to_concept_map[word] = word
                                
                                count += 1
                            except (ValueError, TypeError):
                                continue
            
            print(f"Loaded {count} pretrained embeddings")
            return count
            
        except Exception as e:
            print(f"Error loading pretrained embeddings: {e}")
            return 0
            
    def get_active_concept_clusters(self, min_activation=0.3, min_concepts=2):
        """
        Identify which concept clusters are currently active.
        
        Args:
            min_activation: Minimum average activation level
            min_concepts: Minimum number of active concepts in cluster
            
        Returns:
            List of (cluster_name, avg_activation, active_concepts) tuples
        """
        active_clusters = []
        
        for cluster_name, concepts in self.concept_clusters.items():
            # Get activations for concepts in this cluster
            concept_activations = [
                (c, self.activation_levels.get(c, 0))
                for c in concepts if c in self.active_concepts
            ]
            
            # Skip if too few concepts are active
            if len(concept_activations) < min_concepts:
                continue
                
            # Calculate average activation
            avg_activation = sum(a for _, a in concept_activations) / len(concept_activations)
            
            # Skip if average activation is too low
            if avg_activation < min_activation:
                continue
                
            active_clusters.append((
                cluster_name,
                avg_activation,
                [c for c, _ in concept_activations]
            ))
            
        # Sort by average activation
        active_clusters.sort(key=lambda x: x[1], reverse=True)
        
        return active_clusters