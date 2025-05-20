# cogmenta_core/models/symbolic/vector_symbolic.py
import numpy as np
from scipy.spatial.distance import cosine
import time
import logging
import re
from collections import defaultdict

class VectorSymbolicEngine:
    """
    Vector Symbolic Architecture for representing language facts.
    Provides fuzzy matching, similarity-based reasoning, and graded truth values.
    """
    
    def __init__(self, dimension=1000, sparsity=0.1):
        """
        Initialize the vector symbolic engine.
        
        Args:
            dimension (int): Dimension of concept vectors
            sparsity (float): Sparsity of vectors (0-1)
        """
        self.logger = logging.getLogger(__name__)
        self.dimension = dimension
        self.sparsity = sparsity
        self.concept_vectors = {}
        self.relation_vectors = {}
        self.facts = []
        self.fact_index = {}  # For faster lookup
        
        # Cache for similarity calculations
        self.similarity_cache = {}
        
        self.logger.info(f"[VSA] Initialized with dimension={dimension}, sparsity={sparsity}")

    def init_from_conceptnet(self, conceptnet_path, max_concepts=100000):
        """Initialize vector space from ConceptNet embeddings"""
        self.logger.info(f"Loading ConceptNet embeddings from {conceptnet_path}")
        
        concepts_loaded = 0
        with open(conceptnet_path, 'r', encoding='utf-8') as f:
            for line in f:
                if concepts_loaded >= max_concepts:
                    break
                    
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                    
                # Extract relation, subject, object
                try:
                    relation = parts[1]
                    subject = parts[2].split('/')[-1]
                    object_val = parts[3].split('/')[-1]
                    
                    # Check if English concepts
                    if '/c/en/' in parts[2] and '/c/en/' in parts[3]:
                        # Create concept vectors
                        subj_vector = self.get_concept_vector(subject)
                        obj_vector = self.get_concept_vector(object_val)
                        
                        # Add relation
                        self.create_fact(subject, relation, object_val, confidence=0.8)
                        concepts_loaded += 1
                except Exception as e:
                    self.logger.warning(f"Error processing ConceptNet line: {e}")
                    
        self.logger.info(f"Loaded {concepts_loaded} concept relations from ConceptNet")
        return concepts_loaded

    def process_text(self, text):
        """
        Process text and reason about it.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Processing results
        """
        # Extract potential facts from text
        facts = self._extract_facts_from_text(text)
        
        if facts:
            # Store the facts
            for fact in facts:
                self.create_fact(fact['subject'], fact['predicate'], fact['object'], fact['confidence'])
                
            # Query related facts
            query_results = self.query_facts_by_text(text)
            
            # Format response
            response = self._format_response(query_results)
            
            return {
                'response': response,
                'facts': facts,
                'query_results': query_results
            }
        else:
            # If no facts extracted, try to query based on entities
            query_results = self.query_facts_by_text(text)
            
            if query_results:
                response = self._format_response(query_results)
                return {
                    'response': response,
                    'facts': [],
                    'query_results': query_results
                }
            
            return {
                'response': "I couldn't extract any relationships from that text.",
                'facts': [],
                'query_results': []
            }
    
    def _extract_facts_from_text(self, text):
        """
        Extract potential facts from text using pattern matching.
        
        Args:
            text (str): Input text
            
        Returns:
            list: Extracted facts
        """
        facts = []
        
        # Look for simple "X likes/hates/knows Y" patterns
        patterns = [
            r"(\w+)\s+(likes|loves|hates|knows|trusts|fears|avoids)\s+(\w+)",
            r"(\w+)\s+(is|seems|appears)\s+(happy|sad|angry|excited|afraid)"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) == 3:
                    subject, predicate, object_val = groups
                    facts.append({
                        'subject': subject.lower(),
                        'predicate': predicate.lower(),
                        'object': object_val.lower(),
                        'confidence': 0.9
                    })
        
        # Look for "X doesn't like Y" patterns
        negation_patterns = [
            r"(\w+)\s+(?:doesn't|does\s+not|didn't|did\s+not)\s+(like|love|hate|know|trust|fear|avoid)\s+(\w+)"
        ]
        
        for pattern in negation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) == 3:
                    subject, predicate, object_val = groups
                    facts.append({
                        'subject': subject.lower(),
                        'predicate': f"not_{predicate.lower()}",
                        'object': object_val.lower(),
                        'confidence': 0.8
                    })
        
        # Look for "X is a Y" patterns
        is_a_patterns = [
            r"(\w+)\s+is\s+(?:a|an)\s+(\w+)"
        ]
        
        for pattern in is_a_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) == 2:
                    subject, object_val = groups
                    facts.append({
                        'subject': subject.lower(),
                        'predicate': "is_a",
                        'object': object_val.lower(),
                        'confidence': 0.9
                    })
        
        return facts
    
    def create_concept_vector(self, concept_name):
        """
        Create a high-dimensional sparse vector for a concept.
        
        Args:
            concept_name (str): Name of the concept
            
        Returns:
            ndarray: Concept vector
        """
        vector = np.zeros(self.dimension)
        # Set random elements to 1 based on sparsity
        indices = np.random.choice(self.dimension, 
                                  int(self.dimension * self.sparsity), 
                                  replace=False)
        vector[indices] = 1
        self.concept_vectors[concept_name] = vector
        return vector
    
    def get_concept_vector(self, concept_name):
        """
        Get vector for a concept, creating it if it doesn't exist.
        
        Args:
            concept_name (str): Name of the concept
            
        Returns:
            ndarray: Concept vector
        """
        if concept_name not in self.concept_vectors:
            return self.create_concept_vector(concept_name)
        return self.concept_vectors[concept_name]
    
    def bind(self, vector1, vector2):
        """
        Bind two vectors together using element-wise multiplication.
        
        Args:
            vector1 (ndarray): First vector
            vector2 (ndarray): Second vector
            
        Returns:
            ndarray: Bound vector
        """
        return np.multiply(vector1, vector2)
    
    def encode_concept(self, concept):
        """
        Encode a concept into its vector representation.
        
        Args:
            concept (str or ndarray): Concept name or existing vector
            
        Returns:
            ndarray: Vector representation of the concept
        """
        if isinstance(concept, str):
            # If we already have this concept, return existing vector
            if concept in self.concept_vectors:
                return self.concept_vectors[concept]
            
            # Otherwise create a new vector
            vector = self.create_concept_vector(concept)
            self.concept_vectors[concept] = vector
            return vector
        else:
            # If passed a vector directly, just return it
            return concept
            
    def unbind(self, bound_vector, key_vector):
        """
        Unbind a vector using a key (inverse operation).
        
        Args:
            bound_vector (ndarray): Bound vector
            key_vector (ndarray): Key vector
            
        Returns:
            ndarray: Unbound vector
        """
        # Simple implementation using circular convolution
        return np.fft.ifft(np.fft.fft(bound_vector) / np.fft.fft(key_vector)).real
    
    def create_fact(self, subject, predicate, object_value, confidence=1.0, 
                   timestamp=None, source=None):
        """
        Create a fact in the vector-symbolic space.
        
        Args:
            subject (str): Subject entity
            predicate (str): Relation type
            object_value (str): Object entity
            confidence (float): Confidence value (0-1)
            timestamp: Time-related information
            source: Source of this fact
            
        Returns:
            dict: Created fact
        """
        # Normalize inputs
        subject = subject.lower()
        predicate = predicate.lower()
        object_value = object_value.lower()
        
        # Get or create vectors
        subj_vector = self.get_concept_vector(subject)
        pred_vector = self.get_concept_vector(predicate)
        obj_vector = self.get_concept_vector(object_value)
        
        # Bind vectors to create fact representation
        # Simple implementation: subject⊗predicate⊗object
        bound_vector = self.bind(self.bind(subj_vector, pred_vector), obj_vector)
        
        # Use current time if not provided
        if timestamp is None:
            timestamp = time.time()
            
        # Store the fact with metadata
        fact = {
            'subject': subject,
            'predicate': predicate,
            'object': object_value,
            'vector': bound_vector,
            'confidence': confidence,
            'timestamp': timestamp,
            'source': source,
            'id': f"{subject}_{predicate}_{object_value}_{timestamp}"
        }
        
        # Add to facts list and index
        self.facts.append(fact)
        self.fact_index[fact['id']] = fact
        
        self.logger.info(f"[VSA] Created fact: {subject} {predicate} {object_value} (conf={confidence:.2f})")
        return fact
    
    def query_facts(self, subject=None, predicate=None, object_value=None, 
                   threshold=0.7):
        """
        Query facts that match the given pattern.
        
        Args:
            subject (str): Subject to match (optional)
            predicate (str): Predicate to match (optional)
            object_value (str): Object to match (optional)
            threshold (float): Similarity threshold
            
        Returns:
            list: Matching facts
        """
        results = []
        
        # Normalize inputs if provided
        if subject:
            subject = subject.lower()
        if predicate:
            predicate = predicate.lower()
        if object_value:
            object_value = object_value.lower()
        
        for fact in self.facts:
            # Check exact matches for provided fields
            match = True
            
            if subject and fact['subject'] != subject:
                # If subject provided but doesn't match exactly,
                # check similarity
                if subject in self.concept_vectors:
                    similarity = self.similarity(subject, fact['subject'])
                    if similarity < threshold:
                        match = False
                else:
                    match = False
                
            if predicate and fact['predicate'] != predicate:
                # For predicates, we usually want exact matches
                match = False
                
            if object_value and fact['object'] != object_value:
                # For objects, also check similarity
                if object_value in self.concept_vectors:
                    similarity = self.similarity(object_value, fact['object'])
                    if similarity < threshold:
                        match = False
                else:
                    match = False
                
            if match:
                results.append(fact)
                
        # Sort by confidence and recency
        results.sort(key=lambda x: (x['confidence'], x['timestamp']), reverse=True)
                
        return results
    
    def query_facts_by_text(self, text, threshold=0.7):
        """
        Extract entities from text and query related facts.
        
        Args:
            text (str): Query text
            threshold (float): Similarity threshold
            
        Returns:
            list: Matching facts
        """
        # Extract potential entities from text
        entities = self._extract_entities(text)
        
        # Query facts for each entity
        all_results = []
        for entity in entities:
            # Try as subject
            subj_results = self.query_facts(subject=entity, threshold=threshold)
            all_results.extend(subj_results)
            
            # Try as object
            obj_results = self.query_facts(object_value=entity, threshold=threshold)
            all_results.extend(obj_results)
            
        # Deduplicate
        unique_results = []
        seen_ids = set()
        for result in all_results:
            if result['id'] not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result['id'])
                
        # Sort by confidence and recency
        unique_results.sort(key=lambda x: (x['confidence'], x['timestamp']), reverse=True)
        
        return unique_results
    
    def query_by_vector(self, query_vector, threshold=0.7, limit=10):
        """
        Query facts by vector similarity.
        
        Args:
            query_vector (ndarray): Query vector
            threshold (float): Similarity threshold
            limit (int): Maximum number of results
            
        Returns:
            list: Similar facts
        """
        results = []
        
        for fact in self.facts:
            # Calculate similarity between query vector and fact vector
            similarity = 1 - cosine(query_vector, fact['vector'])
            
            if similarity >= threshold:
                fact_copy = fact.copy()
                fact_copy['similarity'] = similarity
                results.append(fact_copy)
                
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Limit results
        return results[:limit]
    
    def _extract_entities(self, text):
        """
        Extract potential entities from text.
        
        Args:
            text (str): Input text
            
        Returns:
            list: Extracted entities
        """
        # Simple word-based extraction
        words = text.lower().split()
        entities = [word for word in words if word.isalpha() and len(word) > 3]
        
        # Remove stopwords (simple approach)
        stopwords = ['what', 'when', 'where', 'how', 'who', 'does', 'did', 'will',
                    'the', 'and', 'but', 'because', 'however', 'thus']
        entities = [e for e in entities if e not in stopwords]
        
        # Also extract capitalized words as potential entities
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend([word.lower() for word in capitalized])
        
        # Deduplicate
        return list(set(entities))
    
    def similarity(self, concept1, concept2):
        """
        Calculate similarity between two concepts.
        
        Args:
            concept1 (str): First concept
            concept2 (str): Second concept
            
        Returns:
            float: Similarity score (0-1)
        """
        # Check cache first
        cache_key = tuple(sorted([concept1, concept2]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Get vectors
        if concept1 not in self.concept_vectors:
            self.get_concept_vector(concept1)
        if concept2 not in self.concept_vectors:
            self.get_concept_vector(concept2)
            
        vec1 = self.concept_vectors[concept1]
        vec2 = self.concept_vectors[concept2]
        
        # Using cosine similarity
        similarity = 1 - cosine(vec1, vec2)
        
        # Cache the result
        self.similarity_cache[cache_key] = similarity
        
        return similarity
    
    def decay_confidence(self, fact_id=None, decay_rate=0.01):
        """
        Decay the confidence of facts over time.
        
        Args:
            fact_id (str): Specific fact ID to decay (optional)
            decay_rate (float): Decay rate
            
        Returns:
            int: Number of facts affected
        """
        count = 0
        
        if fact_id:
            # Decay specific fact
            if fact_id in self.fact_index:
                fact = self.fact_index[fact_id]
                fact['confidence'] = max(0, fact['confidence'] - decay_rate)
                count = 1
        else:
            # Decay all facts
            for fact in self.facts:
                fact['confidence'] = max(0, fact['confidence'] - decay_rate)
                count += 1
                
        return count
    
    def add_concept_relation(self, concept, related_concept, relation_type="related_to", strength=0.5):
        """
        Add a relation between concepts.
        
        Args:
            concept (str): First concept
            related_concept (str): Related concept
            relation_type (str): Type of relation
            strength (float): Relation strength (0-1)
            
        Returns:
            bool: Success flag
        """
        # Ensure both concepts have vectors
        self.get_concept_vector(concept)
        self.get_concept_vector(related_concept)
        
        # Create a fact representing the relation
        self.create_fact(concept, relation_type, related_concept, strength)
        
        return True
    
    def _format_response(self, query_results):
        """
        Format query results as a natural language response.
        
        Args:
            query_results (list): Query results
            
        Returns:
            str: Formatted response
        """
        if not query_results:
            return "I don't have any information about that."
            
        # Group facts by confidence level
        high_conf = []
        med_conf = []
        low_conf = []
        
        for fact in query_results:
            if fact['confidence'] >= 0.8:
                high_conf.append(fact)
            elif fact['confidence'] >= 0.5:
                med_conf.append(fact)
            else:
                low_conf.append(fact)
                
        # Build response
        response_parts = []
        
        if high_conf:
            response_parts.append("I know that:")
            for fact in high_conf[:3]:  # Limit to 3 facts
                response_parts.append(f"- {fact['subject']} {fact['predicate']} {fact['object']}")
            if len(high_conf) > 3:
                response_parts.append(f"- And {len(high_conf) - 3} more similar facts")
                
        if med_conf:
            if response_parts:
                response_parts.append("\nI believe, but am less certain, that:")
            else:
                response_parts.append("I believe, but am not entirely certain, that:")
            for fact in med_conf[:2]:  # Limit to 2 facts
                response_parts.append(f"- {fact['subject']} {fact['predicate']} {fact['object']}")
            if len(med_conf) > 2:
                response_parts.append(f"- And {len(med_conf) - 2} more possibilities")
                
        if low_conf:
            if response_parts:
                response_parts.append("\nI have some vague recollection that:")
            else:
                response_parts.append("I have a vague recollection that:")
            for fact in low_conf[:1]:  # Limit to 1 fact
                response_parts.append(f"- {fact['subject']} might {fact['predicate']} {fact['object']}")
            if len(low_conf) > 1:
                response_parts.append(f"- And {len(low_conf) - 1} other possibilities")
                
        return "\n".join(response_parts)
    
    def get_state(self):
        """
        Get the current state of the vector symbolic engine.
        
        Returns:
            dict: Current state
        """
        return {
            'concept_count': len(self.concept_vectors),
            'fact_count': len(self.facts),
            'dimension': self.dimension,
            'sparsity': self.sparsity
        }
    
    def init_scientific_concepts(self):
        """Initialize descriptive scientific knowledge"""
        # Biological concepts
        self.add_concept_relation("cell", "part_of", "organism")
        self.add_concept_relation("DNA", "contains", "genes")
        
        # Earth science descriptions
        self.add_concept_relation("volcano", "produces", "lava")
        self.add_concept_relation("erosion", "shapes", "landscape")
        
        # Medical observations
        self.add_concept_relation("fever", "symptom_of", "infection")

class VectorSymbolicAdapter:
    """
    Adapter class to provide compatibility between the SNN and the VectorSymbolicEngine.
    Adds missing methods needed for proper vector symbolic integration with neural models.
    """
    
    def __init__(self, base_engine, dimension=1000, sparsity=0.1):
        """
        Initialize the adapter for a Vector Symbolic Engine.
        
        Args:
            base_engine: The base VectorSymbolicEngine to adapt
            dimension: Vector dimension (default 1000)
            sparsity: Vector sparsity (default 0.1)
        """
        self.base_engine = base_engine
        self.dimension = dimension
        self.sparsity = sparsity
        
        # Shadow copy of vectors for direct access
        self._vector_dict = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"[VSA Adapter] Initialized with dimension={self.dimension}, sparsity={self.sparsity}")
    
    @property
    def vector_dict(self):
        """
        Property to access the dictionary of vectors.
        
        Returns:
            Dictionary mapping symbols to vectors
        """
        # Update with any new vectors from the base engine
        if hasattr(self.base_engine, 'concept_vectors'):
            self._vector_dict.update(self.base_engine.concept_vectors)
            
        return self._vector_dict
    
    def get_vector_for_symbol(self, symbol, create_if_missing=True):
        """
        Get the vector for a symbol, creating it if it doesn't exist.
        
        Args:
            symbol: Symbol to get vector for
            create_if_missing: Whether to create a vector if missing
            
        Returns:
            Vector for the symbol
        """
        # First check our local vector dictionary
        if symbol in self._vector_dict:
            return self._vector_dict[symbol]
            
        # Then check the base engine's vectors
        if hasattr(self.base_engine, 'concept_vectors') and symbol in self.base_engine.concept_vectors:
            vector = self.base_engine.concept_vectors[symbol]
            # Store in our local dictionary too
            self._vector_dict[symbol] = vector
            return vector
            
        # Create if missing and requested
        if create_if_missing:
            # Create sparse binary vector
            if hasattr(self.base_engine, 'create_concept_vector'):
                # Use base engine's vector creation
                vector = self.base_engine.create_concept_vector(symbol)
                # Store in our local dictionary
                self._vector_dict[symbol] = vector
                return vector
            else:
                # Create our own vector
                vector = np.zeros(self.dimension)
                # Set random elements to 1 based on sparsity
                num_ones = int(self.dimension * self.sparsity)
                indices = np.random.choice(self.dimension, num_ones, replace=False)
                vector[indices] = 1
                
                # Normalize to unit length
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                
                # Store the vector
                self._vector_dict[symbol] = vector
                
                # Also store in base engine if possible
                if hasattr(self.base_engine, 'concept_vectors'):
                    self.base_engine.concept_vectors[symbol] = vector
                
                return vector
        
        return None
    
    def set_vector_for_symbol(self, symbol, vector):
        """
        Set the vector for a symbol, creating it if it doesn't exist.
        
        Args:
            symbol: Symbol string
            vector: Vector to associate with the symbol
        """
        # Try to set in base engine first
        if hasattr(self.base_engine, 'concept_vectors'):
            self.base_engine.concept_vectors[symbol] = vector
        
        # Also store in our local vector dictionary
        self._vector_dict[symbol] = vector
        
        return vector
    
    def binding_operation(self, vector1, vector2):
        """
        Bind two vectors together using appropriate VSA binding.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Bound vector
        """
        # Try to use base engine's binding method
        if hasattr(self.base_engine, 'bind'):
            try:
                return self.base_engine.bind(vector1, vector2)
            except:
                pass
        
        # Fallback to circular convolution implementation
        return np.fft.ifft(np.fft.fft(vector1) * np.fft.fft(vector2)).real
    
    def unbinding_operation(self, bound_vector, key_vector):
        """
        Unbind a vector using a key (inverse operation).
        
        Args:
            bound_vector: Bound vector
            key_vector: Key vector
            
        Returns:
            Unbound vector
        """
        # Try to use base engine's unbinding method
        if hasattr(self.base_engine, 'unbind'):
            try:
                return self.base_engine.unbind(bound_vector, key_vector)
            except:
                pass
        
        # Fallback implementation using circular convolution
        return np.fft.ifft(np.fft.fft(bound_vector) / np.fft.fft(key_vector)).real
    
    def cosine_similarity(self, vector1, vector2):
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Similarity value between -1 and 1
        """
        # Try to use base engine's similarity method
        if hasattr(self.base_engine, 'similarity'):
            try:
                if isinstance(vector1, str) and isinstance(vector2, str):
                    return self.base_engine.similarity(vector1, vector2)
            except:
                pass
        
        # Convert strings to vectors if needed
        if isinstance(vector1, str):
            vector1 = self.get_vector_for_symbol(vector1)
        if isinstance(vector2, str):
            vector2 = self.get_vector_for_symbol(vector2)
            
        # Calculate dot product
        dot_product = np.dot(vector1, vector2)
        
        # Calculate magnitudes
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        
        # Calculate cosine similarity
        if magnitude1 > 0 and magnitude2 > 0:
            return dot_product / (magnitude1 * magnitude2)
        else:
            return 0.0
    
    def superposition(self, vectors):
        """
        Combine multiple vectors using superposition.
        
        Args:
            vectors: List of vectors to combine
            
        Returns:
            Combined vector
        """
        if not vectors:
            return None
            
        # Convert strings to vectors if needed
        processed_vectors = []
        for v in vectors:
            if isinstance(v, str):
                processed_vectors.append(self.get_vector_for_symbol(v))
            else:
                processed_vectors.append(v)
        
        # Simple element-wise addition
        result = np.zeros_like(processed_vectors[0])
        for v in processed_vectors:
            result += v
            
        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
            
        return result
    
    def cleanup(self, noisy_vector, candidates=None):
        """
        Clean up a noisy vector by finding the closest known vector.
        
        Args:
            noisy_vector: Vector to clean up
            candidates: Optional list of candidate symbols to consider
            
        Returns:
            (symbol, clean_vector, similarity) tuple
        """
        if candidates is None:
            candidates = list(self.concept_vectors.keys())
            
        best_similarity = -1
        best_symbol = None
        best_vector = None
        
        for symbol in candidates:
            vector = self.get_vector_for_symbol(symbol, create_if_missing=False)
            if vector is not None:
                similarity = self.cosine_similarity(noisy_vector, vector)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_symbol = symbol
                    best_vector = vector
        
        return best_symbol, best_vector, best_similarity
    
    def encode_relation(self, subject, predicate, object_val):
        """
        Encode a relation using vector symbolic operations.
        
        Args:
            subject: Subject symbol or vector
            predicate: Predicate symbol or vector
            object_val: Object symbol or vector
            
        Returns:
            Vector representing the relation
        """
        # Convert to vectors if needed
        if isinstance(subject, str):
            subject_vec = self.get_vector_for_symbol(subject)
        else:
            subject_vec = subject
            
        if isinstance(predicate, str):
            predicate_vec = self.get_vector_for_symbol(predicate)
        else:
            predicate_vec = predicate
            
        if isinstance(object_val, str):
            object_vec = self.get_vector_for_symbol(object_val)
        else:
            object_vec = object_val
        
        # Bind predicate with object
        pred_obj = self.binding_operation(predicate_vec, object_vec)
        
        # Bind subject with the result
        relation_vec = self.binding_operation(subject_vec, pred_obj)
        
        return relation_vec
    
    def decode_relation(self, relation_vec, subject=None, predicate=None):
        """
        Decode a relation vector, filling in the missing component.
        
        Args:
            relation_vec: Relation vector
            subject: Subject vector or symbol (optional)
            predicate: Predicate vector or symbol (optional)
            
        Returns:
            The missing component of the relation (subject, predicate, or object)
        """
        # Need either subject or predicate to decode
        if subject is None and predicate is None:
            return None
            
        # Convert to vectors if needed
        if isinstance(subject, str):
            subject_vec = self.get_vector_for_symbol(subject)
        else:
            subject_vec = subject
            
        if isinstance(predicate, str):
            predicate_vec = self.get_vector_for_symbol(predicate)
        else:
            predicate_vec = predicate
        
        # Decode based on what we have
        if subject is not None and predicate is not None:
            # Decode object
            unbind_subj = self.unbinding_operation(relation_vec, subject_vec)
            object_vec = self.unbinding_operation(unbind_subj, predicate_vec)
            
            # Find closest known vector
            object_symbol, _, _ = self.cleanup(object_vec)
            return object_symbol
        
        elif subject is not None:
            # Decode predicate-object binding
            pred_obj = self.unbinding_operation(relation_vec, subject_vec)
            
            # We can't separate predicate from object without more info
            # Return the binding for further processing
            return pred_obj
            
        elif predicate is not None:
            # This case is more complex and requires additional knowledge
            # Here we would need to try different known subjects
            # and check if they yield sensible objects
            return None
        
        return None
        
    def create_vector_analogy(self, a, a_prime, b):
        """
        Create a vector analogy (a:a' :: b:?)
        
        Args:
            a: First vector in pair 1
            a_prime: Second vector in pair 1
            b: First vector in pair 2
            
        Returns:
            b_prime: Second vector in pair 2 (answer to analogy)
        """
        # Convert to vectors if needed
        if isinstance(a, str):
            a_vec = self.get_vector_for_symbol(a)
        else:
            a_vec = a
            
        if isinstance(a_prime, str):
            a_prime_vec = self.get_vector_for_symbol(a_prime)
        else:
            a_prime_vec = a_prime
            
        if isinstance(b, str):
            b_vec = self.get_vector_for_symbol(b)
        else:
            b_vec = b
        
        # Compute a' - a
        if hasattr(self.base_engine, 'unbind'):
            # Use unbinding if available
            transformation = self.base_engine.unbind(a_prime_vec, a_vec)
        else:
            # Simple vector difference
            transformation = a_prime_vec - a_vec
        
        # Apply transformation to b
        b_prime_vec = b_vec + transformation
        
        # Normalize
        norm = np.linalg.norm(b_prime_vec)
        if norm > 0:
            b_prime_vec = b_prime_vec / norm
        
        # Find closest known vector
        b_prime, _, similarity = self.cleanup(b_prime_vec)
        
        return b_prime, b_prime_vec, similarity

    # Proxy other methods to the base engine
    def __getattr__(self, name):
        """Proxy other methods to the base engine"""
        if hasattr(self.base_engine, name):
            return getattr(self.base_engine, name)
        raise AttributeError(f"Neither VectorSymbolicAdapter nor base engine has attribute '{name}'")