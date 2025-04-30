"""
Symbol Grounding module for the cognitive architecture.
Provides mechanisms to connect symbolic representations with subsymbolic
neural patterns, enabling integration between symbolic and neural components.
"""

import numpy as np
import re
from collections import defaultdict

class SymbolGrounding:
    def __init__(self, snn=None, symbolic_engine=None):
        """
        Initialize the symbol grounding system
        
        Args:
            snn: SpikingCore neural network instance
            symbolic_engine: PrologEngine symbolic reasoning instance
        """
        self.snn = snn
        self.symbolic = symbolic_engine
        
        # Symbol-to-pattern mappings
        self.symbol_patterns = {}  # Maps symbols to neural activation patterns
        self.pattern_symbols = {}  # Maps pattern hashes to symbols
        
        # Concept hierarchies for generalization
        self.concept_hierarchy = defaultdict(set)  # Maps concepts to parent concepts
        
        # Neural assembly references
        self.symbol_assemblies = {}  # Maps symbols to neural assemblies
        
        # Confidence and reliability metrics
        self.grounding_confidence = {}  # Confidence in each grounding
        self.grounding_history = []  # History of grounding operations
        
        # Learning parameters
        self.learning_rate = 0.1
        self.min_confidence = 0.3
        self.max_patterns_per_symbol = 5

    def learn_symbol_grounding(self, symbol, neural_pattern, confidence=0.8, learning_rate=None):
        # Use provided learning rate or default
        current_lr = learning_rate if learning_rate is not None else self.learning_rate
        
        if symbol is None or neural_pattern is None:
            return False, 0
            
        # Convert to numpy array and normalize
        pattern = np.array(neural_pattern)
        pattern_norm = np.linalg.norm(pattern)
        if pattern_norm > 0:
            pattern = pattern / pattern_norm  # Important: normalize to unit vector
        
        # Check if we already have patterns for this symbol
        if symbol in self.symbol_patterns and self.symbol_patterns[symbol]:
            # Find most similar existing pattern
            best_pattern_idx = 0
            best_similarity = -1
            
            for i, existing_pattern in enumerate(self.symbol_patterns[symbol]):
                similarity = self._calculate_pattern_similarity(pattern, existing_pattern)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pattern_idx = i
            
            # If similar enough, update existing pattern through weighted average
            if best_similarity > 0.3:
                existing_pattern = self.symbol_patterns[symbol][best_pattern_idx]
                pattern_hash = self._hash_pattern(existing_pattern)
                existing_conf = self.grounding_confidence.get((symbol, pattern_hash), 0.5)
                
                # Calculate weighted average with stability factor
                stability_factor = 0.7  # Prevents rapid changes
                updated_pattern = (existing_pattern * stability_factor + 
                                pattern * (1-stability_factor))
                # Re-normalize
                updated_pattern = updated_pattern / np.linalg.norm(updated_pattern)
                
                # Update confidence carefully - avoid sudden drops
                updated_conf = max(existing_conf * 0.9, 
                                (existing_conf + confidence * current_lr) / (1 + current_lr))
                
                # Replace existing pattern
                self.symbol_patterns[symbol][best_pattern_idx] = updated_pattern
                
                # Update hash and confidence
                new_hash = self._hash_pattern(updated_pattern)
                self.pattern_symbols[new_hash] = symbol
                self.grounding_confidence[(symbol, new_hash)] = updated_conf
                
                # If old hash is different, clean up
                if pattern_hash != new_hash and pattern_hash in self.pattern_symbols:
                    del self.pattern_symbols[pattern_hash]
                    
                return True, updated_conf
        
        # Add new pattern if not updated existing one
        if symbol not in self.symbol_patterns:
            self.symbol_patterns[symbol] = []
        
        # Store pattern and confidence
        self.symbol_patterns[symbol].append(pattern)
        pattern_hash = self._hash_pattern(pattern)
        self.pattern_symbols[pattern_hash] = symbol
        self.grounding_confidence[(symbol, pattern_hash)] = confidence
        
        # Record in history
        self.grounding_history.append({
            "operation": "learn",
            "symbol": symbol,
            "pattern_hash": pattern_hash,
            "confidence": confidence,
            "learning_rate": current_lr
        })
        
        return True, confidence

    '''
    def learn_symbol_grounding(self, symbol, neural_pattern, confidence=0.8, learning_rate=None):
        """Improved learning with averaging and continuous representation"""
        current_lr = learning_rate if learning_rate is not None else self.learning_rate
        
        if symbol is None or neural_pattern is None:
            return False, 0
            
        pattern = np.array(neural_pattern)
        
        # Check if we already have patterns for this symbol
        if symbol in self.symbol_patterns and self.symbol_patterns[symbol]:
            # Instead of adding a new pattern, merge with existing patterns
            best_pattern_idx = 0
            best_similarity = -1
            
            # Find most similar existing pattern
            for i, existing_pattern in enumerate(self.symbol_patterns[symbol]):
                similarity = self._calculate_pattern_similarity(pattern, existing_pattern)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pattern_idx = i
            
            # If similar enough, update existing pattern through averaging
            if best_similarity > 0.3:
                # Weighted average based on confidence
                existing_pattern = self.symbol_patterns[symbol][best_pattern_idx]
                pattern_hash = self._hash_pattern(existing_pattern)
                existing_conf = self.grounding_confidence.get((symbol, pattern_hash), 0.5)
                
                # Calculate weighted average
                updated_pattern = (existing_pattern * existing_conf + pattern * confidence) / (existing_conf + confidence)
                updated_conf = (existing_conf + confidence) / 2  # Average confidence
                
                # Replace existing pattern
                self.symbol_patterns[symbol][best_pattern_idx] = updated_pattern
                
                # Update hash and confidence
                new_hash = self._hash_pattern(updated_pattern)
                self.pattern_symbols[new_hash] = symbol
                self.grounding_confidence[(symbol, new_hash)] = updated_conf
                
                # If old hash is different, clean up
                if pattern_hash != new_hash and pattern_hash in self.pattern_symbols:
                    del self.pattern_symbols[pattern_hash]
                    
                return True, updated_conf
            # Store the pattern for this symbol
            if symbol not in self.symbol_patterns:
                self.symbol_patterns[symbol] = []
                
            # Limit number of patterns per symbol
            if len(self.symbol_patterns[symbol]) >= self.max_patterns_per_symbol:
                # Find and remove the pattern with lowest confidence
                min_conf = float('inf')
                min_pattern_hash = None
                
                for p_hash in [self._hash_pattern(p) for p in self.symbol_patterns[symbol]]:
                    p_conf = self.grounding_confidence.get((symbol, p_hash), 0)
                    if p_conf < min_conf:
                        min_conf = p_conf
                        min_pattern_hash = p_hash
                        
                if min_pattern_hash and min_conf < confidence:
                    # Find and remove the pattern with this hash
                    for i, p in enumerate(self.symbol_patterns[symbol]):
                        if self._hash_pattern(p) == min_pattern_hash:
                            self.symbol_patterns[symbol].pop(i)
                            break
                            
                    # Remove confidence entry
                    if (symbol, min_pattern_hash) in self.grounding_confidence:
                        del self.grounding_confidence[(symbol, min_pattern_hash)]
                else:
                    # Don't add the new pattern if it's not better than existing ones
                    return False, min_conf
            
            # Add the new pattern
            self.symbol_patterns[symbol].append(pattern)
            self.pattern_symbols[pattern_hash] = symbol
            self.grounding_confidence[(symbol, pattern_hash)] = confidence
            
            # Record in history
            self.grounding_history.append({
                "operation": "learn",
                "symbol": symbol,
                "pattern_hash": pattern_hash,
                "confidence": confidence,
                "learning": current_lr
            })
            
            return True, confidence

    def learn_symbol_grounding(self, symbol, neural_pattern, confidence=0.8, learning_rate=None):
        """
        Learn association between a symbol and neural activation pattern
        
        Args:
            symbol: The symbolic representation (string)
            neural_pattern: Neural activation pattern (array of neuron activations)
            confidence: Initial confidence in this grounding (0-1)
            learning_rate: Optional learning rate override
            
        Returns:
            Success status and updated confidence
        """
        # Use the provided learning rate or fall back to the default
        current_lr = learning_rate if learning_rate is not None else self.learning_rate
        
        if symbol is None or neural_pattern is None:
            return False, 0
            
        # Convert pattern to numpy array if it isn't already
        pattern = np.array(neural_pattern)
        
        # Compute pattern hash for storage
        pattern_hash = self._hash_pattern(pattern)
        
        # If this pattern is already associated with a different symbol
        if pattern_hash in self.pattern_symbols and self.pattern_symbols[pattern_hash] != symbol:
            # Get the existing symbol
            existing_symbol = self.pattern_symbols[pattern_hash]
            existing_confidence = self.grounding_confidence.get((existing_symbol, pattern_hash), 0)
            
            # If existing association is stronger, don't override
            if existing_confidence > confidence:
                return False, existing_confidence
            
        # Store the pattern for this symbol
        if symbol not in self.symbol_patterns:
            self.symbol_patterns[symbol] = []
            
        # Limit number of patterns per symbol
        if len(self.symbol_patterns[symbol]) >= self.max_patterns_per_symbol:
            # Find and remove the pattern with lowest confidence
            min_conf = float('inf')
            min_pattern_hash = None
            
            for p_hash in [self._hash_pattern(p) for p in self.symbol_patterns[symbol]]:
                p_conf = self.grounding_confidence.get((symbol, p_hash), 0)
                if p_conf < min_conf:
                    min_conf = p_conf
                    min_pattern_hash = p_hash
                    
            if min_pattern_hash and min_conf < confidence:
                # Find and remove the pattern with this hash
                for i, p in enumerate(self.symbol_patterns[symbol]):
                    if self._hash_pattern(p) == min_pattern_hash:
                        self.symbol_patterns[symbol].pop(i)
                        break
                        
                # Remove confidence entry
                if (symbol, min_pattern_hash) in self.grounding_confidence:
                    del self.grounding_confidence[(symbol, min_pattern_hash)]
            else:
                # Don't add the new pattern if it's not better than existing ones
                return False, min_conf
        
        # Add the new pattern
        self.symbol_patterns[symbol].append(pattern)
        self.pattern_symbols[pattern_hash] = symbol
        self.grounding_confidence[(symbol, pattern_hash)] = confidence
        
        # Record in history
        self.grounding_history.append({
            "operation": "learn",
            "symbol": symbol,
            "pattern_hash": pattern_hash,
            "confidence": confidence,
            "learning": current_lr
        })
        
        return True, confidence
    '''
    
    def associate_symbol_with_assembly(self, symbol, assembly_name):
        """
        Associate a symbol with a neural assembly
        
        Args:
            symbol: The symbol to associate
            assembly_name: Name of the neural assembly
            
        Returns:
            Boolean indicating success
        """
        if not self.snn or not hasattr(self.snn, 'get_assembly'):
            return False
            
        # Get the assembly
        assembly = self.snn.get_assembly(assembly_name) if hasattr(self.snn, 'get_assembly') else None
        if not assembly:
            return False
            
        # Store the association
        self.symbol_assemblies[symbol] = assembly_name
        
        # If the SNN has a concept mapping feature, use it
        if hasattr(self.snn, 'map_concept_to_assembly'):
            self.snn.map_concept_to_assembly(symbol, assembly_name)
            
        return True
    
    def recognize_symbol_from_pattern(self, neural_pattern, threshold=0.7):
        """
        Recognize a symbol from a neural activation pattern
        
        Args:
            neural_pattern: Neural activation pattern
            threshold: Similarity threshold for recognition
            
        Returns:
            List of (symbol, confidence) tuples for matched symbols
        """
        pattern = np.array(neural_pattern)
        matches = []
        
        # Check against all known symbols
        for symbol, patterns in self.symbol_patterns.items():
            max_similarity = 0
            best_pattern_hash = None
            
            for stored_pattern in patterns:
                similarity = self._calculate_pattern_similarity(pattern, stored_pattern)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_pattern_hash = self._hash_pattern(stored_pattern)
                    
            # If similarity exceeds threshold, add to matches
            if max_similarity >= threshold:
                base_confidence = self.grounding_confidence.get((symbol, best_pattern_hash), 0.5)
                # Adjust confidence based on similarity
                confidence = base_confidence * max_similarity
                matches.append((symbol, confidence))
                
        # Sort by confidence
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def activate_symbol(self, symbol, activation_strength=0.8):
        """
        Activate the neural pattern associated with a symbol
        
        Args:
            symbol: The symbol to activate
            activation_strength: Strength of activation (0-1)
            
        Returns:
            Activation success and the activated neurons
        """
        if not self.snn:
            return False, []
            
        # First try to activate via assembly if available
        if symbol in self.symbol_assemblies:
            assembly_name = self.symbol_assemblies[symbol]
            if hasattr(self.snn, 'activate_assembly'):
                success = self.snn.activate_assembly(assembly_name, activation_strength)
                if success:
                    # Get activated neurons
                    if hasattr(self.snn, 'get_assembly_neurons'):
                        neurons = self.snn.get_assembly_neurons(assembly_name)
                        return True, neurons
                    return True, []
        
        # Fall back to pattern-based activation
        if symbol in self.symbol_patterns and self.symbol_patterns[symbol]:
            # Use the first pattern (highest confidence one should be first)
            pattern = self.symbol_patterns[symbol][0]
            
            # Scale pattern by activation strength
            activation = pattern * activation_strength
            
            # Apply to SNN
            if hasattr(self.snn, 'apply_activation_pattern'):
                self.snn.apply_activation_pattern(activation)
                
                # Return the indices of activated neurons
                activated_neurons = np.where(activation > 0.5)[0].tolist()
                return True, activated_neurons
                
        return False, []
    
    def ground_symbolic_fact(self, subject, predicate, object):
        """
        Ground a symbolic fact in neural representations
        
        Args:
            subject: Subject of the fact
            predicate: Predicate/relation
            object: Object of the fact
            
        Returns:
            Success status and confidence
        """
        if not self.snn:
            return False, 0
            
        # First activate the subject
        subj_success, subj_neurons = self.activate_symbol(subject, 0.7)
        
        # Then activate the predicate
        pred_success, pred_neurons = self.activate_symbol(predicate, 0.7)
        
        # Finally activate the object
        obj_success, obj_neurons = self.activate_symbol(object, 0.7)
        
        # Record this triple activation pattern if all successful
        if subj_success and pred_success and obj_success:
            # Create a combined pattern
            if hasattr(self.snn, 'get_current_activation'):
                # Get the current network activation after activating all three
                combined_pattern = self.snn.get_current_activation()
                
                # Learn this as a grounding for the entire fact
                fact_symbol = f"{predicate}({subject}, {object})"
                success, confidence = self.learn_symbol_grounding(fact_symbol, combined_pattern, 0.7)
                
                # If SNN supports creating assemblies, create one for this fact
                if hasattr(self.snn, 'create_assembly') and success:
                    assembly_name = f"{subject}_{predicate}_{object}"
                    # Create assembly from activated neurons
                    all_neurons = list(set(subj_neurons + pred_neurons + obj_neurons))
                    if hasattr(self.snn, 'create_assembly_from_neurons'):
                        self.snn.create_assembly_from_neurons(assembly_name, all_neurons)
                        # Associate with the fact symbol
                        self.associate_symbol_with_assembly(fact_symbol, assembly_name)
                        
                return success, confidence
                
        # If we couldn't fully ground it, return failure
        return False, 0
    
    def neural_to_symbolic_translation(self, neural_pattern, context=None):
        """
        Translate neural pattern to symbolic representation
        
        Args:
            neural_pattern: Neural activation pattern
            context: Optional context to disambiguate
            
        Returns:
            List of (symbol, confidence) pairs
        """
        # First recognize individual symbols
        symbol_matches = self.recognize_symbol_from_pattern(neural_pattern)
        
        # If no matches, return empty list
        if not symbol_matches:
            return []
            
        # Check if any match is a fact (predicate with arguments)
        fact_matches = []
        relation_matches = []
        entity_matches = []
        
        for symbol, confidence in symbol_matches:
            # Check if it's a fact pattern (predicate with arguments)
            fact_pattern = r'(\w+)\((\w+),\s*(\w+)\)'
            match = re.match(fact_pattern, symbol)
            if match:
                fact_matches.append((symbol, confidence, match.groups()))
                continue
                
            # Check if it's a relation
            if symbol in ["trusts", "likes", "knows", "fears", "hates", "avoids"]:
                relation_matches.append((symbol, confidence))
                continue
                
            # Assume it's an entity
            entity_matches.append((symbol, confidence))
            
        # If we have good fact matches, return those
        if fact_matches:
            return [(symbol, confidence) for symbol, confidence, _ in fact_matches]
            
        # If not, try to construct facts from relations and entities
        constructed_facts = []
        
        if relation_matches and len(entity_matches) >= 2:
            # Take the top relation and top two entities
            top_relation, rel_conf = relation_matches[0]
            top_entities = entity_matches[:2]
            
            # Construct facts
            subj, subj_conf = top_entities[0]
            obj, obj_conf = top_entities[1]
            
            # Combined confidence
            fact_conf = rel_conf * 0.6 + subj_conf * 0.2 + obj_conf * 0.2
            
            # Create symbolic fact
            fact = f"{top_relation}({subj}, {obj})"
            constructed_facts.append((fact, fact_conf))
            
        # Return constructed facts if any, otherwise return top symbols
        if constructed_facts:
            return constructed_facts
        else:
            return symbol_matches[:3]  # Return top 3 symbol matches
    
    def symbolic_to_neural_translation(self, symbol):
        """
        Translate symbolic representation to neural pattern
        
        Args:
            symbol: Symbol to translate
            
        Returns:
            Neural activation pattern and confidence
        """
        # Check if we have direct grounding for this symbol
        if symbol in self.symbol_patterns and self.symbol_patterns[symbol]:
            # Get the first pattern (should be highest confidence)
            pattern = self.symbol_patterns[symbol][0]
            pattern_hash = self._hash_pattern(pattern)
            confidence = self.grounding_confidence.get((symbol, pattern_hash), 0.5)
            return pattern, confidence
            
        # If it's a fact, try to construct from components
        fact_pattern = r'(\w+)\((\w+),\s*(\w+)\)'
        match = re.match(fact_pattern, symbol)
        if match:
            predicate, subject, object = match.groups()
            
            # Get patterns for each component
            component_patterns = []
            total_confidence = 1.0
            
            for component in [predicate, subject, object]:
                if component in self.symbol_patterns and self.symbol_patterns[component]:
                    # Get the first pattern for this component
                    c_pattern = self.symbol_patterns[component][0]
                    c_hash = self._hash_pattern(c_pattern)
                    c_conf = self.grounding_confidence.get((component, c_hash), 0.5)
                    
                    component_patterns.append((c_pattern, c_conf))
                    total_confidence *= c_conf
                else:
                    # Missing component grounding
                    return None, 0
                    
            # If we have all components, combine them
            if len(component_patterns) == 3:
                # Simple strategy: element-wise max of all patterns
                combined = np.maximum.reduce([p for p, _ in component_patterns])
                
                # Adjust confidence based on completeness
                confidence = total_confidence ** (1/3)  # Geometric mean
                
                return combined, confidence
                
        # No grounding found
        return None, 0
    
    def update_grounding_confidence(self, symbol, pattern_hash, success_factor):
        """
        Update confidence in a symbol-pattern grounding based on success/failure
        
        Args:
            symbol: The symbol
            pattern_hash: Hash of the pattern
            success_factor: Positive for success, negative for failure (-1 to 1)
            
        Returns:
            New confidence value
        """
        if (symbol, pattern_hash) not in self.grounding_confidence:
            return 0
            
        current_confidence = self.grounding_confidence[(symbol, pattern_hash)]
        
        # Update confidence
        if success_factor > 0:
            # Success: increase confidence
            new_confidence = current_confidence + self.learning_rate * (1 - current_confidence) * success_factor
        else:
            # Failure: decrease confidence
            new_confidence = current_confidence + self.learning_rate * current_confidence * success_factor
            
        # Ensure confidence is within bounds
        new_confidence = max(0, min(1, new_confidence))
        
        # Update stored confidence
        self.grounding_confidence[(symbol, pattern_hash)] = new_confidence
        
        # If confidence drops below threshold, consider removing the grounding
        if new_confidence < self.min_confidence:
            self._remove_low_confidence_grounding(symbol, pattern_hash)
            
        return new_confidence
    
    def _remove_low_confidence_grounding(self, symbol, pattern_hash):
        """Remove a grounding that has fallen below confidence threshold"""
        if symbol in self.symbol_patterns:
            # Find and remove the pattern with this hash
            for i, pattern in enumerate(self.symbol_patterns[symbol]):
                if self._hash_pattern(pattern) == pattern_hash:
                    self.symbol_patterns[symbol].pop(i)
                    break
                    
        # Clean up other mappings
        if pattern_hash in self.pattern_symbols:
            del self.pattern_symbols[pattern_hash]
            
        if (symbol, pattern_hash) in self.grounding_confidence:
            del self.grounding_confidence[(symbol, pattern_hash)]
            
        # Record in history
        self.grounding_history.append({
            "operation": "remove",
            "symbol": symbol,
            "pattern_hash": pattern_hash,
            "reason": "low_confidence"
        })
    
    def add_concept_hierarchy(self, concept, parent_concept, confidence=0.9):
        """
        Add hierarchical relationship between concepts for generalization
        
        Args:
            concept: The child concept
            parent_concept: The parent concept
            confidence: Confidence in this relationship
            
        Returns:
            Success status
        """
        # Add the parent-child relationship
        self.concept_hierarchy[concept].add((parent_concept, confidence))
        
        # If we have symbolic reasoning, add this as a rule
        if self.symbolic:
            try:
                # Format: isa(child, parent) with confidence
                rule = f"confident_fact(isa, {concept}, {parent_concept}, {confidence})"
                self.symbolic.prolog.assertz(rule)
                
                # Add inheritance rule if not already present
                inheritance_rule = f"confident_fact(P, {parent_concept}, O, C1) :- confident_fact(isa, {concept}, {parent_concept}, C2), confident_fact(P, {concept}, O, C3), C1 is C2 * C3"
                self.symbolic.prolog.assertz(inheritance_rule)
                
                return True
            except Exception as e:
                print(f"[SymbolGrounding] Failed to add concept hierarchy: {e}")
                return False
                
        return True
    
    def get_parent_concepts(self, concept, min_confidence=0.5):
        """
        Get parent concepts for a given concept
        
        Args:
            concept: The concept to find parents for
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of (parent_concept, confidence) pairs
        """
        parents = []
        
        # Direct parents from hierarchy
        for parent, confidence in self.concept_hierarchy.get(concept, set()):
            if confidence >= min_confidence:
                parents.append((parent, confidence))
                
        # If symbolic reasoning available, query for additional parents
        if self.symbolic:
            try:
                query = f"isa,{concept},Parent"
                results = self.symbolic.reason_with_uncertainty(query)
                
                if results.get("certain"):
                    for result in results["certain"]:
                        parent = result.get("object")
                        confidence = result.get("confidence", 0.8)
                        if confidence >= min_confidence:
                            parents.append((parent, confidence))
                            
            except Exception as e:
                print(f"[SymbolGrounding] Failed to query parent concepts: {e}")
                
        return parents
    
    def _hash_pattern(self, pattern):
        """
        Create a hash for a neural pattern for efficient storage/lookup
        
        Args:
            pattern: Neural activation pattern
            
        Returns:
            Hash value for the pattern
        """
        # Convert to binary pattern for more stable hashing
        binary_pattern = (np.array(pattern) > 0.5).astype(int)
        
        # Create a string representation for hashing
        pattern_str = ''.join(map(str, binary_pattern))
        
        # Return hash
        return hash(pattern_str)
    
    '''
    def _calculate_pattern_similarity(self, pattern1, pattern2):
        """Improved similarity calculation using continuous values"""
        p1 = np.array(pattern1)
        p2 = np.array(pattern2)
        
        # Make sure dimensions match
        if p1.shape != p2.shape:
            min_len = min(len(p1), len(p2))
            p1 = p1[:min_len]
            p2 = p2[:min_len]
        
        # Use continuous cosine similarity
        dot_product = np.dot(p1, p2)
        norm1 = np.linalg.norm(p1)
        norm2 = np.linalg.norm(p2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0
            
        similarity = dot_product / (norm1 * norm2)
        return similarity
    '''
    
    def _calculate_pattern_similarity(self, pattern1, pattern2):
        """Improved similarity calculation using continuous cosine similarity"""
        p1 = np.array(pattern1)
        p2 = np.array(pattern2)
        
        # Make sure dimensions match
        if p1.shape != p2.shape:
            min_len = min(len(p1), len(p2))
            p1 = p1[:min_len]
            p2 = p2[:min_len]
        
        # Use continuous cosine similarity
        dot_product = np.dot(p1, p2)
        norm1 = np.linalg.norm(p1)
        norm2 = np.linalg.norm(p2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0
            
        similarity = dot_product / (norm1 * norm2)
        return max(0, similarity)  # Ensure non-negative