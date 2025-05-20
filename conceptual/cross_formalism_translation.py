"""
Cross-Formalism Translation Layer for Cogmenta Core.
Provides translation between different knowledge representation formalisms.
Enables seamless integration between symbolic, vector, and neural representations.
"""

import numpy as np
import re
import uuid
import time
from collections import defaultdict

class CrossFormalismTranslation:
    """
    Translates between different knowledge representation formalisms.
    Provides bidirectional mapping between symbolic, vector, and neural representations.
    Acts as a bridge between different components of the Cogmenta Core cognitive architecture.
    """
    
    def __init__(self, 
                concept_system=None, 
                vector_symbolic=None, 
                prolog_engine=None, 
                snn=None,
                meaning_map=None):
        """
        Initialize the cross-formalism translation layer.
        
        Args:
            concept_system: ConceptEmbeddingSystem instance (optional)
            vector_symbolic: VectorSymbolicEngine instance (optional)
            prolog_engine: PrologEngine instance (optional)
            snn: EnhancedSpikingCore instance (optional)
            meaning_map: StructuredMeaningMap instance (optional)
        """
        self.concept_system = concept_system
        self.vector_symbolic = vector_symbolic
        self.prolog_engine = prolog_engine
        self.snn = snn
        self.meaning_map = meaning_map
        
        # Translation caches for efficiency
        self.symbolic_to_vector_cache = {}
        self.vector_to_symbolic_cache = {}
        self.symbolic_to_neural_cache = {}
        self.neural_to_symbolic_cache = {}
        
        # Confidence thresholds
        self.translation_threshold = 0.6
        
        # Track translation operations for analysis
        self.translation_history = []
        
        # Mapping between predicate names in different formalisms
        self.predicate_mappings = {
            'symbolic_to_vector': {
                'trusts': 'trusts',
                'likes': 'likes',
                'fears': 'fears',
                'knows': 'knows',
                'is_a': 'is_a'
            },
            'vector_to_symbolic': {
                'trusts': 'trusts',
                'likes': 'likes',
                'fears': 'fears',
                'knows': 'knows',
                'is_a': 'is_a'
            }
        }
    
    def translate_symbolic_to_vector(self, symbolic_fact):
        """
        Translate a symbolic fact to vector representation.
        
        Args:
            symbolic_fact: Dict with subject, predicate, object
            
        Returns:
            Dict with vector representation and confidence
        """
        # Check cache
        cache_key = f"{symbolic_fact['subject']}_{symbolic_fact['predicate']}_{symbolic_fact['object']}"
        if cache_key in self.symbolic_to_vector_cache:
            return self.symbolic_to_vector_cache[cache_key]
        
        # Extract components
        subject = symbolic_fact['subject'].lower()
        predicate = symbolic_fact['predicate'].lower()
        object_val = symbolic_fact['object'].lower()
        confidence = symbolic_fact.get('confidence', 1.0)
        
        # Map predicate if needed
        if predicate in self.predicate_mappings['symbolic_to_vector']:
            vector_predicate = self.predicate_mappings['symbolic_to_vector'][predicate]
        else:
            vector_predicate = predicate
        
        # Translate using vector symbolic engine if available
        if self.vector_symbolic:
            # Create or retrieve vectors
            vector_fact = self.vector_symbolic.create_fact(
                subject, 
                vector_predicate, 
                object_val, 
                confidence
            )
            
            result = {
                'type': 'vector_fact',
                'subject': subject,
                'predicate': vector_predicate,
                'object': object_val,
                'vector': vector_fact['vector'].tolist() if isinstance(vector_fact['vector'], np.ndarray) else vector_fact['vector'],
                'confidence': confidence,
                'translated_from': 'symbolic'
            }
        else:
            # Fall back to concept system if available
            if self.concept_system:
                # Get concept vectors
                subj_concept, subj_conf = self.concept_system.get_concept_from_text(subject)
                obj_concept, obj_conf = self.concept_system.get_concept_from_text(object_val)
                
                if subj_concept and obj_concept:
                    # Combine confidences
                    combined_conf = confidence * subj_conf * obj_conf
                    
                    # Create relation in concept system
                    self.concept_system.add_concept_relation(
                        subj_concept,
                        obj_concept,
                        vector_predicate,
                        combined_conf
                    )
                    
                    # Get vectors
                    subj_vector = self.concept_system.get_concept_vector(subj_concept)
                    obj_vector = self.concept_system.get_concept_vector(obj_concept)
                    
                    # Simple representation: concatenate vectors
                    combined_vector = np.concatenate([subj_vector, obj_vector])
                    
                    result = {
                        'type': 'vector_fact',
                        'subject': subject,
                        'predicate': vector_predicate,
                        'object': object_val,
                        'vector': combined_vector.tolist(),
                        'confidence': combined_conf,
                        'translated_from': 'symbolic'
                    }
                else:
                    # Cannot translate
                    result = {
                        'type': 'translation_failure',
                        'error': 'Subject or object concept not found',
                        'confidence': 0.0
                    }
            else:
                # Cannot translate
                result = {
                    'type': 'translation_failure',
                    'error': 'No vector translation mechanism available',
                    'confidence': 0.0
                }
        
        # Cache the result
        self.symbolic_to_vector_cache[cache_key] = result
        
        # Record translation operation
        self.translation_history.append({
            'type': 'symbolic_to_vector',
            'input': symbolic_fact,
            'output': result,
            'timestamp': time.time()
        })
        
        return result
    
    def translate_vector_to_symbolic(self, vector_fact):
        """
        Translate a vector fact to symbolic representation.
        
        Args:
            vector_fact: Dict with vector representation
            
        Returns:
            Dict with symbolic representation and confidence
        """
        # Extract components
        subject = vector_fact.get('subject', '').lower()
        predicate = vector_fact.get('predicate', '').lower()
        object_val = vector_fact.get('object', '').lower()
        confidence = vector_fact.get('confidence', 1.0)
        
        # Check cache if we have all components
        if subject and predicate and object_val:
            cache_key = f"{subject}_{predicate}_{object_val}"
            if cache_key in self.vector_to_symbolic_cache:
                return self.vector_to_symbolic_cache[cache_key]
        
        # Map predicate if needed
        if predicate in self.predicate_mappings['vector_to_symbolic']:
            symbolic_predicate = self.predicate_mappings['vector_to_symbolic'][predicate]
        else:
            symbolic_predicate = predicate
        
        # If we don't have explicit components but have a vector, try to decode
        if 'vector' in vector_fact and (not subject or not predicate or not object_val):
            vector = vector_fact['vector']
            if isinstance(vector, list):
                vector = np.array(vector)
            
            # Try to decode using vector symbolic engine
            if self.vector_symbolic:
                # Query by vector
                similar_facts = self.vector_symbolic.query_by_vector(vector)
                
                if similar_facts:
                    # Use the most similar fact
                    best_match = similar_facts[0]
                    subject = best_match['subject']
                    symbolic_predicate = best_match['predicate']
                    object_val = best_match['object']
                    confidence = best_match.get('similarity', 0.5) * confidence
            
            # If still missing components, try concept system
            if (not subject or not predicate or not object_val) and self.concept_system:
                # This is more speculative - try to decode the vector components
                if len(vector) >= self.concept_system.embedding_dim * 2:
                    # Assume concatenated vectors
                    subj_vector = vector[:self.concept_system.embedding_dim]
                    obj_vector = vector[self.concept_system.embedding_dim:self.concept_system.embedding_dim*2]
                    
                    # Find closest concepts
                    nearest_subjects = []
                    nearest_objects = []
                    
                    for concept, concept_vector in self.concept_system.concepts.items():
                        # Compare with subject vector
                        subj_sim = 1 - np.linalg.norm(subj_vector - concept_vector)
                        if subj_sim > 0.7:
                            nearest_subjects.append((concept, subj_sim))
                            
                        # Compare with object vector
                        obj_sim = 1 - np.linalg.norm(obj_vector - concept_vector)
                        if obj_sim > 0.7:
                            nearest_objects.append((concept, obj_sim))
                    
                    # Use best matches if found
                    if nearest_subjects:
                        nearest_subjects.sort(key=lambda x: x[1], reverse=True)
                        subject = nearest_subjects[0][0]
                        subj_confidence = nearest_subjects[0][1]
                    else:
                        subj_confidence = 0.5
                        
                    if nearest_objects:
                        nearest_objects.sort(key=lambda x: x[1], reverse=True)
                        object_val = nearest_objects[0][0]
                        obj_confidence = nearest_objects[0][1]
                    else:
                        obj_confidence = 0.5
                        
                    # Adjust confidence
                    confidence = confidence * subj_confidence * obj_confidence
        
        # If we have the components, create symbolic fact
        if subject and symbolic_predicate and object_val:
            # Create symbolic fact
            symbolic_fact = {
                'type': 'symbolic_fact',
                'subject': subject,
                'predicate': symbolic_predicate,
                'object': object_val,
                'confidence': confidence,
                'translated_from': 'vector'
            }
            
            # Add to Prolog KB if available
            if self.prolog_engine and confidence >= self.translation_threshold:
                try:
                    self.prolog_engine.prolog.assertz(
                        f"confident_fact({symbolic_predicate}, {subject}, {object_val}, {confidence})"
                    )
                except Exception as e:
                    print(f"Error asserting fact to Prolog: {e}")
            
            # Cache the result
            if subject and predicate and object_val:
                cache_key = f"{subject}_{predicate}_{object_val}"
                self.vector_to_symbolic_cache[cache_key] = symbolic_fact
            
            # Record translation operation
            self.translation_history.append({
                'type': 'vector_to_symbolic',
                'input': vector_fact,
                'output': symbolic_fact,
                'timestamp': time.time()
            })
            
            return symbolic_fact
        else:
            # Cannot translate
            result = {
                'type': 'translation_failure',
                'error': 'Could not decode vector to symbolic components',
                'confidence': 0.0
            }
            
            # Record translation operation
            self.translation_history.append({
                'type': 'vector_to_symbolic',
                'input': vector_fact,
                'output': result,
                'timestamp': time.time()
            })
            
            return result
    
    def translate_symbolic_to_neural(self, symbolic_fact):
        """
        Translate a symbolic fact to neural representation.
        
        Args:
            symbolic_fact: Dict with subject, predicate, object
            
        Returns:
            Dict with neural representation and confidence
        """
        # Check cache
        cache_key = f"{symbolic_fact['subject']}_{symbolic_fact['predicate']}_{symbolic_fact['object']}"
        if cache_key in self.symbolic_to_neural_cache:
            return self.symbolic_to_neural_cache[cache_key]
        
        # Extract components
        subject = symbolic_fact['subject'].lower()
        predicate = symbolic_fact['predicate'].lower()
        object_val = symbolic_fact['object'].lower()
        confidence = symbolic_fact.get('confidence', 1.0)
        
        # Use SNN if available
        if self.snn:
            # Convert to neural pattern
            result = None
            
            # Check if SNN has symbol grounding system
            if hasattr(self.snn, 'ground_symbolic_fact'):
                # Use symbol grounding
                success, activation = self.snn.ground_symbolic_fact(subject, predicate, object_val)
                
                if success:
                    # Get current activation pattern
                    activation_pattern = self.snn.get_current_activation()
                    
                    result = {
                        'type': 'neural_representation',
                        'subject': subject,
                        'predicate': predicate,
                        'object': object_val,
                        'activation_pattern': activation_pattern.tolist() if isinstance(activation_pattern, np.ndarray) else activation_pattern,
                        'confidence': confidence * activation,
                        'translated_from': 'symbolic'
                    }
            
            # If no symbol grounding or it failed, try more basic approach
            if not result:
                # Convert to text and activate network
                fact_text = f"{subject} {predicate} {object_val}"
                
                # Activate SNN with the text
                if hasattr(self.snn, 'process_input'):
                    snn_result = self.snn.process_input(fact_text)
                    
                    # Get activation pattern
                    if 'membrane_potentials' in snn_result:
                        activation_pattern = snn_result['membrane_potentials']
                        
                        result = {
                            'type': 'neural_representation',
                            'subject': subject,
                            'predicate': predicate,
                            'object': object_val,
                            'activation_pattern': activation_pattern.tolist() if isinstance(activation_pattern, np.ndarray) else activation_pattern,
                            'phi': snn_result.get('phi', 0.0),
                            'confidence': confidence * 0.7,  # Lower confidence for this approach
                            'translated_from': 'symbolic'
                        }
                    else:
                        result = {
                            'type': 'translation_failure',
                            'error': 'SNN activation did not produce expected pattern',
                            'confidence': 0.0
                        }
                else:
                    result = {
                        'type': 'translation_failure',
                        'error': 'SNN lacks required functionality',
                        'confidence': 0.0
                    }
        else:
            # Cannot translate
            result = {
                'type': 'translation_failure',
                'error': 'No neural translation mechanism available',
                'confidence': 0.0
            }
        
        # Cache the result
        self.symbolic_to_neural_cache[cache_key] = result
        
        # Record translation operation
        self.translation_history.append({
            'type': 'symbolic_to_neural',
            'input': symbolic_fact,
            'output': result,
            'timestamp': time.time()
        })
        
        return result
    
    def translate_neural_to_symbolic(self, neural_pattern):
        """
        Translate a neural activation pattern to symbolic representation.
        
        Args:
            neural_pattern: Dict with activation pattern
            
        Returns:
            Dict with symbolic representation and confidence
        """
        # Extract components
        activation_pattern = neural_pattern.get('activation_pattern')
        if not activation_pattern:
            return {
                'type': 'translation_failure',
                'error': 'No activation pattern provided',
                'confidence': 0.0
            }
        
        # Convert to numpy array if needed
        if isinstance(activation_pattern, list):
            activation_pattern = np.array(activation_pattern)
        
        # Hash the pattern for cache lookup
        pattern_hash = hash(str(activation_pattern.tolist()))
        
        # Check cache
        if pattern_hash in self.neural_to_symbolic_cache:
            return self.neural_to_symbolic_cache[pattern_hash]
        
        # Use SNN for neural-to-symbolic translation if available
        if self.snn:
            result = None
            
            # Check if SNN has neural-to-symbolic translation
            if hasattr(self.snn, 'neural_to_symbolic_translation'):
                # Apply the pattern to the network
                if hasattr(self.snn, 'apply_activation_pattern'):
                    self.snn.apply_activation_pattern(activation_pattern)
                
                # Translate the current state
                symbolic_results = self.snn.neural_to_symbolic_translation(activation_pattern)
                
                if symbolic_results:
                    # Use the highest confidence match
                    best_match = max(symbolic_results, key=lambda x: x[1])
                    symbol, confidence = best_match
                    
                    # Extract subject, predicate, object if possible
                    match = re.match(r'(\w+)\((\w+),\s*(\w+)\)', symbol)
                    
                    if match:
                        predicate, subject, object_val = match.groups()
                        
                        result = {
                            'type': 'symbolic_fact',
                            'subject': subject,
                            'predicate': predicate,
                            'object': object_val,
                            'confidence': confidence,
                            'translated_from': 'neural'
                        }
                    else:
                        # Could not parse the symbol
                        result = {
                            'type': 'partial_translation',
                            'symbol': symbol,
                            'confidence': confidence,
                            'translated_from': 'neural'
                        }
            
            # If no direct translation available or it failed, try abductive reasoning
            if not result and hasattr(self.snn, 'abductive_reasoning'):
                # Apply the pattern to the network
                if hasattr(self.snn, 'apply_activation_pattern'):
                    self.snn.apply_activation_pattern(activation_pattern)
                
                # Use abductive reasoning to generate hypotheses
                hypotheses = self.snn.abductive_reasoning("pattern")  # Generic placeholder
                
                if hypotheses:
                    # Use the first hypothesis
                    hypothesis = hypotheses[0]
                    
                    # Extract subject, predicate, object if possible
                    match = re.match(r'(\w+)\((\w+)(?:,\s*(\w+))?\)', hypothesis)
                    
                    if match:
                        groups = match.groups()
                        
                        if len(groups) == 3 and groups[2]:
                            # Binary predicate
                            predicate, subject, object_val = groups
                            
                            result = {
                                'type': 'symbolic_fact',
                                'subject': subject,
                                'predicate': predicate,
                                'object': object_val,
                                'confidence': 0.6,  # Lower confidence for abductive reasoning
                                'translated_from': 'neural'
                            }
                        elif len(groups) >= 2:
                            # Unary predicate
                            predicate, subject = groups[:2]
                            
                            result = {
                                'type': 'symbolic_fact',
                                'subject': subject,
                                'predicate': predicate,
                                'object': 'true',
                                'confidence': 0.6,  # Lower confidence for abductive reasoning
                                'translated_from': 'neural'
                            }
            
            if not result:
                # Try to detect active concepts using SNN's concept detection mechanisms
                if hasattr(self.snn, '_detect_active_concepts_from_regions'):
                    active_concepts = self.snn._detect_active_concepts_from_regions()
                    
                    if active_concepts:
                        # Sort by activation level
                        sorted_concepts = sorted(active_concepts.items(), key=lambda x: x[1], reverse=True)
                        
                        # Use top two concepts if available
                        if len(sorted_concepts) >= 2:
                            # Create a relation between top concepts
                            result = {
                                'type': 'symbolic_fact',
                                'subject': sorted_concepts[0][0],
                                'predicate': 'related_to',
                                'object': sorted_concepts[1][0],
                                'confidence': sorted_concepts[0][1] * sorted_concepts[1][1],
                                'translated_from': 'neural',
                                'note': 'Generated from active concepts'
                            }
                        elif len(sorted_concepts) == 1:
                            # Create a unary predicate for the single concept
                            result = {
                                'type': 'symbolic_fact',
                                'subject': sorted_concepts[0][0],
                                'predicate': 'active',
                                'object': 'true',
                                'confidence': sorted_concepts[0][1],
                                'translated_from': 'neural',
                                'note': 'Generated from single active concept'
                            }
                
                if not result:
                    # Translation failed
                    result = {
                        'type': 'translation_failure',
                        'error': 'Could not translate neural pattern to symbolic representation',
                        'confidence': 0.0
                    }
        else:
            # Cannot translate
            result = {
                'type': 'translation_failure',
                'error': 'No neural translation mechanism available',
                'confidence': 0.0
            }
        
        # Cache the result
        self.neural_to_symbolic_cache[pattern_hash] = result
        
        # Record translation operation
        self.translation_history.append({
            'type': 'neural_to_symbolic',
            'input': {'activation_pattern': '[...]'},  # Simplified for logging
            'output': result,
            'timestamp': time.time()
        })
        
        return result
    
    def translate_structured_meaning_to_symbolic(self, meaning):
        """
        Translate a structured meaning representation to symbolic facts.
        
        Args:
            meaning: Structured meaning representation
            
        Returns:
            List of symbolic facts
        """
        symbolic_facts = []
        
        # Extract propositions from meaning structure
        propositions = meaning.get('propositions', [])
        
        for prop in propositions:
            # Create symbolic fact
            fact = {
                'type': 'symbolic_fact',
                'subject': prop['subject'].lower(),
                'predicate': prop['predicate'].lower(),
                'object': prop['object'].lower(),
                'negated': prop.get('negated', False),
                'confidence': prop.get('confidence', 1.0),
                'translated_from': 'structured_meaning'
            }
            
            symbolic_facts.append(fact)
            
            # Add to Prolog KB if available
            if self.prolog_engine and fact['confidence'] >= self.translation_threshold:
                try:
                    if fact['negated']:
                        # Use special predicate for negated facts
                        neg_pred = f"not_{fact['predicate']}"
                        self.prolog_engine.prolog.assertz(
                            f"confident_fact({neg_pred}, {fact['subject']}, {fact['object']}, {fact['confidence']})"
                        )
                    else:
                        self.prolog_engine.prolog.assertz(
                            f"confident_fact({fact['predicate']}, {fact['subject']}, {fact['object']}, {fact['confidence']})"
                        )
                except Exception as e:
                    print(f"Error asserting fact to Prolog: {e}")
        
        # Record translation operation
        self.translation_history.append({
            'type': 'structured_meaning_to_symbolic',
            'input': {'meaning_id': meaning.get('id', 'unknown')},
            'output': {'fact_count': len(symbolic_facts)},
            'timestamp': time.time()
        })
        
        return symbolic_facts
    
    def translate_symbolic_to_structured_meaning(self, symbolic_facts, source_text=""):
        """
        Translate symbolic facts to a structured meaning representation.
        
        Args:
            symbolic_facts: List of symbolic facts
            source_text: Original text (optional)
            
        Returns:
            Structured meaning representation
        """
        # Create meaning structure
        meaning = {
            'id': str(uuid.uuid4()),
            'type': 'meaning_structure',
            'text': source_text,
            'timestamp': time.time(),
            'concepts': [],
            'propositions': [],
            'frames': []
        }
        
        # Track entities for concept creation
        entities = set()
        
        # Add propositions from symbolic facts
        for fact in symbolic_facts:
            subject = fact['subject'].lower()
            predicate = fact['predicate'].lower()
            object_val = fact['object'].lower()
            
            # Add entities to set
            entities.add(subject)
            if object_val != 'true':  # Skip for unary predicates
                entities.add(object_val)
            
            # Create proposition
            proposition = {
                'id': str(uuid.uuid4()),
                'type': 'proposition',
                'subject': subject,
                'predicate': predicate,
                'object': object_val,
                'negated': fact.get('negated', False),
                'confidence': fact.get('confidence', 1.0),
                'source': 'symbolic_fact'
            }
            
            meaning['propositions'].append(proposition)
        
        # Create concepts from entities
        for entity in entities:
            concept = {
                'id': str(uuid.uuid4()),
                'type': 'entity',
                'text': entity,
                'confidence': 1.0
            }
            
            meaning['concepts'].append(concept)
            
            # Add concept from concept system if available
            if self.concept_system:
                concept_name, confidence = self.concept_system.get_concept_from_text(entity)
                if concept_name and confidence >= 0.7:
                    concept_obj = {
                        'id': str(uuid.uuid4()),
                        'type': 'concept',
                        'text': entity,
                        'concept': concept_name,
                        'confidence': confidence
                    }
                    meaning['concepts'].append(concept_obj)
        
        # Record translation operation
        self.translation_history.append({
            'type': 'symbolic_to_structured_meaning',
            'input': {'fact_count': len(symbolic_facts)},
            'output': {'meaning_id': meaning['id']},
            'timestamp': time.time()
        })
        
        return meaning
    
    def translate_to_meaning_graph(self, source, source_type):
        """
        Translate from any formalism to a meaning graph.
        
        Args:
            source: Source representation
            source_type: Type of source ('symbolic', 'vector', 'neural', 'meaning')
            
        Returns:
            ID of created meaning graph
        """
        if not self.meaning_map:
            return None
            
        # Process based on source type
        if source_type == 'symbolic':
            # Source is a list of symbolic facts
            # First create a structured meaning
            meaning = self.translate_symbolic_to_structured_meaning(
                source, 
                source_text=""
            )
            
            # Then create meaning graph
            text = " ".join([
                f"{p['subject']} {p['predicate']} {p['object']}."
                for p in meaning['propositions']
            ])
            
            graph_id = self.meaning_map.create_meaning_graph(text)
            
            return graph_id
            
        elif source_type == 'vector':
            # Source is a vector representation
            # First translate to symbolic
            if isinstance(source, list):
                # List of vector facts
                symbolic_facts = []
                for vec_fact in source:
                    sym_fact = self.translate_vector_to_symbolic(vec_fact)
                    if sym_fact['type'] != 'translation_failure':
                        symbolic_facts.append(sym_fact)
                
                # Translate symbolic facts to meaning graph
                if symbolic_facts:
                    return self.translate_to_meaning_graph(symbolic_facts, 'symbolic')
            else:
                # Single vector fact
                sym_fact = self.translate_vector_to_symbolic(source)
                if sym_fact['type'] != 'translation_failure':
                    return self.translate_to_meaning_graph([sym_fact], 'symbolic')
                    
            return None
            
        elif source_type == 'neural':
            # Source is a neural pattern
            # First translate to symbolic
            sym_fact = self.translate_neural_to_symbolic(source)
            if sym_fact['type'] != 'translation_failure':
                return self.translate_to_meaning_graph([sym_fact], 'symbolic')
                
            return None
            
        elif source_type == 'meaning':
            # Source is already a structured meaning
            text = source.get('text', "")
            if not text:
                # Construct text from propositions
                text = " ".join([
                    f"{p['subject']} {p['predicate']} {p['object']}."
                    for p in source.get('propositions', [])
                ])
                
            graph_id = self.meaning_map.create_meaning_graph(text)
            return graph_id
            
        return None
    
    def get_translation_stats(self):
        """
        Get statistics about translation operations.
        
        Returns:
            Dict with translation statistics
        """
        stats = {
            'total_translations': len(self.translation_history),
            'by_type': defaultdict(int),
            'success_rate': defaultdict(lambda: {'success': 0, 'failure': 0}),
            'recent_translations': self.translation_history[-10:] if self.translation_history else []
        }
        
        # Calculate stats
        for trans in self.translation_history:
            trans_type = trans['type']
            stats['by_type'][trans_type] += 1
            
            # Check if translation was successful
            output = trans['output']
            if isinstance(output, dict) and output.get('type') == 'translation_failure':
                stats['success_rate'][trans_type]['failure'] += 1
            else:
                stats['success_rate'][trans_type]['success'] += 1
        
        # Calculate percentages
        for trans_type, counts in stats['success_rate'].items():
            total = counts['success'] + counts['failure']
            if total > 0:
                counts['success_percentage'] = (counts['success'] / total) * 100
            else:
                counts['success_percentage'] = 0
        
        return stats