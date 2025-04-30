# cogmenta_core/models/hybrid/dual_snn_coordinator.py

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any

class DualSNNCoordinator:
    """
    Coordinates between two specialized SNNs:
    - Statistical SNN: Handles generalization, similarity, and fuzzy matching
    - Abductive SNN: Handles hypothesis generation and abductive reasoning
    
    Provides integrated interface and handles information flow between
    the two specialized networks and the symbolic system.
    """
    
    def __init__(self, statistical_snn=None, abductive_snn=None):
        """Initialize with existing SNNs or create new ones"""
        # Import here to avoid circular imports
        from models.snn.statistical_snn import StatisticalSNN
        from models.snn.enhanced_spiking_core import EnhancedSpikingCore
        
        # Create or use existing SNNs
        self.statistical_snn = statistical_snn or StatisticalSNN()
        self.abductive_snn = abductive_snn or EnhancedSpikingCore()
        
        # Shared concept space for alignment
        self.shared_concept_space = {}
        
        # Mapping between statistical and abductive concepts
        self.concept_mapping = {}
        
        # Integration metrics
        self.integration_level = 0.0
        self.current_task = None
        
        print("[DualSNN] Coordinator initialized with Statistical and Abductive SNNs")
    
    def process_input(self, text_input, task_type=None):
        """
        Process input with both SNNs and integrate results
        
        Args:
            text_input: Text input to process
            task_type: Specific task type or None for automatic detection
            
        Returns:
            Integrated results from both networks
        """
        # Determine task type if not specified
        if task_type is None:
            task_type = self._detect_task_type(text_input)
        
        self.current_task = task_type
        
        # Process with both networks
        statistical_result = None
        abductive_result = None
        
        # Process with statistical SNN
        if task_type in ['similarity', 'generalization', 'completion', 'analogy']:
            # These tasks favor statistical processing
            statistical_result = self.statistical_snn.process_input(text_input, task_type)
            
            # Process with abductive SNN only for supporting hypotheses
            abductive_result = self.abductive_snn.process_input(text_input)
            
            # Combine results with statistical taking priority
            integrated_result = self._integrate_results(
                statistical_result, 
                abductive_result,
                primary='statistical'
            )
        elif task_type in ['abduction', 'hypothesis', 'causal']:
            # These tasks favor abductive processing
            abductive_result = self.abductive_snn.process_input(text_input)
            
            # Get hypotheses from abductive network
            hypotheses = []
            if hasattr(self.abductive_snn, 'abductive_reasoning'):
                hypotheses = self.abductive_snn.abductive_reasoning(text_input)
            
            # Process with statistical SNN to find similar concepts
            statistical_result = self.statistical_snn.process_input(text_input, 'similarity')
            
            # Enhance hypotheses with statistical similarity
            enhanced_hypotheses = self._enhance_hypotheses(hypotheses, statistical_result)
            
            # Combine results with abductive taking priority
            integrated_result = self._integrate_results(
                statistical_result, 
                abductive_result,
                primary='abductive',
                enhanced_hypotheses=enhanced_hypotheses
            )
        else:
            # Balanced processing for general tasks
            statistical_result = self.statistical_snn.process_input(text_input, 'similarity')
            abductive_result = self.abductive_snn.process_input(text_input)
            
            # Get hypotheses from abductive network
            hypotheses = []
            if hasattr(self.abductive_snn, 'abductive_reasoning'):
                hypotheses = self.abductive_snn.abductive_reasoning(text_input)
            
            # Enhance hypotheses with statistical similarity
            enhanced_hypotheses = self._enhance_hypotheses(hypotheses, statistical_result)
            
            # Combine results with balanced integration
            integrated_result = self._integrate_results(
                statistical_result, 
                abductive_result,
                primary='balanced',
                enhanced_hypotheses=enhanced_hypotheses
            )
        
        # Calculate integration level (phi)
        self.integration_level = self._calculate_integration_level(
            statistical_result, abductive_result
        )
        
        # Add integration metrics to result
        integrated_result['integration_level'] = self.integration_level
        integrated_result['task_type'] = task_type
        
        return integrated_result
    
    def _detect_task_type(self, text_input):
        """Detect the appropriate task type based on input text"""
        text_lower = text_input.lower()
        
        # Check for similarity queries
        if any(term in text_lower for term in ['similar', 'like', 'related to', 'resembles']):
            return 'similarity'
            
        # Check for generalization queries
        if any(term in text_lower for term in ['generalize', 'common', 'pattern', 'group']):
            return 'generalization'
            
        # Check for completion queries
        if any(term in text_lower for term in ['complete', 'fill in', 'missing', 'rest of']):
            return 'completion'
            
        # Check for analogy queries
        if any(term in text_lower for term in ['analogy', 'is to', 'as', 'equivalent']):
            return 'analogy'
            
        # Check for abductive/hypothesis queries
        if any(term in text_lower for term in ['why', 'hypothesis', 'reason', 'explain', 'could be']):
            return 'abduction'
            
        # Check for causal queries
        if any(term in text_lower for term in ['cause', 'effect', 'leads to', 'results in']):
            return 'causal'
            
        # Default to balanced processing
        return 'balanced'
    
    def _enhance_hypotheses(self, hypotheses, statistical_result):
        """Enhance hypotheses with statistical similarity information"""
        enhanced = []
        
        # Extract similar concepts from statistical result
        similar_concepts = []
        if (isinstance(statistical_result, dict) and 
            'similar_concepts' in statistical_result):
            similar_concepts = statistical_result['similar_concepts']
        
        # Process each hypothesis
        for hypo in hypotheses:
            # Parse hypothesis
            hypo_parts = self._parse_hypothesis(hypo)
            if not hypo_parts:
                enhanced.append({'hypothesis': hypo, 'confidence': 0.5})
                continue
                
            subject, predicate, obj = hypo_parts
            
            # Check if subject or object are similar to known concepts
            subj_similarities = []
            obj_similarities = []
            
            for concept, sim_score in similar_concepts:
                # Calculate string similarity
                subj_sim = self._string_similarity(subject, concept)
                if subj_sim > 0.7:
                    subj_similarities.append((concept, subj_sim * sim_score))
                    
                obj_sim = self._string_similarity(obj, concept)
                if obj_sim > 0.7:
                    obj_similarities.append((concept, obj_sim * sim_score))
            
            # Calculate confidence boost based on similarity
            confidence_boost = 0.0
            
            if subj_similarities:
                confidence_boost += max(sim for _, sim in subj_similarities) * 0.3
                
            if obj_similarities:
                confidence_boost += max(sim for _, sim in obj_similarities) * 0.3
            
            # Base confidence - 0.5 plus boost (up to 0.3 from similarities)
            confidence = min(0.9, 0.5 + confidence_boost)
            
            enhanced.append({
                'hypothesis': hypo,
                'confidence': confidence,
                'subject_similarities': subj_similarities[:2],  # Top 2
                'object_similarities': obj_similarities[:2]     # Top 2
            })
        
        return enhanced
    
    def _parse_hypothesis(self, hypothesis):
        """Parse a hypothesis string into components"""
        import re
        
        # Match pattern like "predicate(subject, object)"
        match = re.match(r"(\w+)\((\w+)(?:,\s*(\w+))?\)", hypothesis)
        if match:
            groups = match.groups()
            predicate = groups[0]
            subject = groups[1]
            obj = groups[2] if len(groups) > 2 and groups[2] else "true"
            
            return (subject, predicate, obj)
        
        return None
    
    def _string_similarity(self, str1, str2):
        """Calculate simple string similarity"""
        # Convert to lowercase
        s1 = str1.lower()
        s2 = str2.lower()
        
        # Check for exact match
        if s1 == s2:
            return 1.0
            
        # Check for substring
        if s1 in s2 or s2 in s1:
            # Return ratio of shorter to longer length
            return min(len(s1), len(s2)) / max(len(s1), len(s2))
        
        # Count matching characters (very simple approach)
        matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))
        
        # Return ratio of matches to average length
        avg_len = (len(s1) + len(s2)) / 2
        if avg_len > 0:
            return matches / avg_len
        else:
            return 0.0
    
    def _integrate_results(self, statistical_result, abductive_result, 
                           primary='balanced', enhanced_hypotheses=None):
        """
        Integrate results from both networks
        
        Args:
            statistical_result: Result from statistical SNN
            abductive_result: Result from abductive SNN
            primary: Which result takes priority ('statistical', 'abductive', or 'balanced')
            enhanced_hypotheses: Enhanced hypotheses with confidence (optional)
            
        Returns:
            Integrated results
        """
        # Initialize integrated result
        integrated = {
            'result_type': primary,
            'statistical': statistical_result,
            'abductive': abductive_result
        }
        
        # Add enhanced hypotheses if available
        if enhanced_hypotheses:
            integrated['hypotheses'] = enhanced_hypotheses
        
        # Extract phi value from abductive result if available
        if (isinstance(abductive_result, dict) and 
            'phi' in abductive_result):
            integrated['phi'] = abductive_result['phi']
        
        # Extract emergent pattern if available
        if (isinstance(abductive_result, dict) and 
            'emergent_pattern' in abductive_result and 
            abductive_result['emergent_pattern']):
            integrated['emergent_pattern'] = abductive_result['emergent_pattern']
        
        # Process based on primary result type
        if primary == 'statistical':
            # Add statistical operation results
            if isinstance(statistical_result, dict):
                # Add operation-specific results
                if 'operation' in statistical_result:
                    operation = statistical_result['operation']
                    integrated['operation'] = operation
                    
                    # Add relevant operation results
                    if operation == 'similarity' and 'similar_concepts' in statistical_result:
                        integrated['similar_concepts'] = statistical_result['similar_concepts']
                        
                    elif operation == 'generalization' and 'generalizations' in statistical_result:
                        integrated['generalizations'] = statistical_result['generalizations']
                        
                    elif operation == 'completion' and 'completed_pattern' in statistical_result:
                        integrated['completed_pattern'] = statistical_result['completed_pattern']
                        
                    elif operation == 'analogy' and 'result' in statistical_result:
                        integrated['analogy_result'] = statistical_result['result']
        
        elif primary == 'abductive':
            # Add abductive reasoning results
            if isinstance(abductive_result, dict):
                # Add key abductive results
                for key in ['region_activations', 'membrane_potentials']:
                    if key in abductive_result:
                        integrated[key] = abductive_result[key]
                
                # Add enhanced hypotheses (with statistical information)
                if enhanced_hypotheses:
                    # Sort by confidence
                    sorted_hypotheses = sorted(
                        enhanced_hypotheses, 
                        key=lambda x: x.get('confidence', 0), 
                        reverse=True
                    )
                    integrated['primary_hypothesis'] = sorted_hypotheses[0] if sorted_hypotheses else None
                elif 'hypotheses' in integrated:
                    # Sort by confidence
                    sorted_hypotheses = sorted(
                        integrated['hypotheses'], 
                        key=lambda x: x.get('confidence', 0), 
                        reverse=True
                    )
                    integrated['primary_hypothesis'] = sorted_hypotheses[0] if sorted_hypotheses else None
        
        else:  # balanced
            # Include both types of results with equal priority
            if isinstance(statistical_result, dict) and 'similar_concepts' in statistical_result:
                integrated['similar_concepts'] = statistical_result['similar_concepts']
                
            if enhanced_hypotheses:
                # Sort by confidence
                sorted_hypotheses = sorted(
                    enhanced_hypotheses, 
                    key=lambda x: x.get('confidence', 0), 
                    reverse=True
                )
                integrated['primary_hypothesis'] = sorted_hypotheses[0] if sorted_hypotheses else None
        
        return integrated
    
    def _calculate_integration_level(self, statistical_result, abductive_result):
        """Calculate integration level (phi) between the two networks"""
        # Base phi value
        phi = 0.3  # Minimum integration
        
        # Factors affecting integration
        factors = []
        
        # Both networks active with significant results
        if statistical_result and abductive_result:
            factors.append(0.2)  # Both active
            
            # Check for shared concepts between results
            stat_concepts = self._extract_concepts(statistical_result)
            abd_concepts = self._extract_concepts(abductive_result)
            
            shared = len(set(stat_concepts).intersection(set(abd_concepts)))
            if shared > 0:
                factors.append(min(0.3, 0.1 * shared))  # Shared concepts boost
        
        # Abductive network has high phi
        if (isinstance(abductive_result, dict) and 
            'phi' in abductive_result):
            factors.append(abductive_result['phi'] * 0.5)  # Scale abductive phi
        
        # Statistical network has high confidence
        if (isinstance(statistical_result, dict) and 
            'confidence' in statistical_result):
            factors.append(statistical_result['confidence'] * 0.3)
            
        # Special case for enhanced hypotheses showing integration
        if (hasattr(self, 'current_task') and 
            self.current_task in ['abduction', 'hypothesis']):
            factors.append(0.1)  # Task-specific boost
        
        # Sum all factors and cap at 0.9
        phi += sum(factors)
        phi = min(0.9, phi)
        
        return phi
    
    def _extract_concepts(self, result):
        """Extract concept names from a result dictionary"""
        concepts = []
        
        if not isinstance(result, dict):
            return concepts
            
        # Look for similar concepts
        if 'similar_concepts' in result:
            for concept, _ in result['similar_concepts']:
                concepts.append(concept)
                
        # Look for generalizations
        if 'generalizations' in result:
            if isinstance(result['generalizations'], list):
                for concept, _ in result['generalizations']:
                    concepts.append(concept)
            elif isinstance(result['generalizations'], dict) and 'generalized_concepts' in result['generalizations']:
                for concept, _ in result['generalizations']['generalized_concepts']:
                    concepts.append(concept)
        
        # Look for hypotheses
        if 'hypotheses' in result:
            for hypo in result['hypotheses']:
                if isinstance(hypo, str):
                    # Extract name from hypothesis string
                    parts = self._parse_hypothesis(hypo)
                    if parts:
                        subject, _, obj = parts
                        concepts.append(subject)
                        concepts.append(obj)
                elif isinstance(hypo, dict) and 'hypothesis' in hypo:
                    # Extract name from hypothesis dict
                    parts = self._parse_hypothesis(hypo['hypothesis'])
                    if parts:
                        subject, _, obj = parts
                        concepts.append(subject)
                        concepts.append(obj)
        
        return concepts
    
    def learn_from_examples(self, examples, concept_name=None):
        """
        Learn from examples with both networks
        
        Args:
            examples: List of examples to learn from
            concept_name: Name of concept to learn
            
        Returns:
            Learning results from both networks
        """
        # Learn with statistical network (for statistical patterns)
        stat_result = self.statistical_snn.learn_from_examples(
            examples, concept_name, learning_type='few_shot'
        )
        
        # Learn with abductive network (if it has learning capability)
        abd_result = None
        if hasattr(self.abductive_snn, 'learning') and hasattr(self.abductive_snn.learning, 'train_pattern'):
            # Convert examples to patterns for abductive learning
            patterns = []
            for example in examples:
                if isinstance(example, dict) and 'features' in example:
                    patterns.append(example['features'])
                elif isinstance(example, np.ndarray):
                    patterns.append(example)
            
            if patterns and len(patterns[0]) == self.abductive_snn.neuron_count:
                # Create target pattern (average of examples)
                target_pattern = np.mean(patterns, axis=0)
                
                # Train abductive network
                abd_result = self.abductive_snn.learning.train_pattern(
                    patterns[0],  # Use first pattern as input
                    target_pattern,
                    epochs=10
                )
        
        # Map the concept between networks for future alignment
        if concept_name and isinstance(stat_result, dict) and 'embedding' in stat_result:
            self.shared_concept_space[concept_name] = {
                'statistical_embedding': stat_result['embedding'],
                'learning_time': time.time()
            }
            
            # Record concept mapping
            self.concept_mapping[concept_name] = {
                'statistical': True,
                'abductive': abd_result is not None
            }
        
        return {
            'statistical_learning': stat_result,
            'abductive_learning': abd_result,
            'concept_name': concept_name,
            'shared_mapping': concept_name in self.shared_concept_space
        }
    
    def generate_hypotheses(self, input_text, num_hypotheses=5, use_statistical=True):
        """
        Generate hypotheses using combined capabilities
        
        Args:
            input_text: Input text to generate hypotheses about
            num_hypotheses: Number of hypotheses to generate
            use_statistical: Whether to enhance with statistical information
            
        Returns:
            Generated hypotheses with confidence scores
        """
        # Get base hypotheses from abductive network
        hypotheses = []
        
        if hasattr(self.abductive_snn, 'abductive_reasoning'):
            # Get hypotheses from abductive reasoning
            hypotheses = self.abductive_snn.abductive_reasoning(input_text)
            
            # Limit to requested number
            hypotheses = hypotheses[:num_hypotheses]
        
        # If no hypotheses or too few, generate some based on statistical similarity
        if len(hypotheses) < num_hypotheses and use_statistical:
            # Process with statistical network to find similar concepts
            stat_result = self.statistical_snn.process_input(input_text, 'similarity')
            
            if isinstance(stat_result, dict) and 'similar_concepts' in stat_result:
                similar_concepts = stat_result['similar_concepts']
                
                # Extract potential subjects and predicates
                parts = self._extract_entities(input_text)
                subjects = parts.get('subjects', [])
                predicates = parts.get('predicates', [])
                objects = parts.get('objects', [])
                
                # If no subjects extracted, use similar concepts
                if not subjects and similar_concepts:
                    subjects = [concept for concept, _ in similar_concepts]
                
                # Default predicates if none extracted
                if not predicates:
                    predicates = ['relates_to', 'similar_to', 'connected_with']
                
                # Default objects if none extracted
                if not objects and similar_concepts and len(similar_concepts) > 1:
                    objects = [concept for concept, _ in similar_concepts[1:]]
                
                # Generate additional hypotheses
                while len(hypotheses) < num_hypotheses and subjects and (predicates or objects):
                    # Select random elements
                    subject = subjects[0] if subjects else "entity"
                    predicate = predicates[0] if predicates else "relates_to"
                    obj = objects[0] if objects else "concept"
                    
                    # Create hypothesis
                    hypothesis = f"{predicate}({subject}, {obj})"
                    
                    # Add if not already present
                    if hypothesis not in hypotheses:
                        hypotheses.append(hypothesis)
                    
                    # Rotate lists for variety
                    if subjects:
                        subjects = subjects[1:] + [subjects[0]]
                    if predicates:
                        predicates = predicates[1:] + [predicates[0]]
                    if objects:
                        objects = objects[1:] + [objects[0]]
        
        # Enhance hypotheses with statistical information
        if use_statistical:
            # Process with statistical network
            stat_result = self.statistical_snn.process_input(input_text, 'similarity')
            
            # Enhance hypotheses
            enhanced = self._enhance_hypotheses(hypotheses, stat_result)
            
            # Sort by confidence
            enhanced.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            return {
                'hypotheses': enhanced,
                'count': len(enhanced),
                'statistical_enhancement': True,
                'primary_hypothesis': enhanced[0] if enhanced else None
            }
        else:
            # Return basic hypotheses without enhancement
            return {
                'hypotheses': [{'hypothesis': h, 'confidence': 0.5} for h in hypotheses],
                'count': len(hypotheses),
                'statistical_enhancement': False
            }
    
    def _extract_entities(self, text):
        """Extract potential entities and predicates from text"""
        text_lower = text.lower()
        words = text_lower.split()
        
        # Simple extraction heuristics
        subjects = []
        predicates = []
        objects = []
        
        # Common entity patterns
        entity_patterns = [
            r'\b([A-Z][a-z]+)\b',  # Capitalized words (names)
            r'\b(alice|bob|charlie|dave|eve|frank)\b',  # Common example names
            r'\b(person|people|human|man|woman|child)\b',  # Person types
            r'\b(dog|cat|bird|animal)\b',  # Animals
            r'\b(tree|plant|flower)\b',  # Plants
            r'\b(car|vehicle|machine|computer)\b'  # Objects
        ]
        
        # Common predicate patterns
        predicate_patterns = [
            r'\b(is|are|was|were)\b',  # Being
            r'\b(has|have|had)\b',  # Possession
            r'\b(likes?|loves?|hates?|fears?)\b',  # Emotions
            r'\b(knows?|thinks?|believes?|understands?)\b',  # Cognition
            r'\b(causes?|leads?\s+to|results?\s+in)\b',  # Causation
            r'\b(relates?\s+to|connects?\s+with)\b'  # Relations
        ]
        
        # Extract with regex
        for pattern in entity_patterns:
            import re
            matches = re.findall(pattern, text)
            for match in matches:
                if match not in subjects and len(subjects) < 3:
                    subjects.append(match)
                elif match not in objects and len(objects) < 3:
                    objects.append(match)
        
        for pattern in predicate_patterns:
            import re
            matches = re.findall(pattern, text)
            for match in matches:
                if match not in predicates and len(predicates) < 3:
                    predicates.append(match)
        
        return {
            'subjects': subjects,
            'predicates': predicates,
            'objects': objects
        }
    
    def process_symbolic_result(self, symbolic_facts):
        """
        Process symbolic facts with both networks
        
        Args:
            symbolic_facts: Facts from symbolic system
            
        Returns:
            Processing results from both networks
        """
        # Process with abductive network
        abd_result = None
        if hasattr(self.abductive_snn, 'process_symbolic_result'):
            abd_result = self.abductive_snn.process_symbolic_result(symbolic_facts)
        
        # Extract unique concepts from facts
        concepts = set()
        for fact in symbolic_facts:
            if isinstance(fact, dict):
                if 'subject' in fact:
                    concepts.add(fact['subject'])
                if 'predicate' in fact:
                    concepts.add(fact['predicate'])
                if 'object' in fact:
                    concepts.add(fact['object'])
        
        # Process concepts with statistical network
        stat_results = {}
        for concept in concepts:
            if concept in self.statistical_snn.concept_embeddings:
                # Find similar concepts
                similar = self.statistical_snn.find_similar_concepts(
                    query_concept=concept, top_k=3
                )
                stat_results[concept] = similar
            else:
                # Learn concept embedding if not already known
                embedding = self.statistical_snn.learn_concept_embedding(concept)
                stat_results[concept] = [('new_concept', 1.0)]
        
        return {
            'abductive_result': abd_result,
            'statistical_result': stat_results,
            'concepts_processed': len(concepts),
            'new_concepts_learned': sum(1 for r in stat_results.values() if r[0][0] == 'new_concept')
        }
    
    def get_integration_metrics(self):
        """Get metrics about the system's integration level"""
        metrics = {
            'integration_level': self.integration_level,
            'statistical_concepts': len(self.statistical_snn.concept_embeddings),
            'concept_mappings': len(self.concept_mapping),
            'shared_concepts': len(self.shared_concept_space)
        }
        
        # Add abductive metrics if available
        if hasattr(self.abductive_snn, 'phi'):
            metrics['abductive_phi'] = self.abductive_snn.phi
        
        if hasattr(self.abductive_snn, 'regions'):
            regions = {}
            for name, region in self.abductive_snn.regions.items():
                regions[name] = region.get('activation', 0)
            metrics['abductive_regions'] = regions
        
        return metrics