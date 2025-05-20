"""
Conceptual Understanding Layer for Cogmenta Core.
Integrates concept embeddings, structured meaning maps, meaning extraction,
and cross-formalism translation into a unified conceptual layer.
"""

import time
import uuid
from collections import defaultdict

class ConceptualUnderstandingLayer:
    """
    Main integration point for conceptual understanding in Cogmenta Core.
    Combines concept-based embeddings, structured meaning maps, meaning extraction,
    and cross-formalism translation into a unified conceptual understanding layer.
    """
    
    def __init__(self, 
                 concept_system=None, 
                 meaning_map=None, 
                 meaning_extraction=None, 
                 translation_layer=None,
                 vector_symbolic=None,
                 prolog_engine=None,
                 snn=None):
        """
        Initialize the conceptual understanding layer.
        
        Args:
            concept_system: ConceptEmbeddingSystem instance (optional)
            meaning_map: StructuredMeaningMap instance (optional)
            meaning_extraction: MeaningExtractionSystem instance (optional)
            translation_layer: CrossFormalismTranslation instance (optional)
            vector_symbolic: VectorSymbolicEngine instance (optional)
            prolog_engine: PrologEngine instance (optional)
            snn: EnhancedSpikingCore instance (optional)
        """
        self.concept_system = concept_system
        self.meaning_map = meaning_map
        self.meaning_extraction = meaning_extraction
        self.translation_layer = translation_layer
        self.vector_symbolic = vector_symbolic
        self.prolog_engine = prolog_engine
        self.snn = snn
        
        # Create missing components if needed
        if not self.concept_system:
            from conceptual.concept_embeddings import ConceptEmbeddingSystem
            self.concept_system = ConceptEmbeddingSystem()
            print("[Conceptual] Created ConceptEmbeddingSystem")
            
        if not self.meaning_map and self.concept_system:
            from conceptual.structured_meaning_map import StructuredMeaningMap
            self.meaning_map = StructuredMeaningMap(self.concept_system)
            print("[Conceptual] Created StructuredMeaningMap")
            
        if not self.meaning_extraction and self.concept_system and self.meaning_map:
            from conceptual.meaning_extraction import MeaningExtractionSystem
            self.meaning_extraction = MeaningExtractionSystem(
                self.concept_system, 
                self.meaning_map
            )
            print("[Conceptual] Created MeaningExtractionSystem")
            
        if not self.translation_layer:
            from conceptual.cross_formalism_translation import CrossFormalismTranslation
            self.translation_layer = CrossFormalismTranslation(
                concept_system=self.concept_system,
                vector_symbolic=self.vector_symbolic,
                prolog_engine=self.prolog_engine,
                snn=self.snn,
                meaning_map=self.meaning_map
            )
            print("[Conceptual] Created CrossFormalismTranslation")
        
        # For tracking processed inputs
        self.processed_inputs = {}  # text -> processing result
        
        # Integration metrics
        self.metrics = {
            'total_processed': 0,
            'conceptual_activations': defaultdict(int),
            'translation_operations': defaultdict(int),
            'process_times': []
        }
    
    def process_input(self, text):
        """
        Process input text through the conceptual understanding layer.
        
        Args:
            text: Input text to process
            
        Returns:
            Structured conceptual understanding result
        """
        # Check if we've processed this already
        if text in self.processed_inputs:
            return self.processed_inputs[text]
            
        # Measure processing time
        start_time = time.time()
        
        # Initialize result
        result = {
            'id': str(uuid.uuid4()),
            'text': text,
            'timestamp': time.time(),
            'concepts': [],
            'meaning': None,
            'meaning_graph_id': None,
            'translations': {}
        }
        
        # Step 1: Extract concepts using concept system
        if self.concept_system:
            extracted_concepts = self.concept_system.extract_concepts_from_text(text)
            result['concepts'] = [
                {
                    'name': concept,
                    'confidence': confidence,
                    'text_span': text_span
                }
                for concept, confidence, text_span in extracted_concepts
            ]
            
            # Activate concepts in concept system
            for concept, confidence, _ in extracted_concepts:
                self.concept_system.activate_concept(concept, confidence)
                self.metrics['conceptual_activations'][concept] += 1
        
        # Step 2: Extract meaning using meaning extraction system
        if self.meaning_extraction:
            meaning = self.meaning_extraction.extract_meaning(text)
            result['meaning'] = meaning
            
            # Get meaning summary
            result['meaning_summary'] = self.meaning_extraction.extract_meaning_summary(text)
        
        # Step 3: Create structured meaning map if available
        if self.meaning_map:
            graph_id = self.meaning_map.create_meaning_graph(text)
            result['meaning_graph_id'] = graph_id
        
        # Step 4: Perform translations between formalisms
        if self.translation_layer:
            translations = {}
            
            # If we have meaning, translate to symbolic
            if result['meaning']:
                symbolic_facts = self.translation_layer.translate_structured_meaning_to_symbolic(
                    result['meaning']
                )
                translations['meaning_to_symbolic'] = symbolic_facts
                self.metrics['translation_operations']['meaning_to_symbolic'] += 1
                
                # If we have vector symbolic engine, translate to vector
                if self.vector_symbolic:
                    vector_facts = []
                    for fact in symbolic_facts:
                        vector_fact = self.translation_layer.translate_symbolic_to_vector(fact)
                        if vector_fact['type'] != 'translation_failure':
                            vector_facts.append(vector_fact)
                    
                    translations['symbolic_to_vector'] = vector_facts
                    self.metrics['translation_operations']['symbolic_to_vector'] += 1
                
                # If we have SNN, translate to neural
                if self.snn:
                    neural_representations = []
                    for fact in symbolic_facts:
                        neural_rep = self.translation_layer.translate_symbolic_to_neural(fact)
                        if neural_rep['type'] != 'translation_failure':
                            neural_representations.append(neural_rep)
                    
                    translations['symbolic_to_neural'] = neural_representations
                    self.metrics['translation_operations']['symbolic_to_neural'] += 1
            
            result['translations'] = translations
        
        # Finalize metrics
        process_time = time.time() - start_time
        self.metrics['process_times'].append(process_time)
        self.metrics['total_processed'] += 1
        
        result['process_time'] = process_time
        
        # Store result
        self.processed_inputs[text] = result
        
        return result
    
    def query_concepts(self, query_text, min_similarity=0.7):
        """
        Query the conceptual understanding layer for concepts related to a query.
        
        Args:
            query_text: Query text
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary with query results
        """
        results = {
            'concepts': [],
            'related_concepts': [],
            'activated_concepts': []
        }
        
        if not self.concept_system:
            return results
            
        # Find direct concept matches
        concept, similarity = self.concept_system.get_concept_from_text(query_text, min_similarity)
        
        if concept:
            # Get concept data
            metadata = self.concept_system.concept_metadata.get(concept, {})
            
            results['concepts'].append({
                'name': concept,
                'similarity': similarity,
                'metadata': metadata
            })
            
            # Get related concepts
            related = self.concept_system.find_related_concepts(concept)
            
            for relation_type, concepts in related.items():
                for rel_concept, weight in concepts:
                    rel_metadata = self.concept_system.concept_metadata.get(rel_concept, {})
                    
                    results['related_concepts'].append({
                        'name': rel_concept,
                        'relation': relation_type,
                        'weight': weight,
                        'metadata': rel_metadata
                    })
        
        # Get currently activated concepts
        active_concepts = self.concept_system.get_most_activated_concepts(top_n=10)
        
        for concept, activation in active_concepts:
            metadata = self.concept_system.concept_metadata.get(concept, {})
            
            results['activated_concepts'].append({
                'name': concept,
                'activation': activation,
                'metadata': metadata
            })
            
        return results
    
    def compare_texts(self, text1, text2):
        """
        Compare two texts at the conceptual level.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Comparison result
        """
        if not self.meaning_extraction:
            return {"error": "Meaning extraction system not available"}
            
        # Use meaning extraction system to compare texts
        comparison = self.meaning_extraction.compare_meanings(text1, text2)
        
        # Process both texts if not already done
        if text1 not in self.processed_inputs:
            self.process_input(text1)
            
        if text2 not in self.processed_inputs:
            self.process_input(text2)
            
        # Get graph IDs if available
        graph_id1 = self.processed_inputs[text1].get('meaning_graph_id')
        graph_id2 = self.processed_inputs[text2].get('meaning_graph_id')
        
        # Compare meaning graphs if available
        graph_similarity = None
        if self.meaning_map and graph_id1 and graph_id2:
            graph_similarity = self.meaning_map.find_similar_meanings(graph_id1)
            # Extract similarity to the other graph
            for gid, sim in graph_similarity:
                if gid == graph_id2:
                    comparison['graph_similarity'] = sim
                    break
        
        return comparison
    
    def get_conceptual_meaning(self, text):
        """
        Get the conceptual meaning representation for text in a format suitable
        for cross-formalism translation.
        
        Args:
            text: Input text
            
        Returns:
            Conceptual meaning representation
        """
        # Process text if not already processed
        if text not in self.processed_inputs:
            self.process_input(text)
            
        result = self.processed_inputs[text]
        
        # Extract meaningful representation
        conceptual_meaning = {
            'text': text,
            'concepts': [c['name'] for c in result.get('concepts', [])],
            'meaning': result.get('meaning'),
            'meaning_graph_id': result.get('meaning_graph_id'),
            'summary': result.get('meaning_summary')
        }
        
        # Add structured meaning map if available
        if self.meaning_map and result.get('meaning_graph_id'):
            graph = self.meaning_map.meaning_graphs.get(result['meaning_graph_id'])
            if graph:
                conceptual_meaning['meaning_graph'] = {
                    'nodes': graph.get('nodes', []),
                    'edges': graph.get('edges', []),
                    'propositions': graph.get('propositions', [])
                }
                
                # Extract conceptual schema
                schema = self.meaning_map.extract_conceptual_schema(result['meaning_graph_id'])
                if schema:
                    conceptual_meaning['conceptual_schema'] = schema
        
        return conceptual_meaning
    
    def translate_between_formalisms(self, source, source_type, target_type):
        """
        Translate between different formalisms.
        
        Args:
            source: Source representation
            source_type: Type of source ('symbolic', 'vector', 'neural', 'meaning')
            target_type: Type of target ('symbolic', 'vector', 'neural', 'meaning', 'graph')
            
        Returns:
            Translated representation
        """
        if not self.translation_layer:
            return {"error": "Translation layer not available"}
            
        # Handle specific translations based on source and target types
        if source_type == 'symbolic' and target_type == 'vector':
            if isinstance(source, list):
                # List of symbolic facts
                vector_facts = []
                for fact in source:
                    vector_fact = self.translation_layer.translate_symbolic_to_vector(fact)
                    if vector_fact['type'] != 'translation_failure':
                        vector_facts.append(vector_fact)
                return vector_facts
            else:
                # Single symbolic fact
                return self.translation_layer.translate_symbolic_to_vector(source)
                
        elif source_type == 'vector' and target_type == 'symbolic':
            if isinstance(source, list):
                # List of vector facts
                symbolic_facts = []
                for fact in source:
                    symbolic_fact = self.translation_layer.translate_vector_to_symbolic(fact)
                    if symbolic_fact['type'] != 'translation_failure':
                        symbolic_facts.append(symbolic_fact)
                return symbolic_facts
            else:
                # Single vector fact
                return self.translation_layer.translate_vector_to_symbolic(source)
                
        elif source_type == 'symbolic' and target_type == 'neural':
            if isinstance(source, list):
                # List of symbolic facts
                neural_representations = []
                for fact in source:
                    neural_rep = self.translation_layer.translate_symbolic_to_neural(fact)
                    if neural_rep['type'] != 'translation_failure':
                        neural_representations.append(neural_rep)
                return neural_representations
            else:
                # Single symbolic fact
                return self.translation_layer.translate_symbolic_to_neural(source)
                
        elif source_type == 'neural' and target_type == 'symbolic':
            return self.translation_layer.translate_neural_to_symbolic(source)
            
        elif source_type == 'meaning' and target_type == 'symbolic':
            return self.translation_layer.translate_structured_meaning_to_symbolic(source)
            
        elif source_type == 'symbolic' and target_type == 'meaning':
            source_text = ""
            if isinstance(source, list) and len(source) > 0 and 'text' in source[0]:
                source_text = source[0]['text']
            return self.translation_layer.translate_symbolic_to_structured_meaning(source, source_text)
            
        elif target_type == 'graph':
            # Any source type can be translated to a meaning graph
            return self.translation_layer.translate_to_meaning_graph(source, source_type)
            
        else:
            return {"error": f"Unsupported translation: {source_type} to {target_type}"}
    
    def get_active_concepts(self, top_n=10):
        """
        Get the currently most activated concepts.
        
        Args:
            top_n: Number of top concepts to return
            
        Returns:
            List of (concept, activation) tuples
        """
        if not self.concept_system:
            return []
            
        return self.concept_system.get_most_activated_concepts(top_n=top_n)
    
    def get_metrics(self):
        """
        Get metrics about the conceptual understanding layer.
        
        Returns:
            Dict with metrics
        """
        metrics = self.metrics.copy()
        
        # Calculate average processing time
        if self.metrics['process_times']:
            metrics['avg_process_time'] = sum(self.metrics['process_times']) / len(self.metrics['process_times'])
        else:
            metrics['avg_process_time'] = 0
            
        # Get top activated concepts
        if self.concept_system:
            metrics['top_activated_concepts'] = self.get_active_concepts(top_n=10)
            
        # Get translation stats
        if self.translation_layer:
            metrics['translation_stats'] = self.translation_layer.get_translation_stats()
            
        return metrics
    
    def initialize_common_concepts(self):
        """
        Initialize a set of common concepts and their relationships.
        """
        if not self.concept_system:
            return
            
        # Create the entity_type concept first
        self.concept_system.create_concept("entity_type", "Type of entity", category="meta")
        
        # Basic entity types
        self.concept_system.create_concept("person", "A human being", category="entity_type")
        self.concept_system.create_concept("organization", "A group of people", category="entity_type")
        self.concept_system.create_concept("location", "A physical place", category="entity_type")
        self.concept_system.create_concept("event", "Something that happens", category="entity_type")
        self.concept_system.create_concept("object", "A physical thing", category="entity_type")
        self.concept_system.create_concept("concept", "An abstract idea", category="entity_type")
        
        # Social relations
        self.concept_system.create_concept("trusts", "Belief in reliability", category="relation")
        self.concept_system.create_concept("likes", "Positive feeling", category="relation")
        self.concept_system.create_concept("knows", "Awareness of", category="relation")
        self.concept_system.create_concept("fears", "Afraid of", category="relation")
        self.concept_system.create_concept("avoids", "Keeps away from", category="relation")
        
        # Add hierarchy relations
        self.concept_system.add_concept_hierarchy_relation("person", "entity_type")
        self.concept_system.add_concept_hierarchy_relation("organization", "entity_type")
        self.concept_system.add_concept_hierarchy_relation("location", "entity_type")
        self.concept_system.add_concept_hierarchy_relation("event", "entity_type")
        self.concept_system.add_concept_hierarchy_relation("object", "entity_type")
        self.concept_system.add_concept_hierarchy_relation("concept", "entity_type")
        
        # Add other relations
        self.concept_system.add_concept_relation("trusts", "likes", "related_to", 0.7)
        self.concept_system.add_concept_relation("fears", "avoids", "causes", 0.8)
        
        # Initialize some frames in meaning map
        if self.meaning_map:
            # Trust frame
            self.meaning_map.create_frame(
                "trust_relation",
                "One entity trusting another entity",
                [
                    {"name": "truster", "description": "Entity that trusts"},
                    {"name": "trusted", "description": "Entity that is trusted"}
                ],
                [
                    {"predicate": "trusts", "subject_role": "truster", "object_role": "trusted"}
                ]
            )
            
            # Like frame
            self.meaning_map.create_frame(
                "like_relation",
                "One entity liking another entity",
                [
                    {"name": "liker", "description": "Entity that likes"},
                    {"name": "liked", "description": "Entity that is liked"}
                ],
                [
                    {"predicate": "likes", "subject_role": "liker", "object_role": "liked"}
                ]
            )
            
        print("[Conceptual] Initialized common concepts and frames")
        
    def process_text_with_conceptual_understanding(self, text):
        """
        Process text with full conceptual understanding pipeline and generate a
        comprehensive report with all available information.
        
        Args:
            text: Input text to process
            
        Returns:
            Comprehensive understanding report
        """
        # Process text through the layer
        result = self.process_input(text)
        
        # Create comprehensive report
        report = {
            'input_text': text,
            'process_time': result.get('process_time', 0),
            'conceptual_understanding': {
                'concepts': result.get('concepts', []),
                'active_concepts': self.get_active_concepts(top_n=5),
                'meaning_summary': result.get('meaning_summary', ''),
                'propositions': result.get('meaning', {}).get('propositions', []) if result.get('meaning') else []
            },
            'structured_representations': {},
            'translations': {}
        }
        
        # Add structured meaning map if available
        if self.meaning_map and result.get('meaning_graph_id'):
            graph_id = result['meaning_graph_id']
            graph = self.meaning_map.meaning_graphs.get(graph_id, {})
            
            report['structured_representations']['meaning_graph'] = {
                'id': graph_id,
                'node_count': len(graph.get('nodes', [])),
                'edge_count': len(graph.get('edges', [])),
                'proposition_count': len(graph.get('propositions', []))
            }
            
            # Add conceptual schema
            schema = self.meaning_map.extract_conceptual_schema(graph_id)
            if schema:
                report['structured_representations']['conceptual_schema'] = schema
                
            # Add derived implications
            implications = self.meaning_map.derive_meaning_implications(graph_id)
            if implications:
                report['structured_representations']['derived_implications'] = implications
        
        # Add translations if available
        if 'translations' in result:
            for trans_type, trans_result in result['translations'].items():
                if trans_result:
                    report['translations'][trans_type] = {
                        'count': len(trans_result),
                        'first_result': trans_result[0] if len(trans_result) > 0 else None
                    }
        
        return report