# cogmenta_core/models/hybrid/knowledge_mediator.py
class KnowledgeMediator:
    """
    Mediates knowledge between different symbolic formalisms,
    maintaining consistency and enabling cross-formalism queries.
    """
    
    def __init__(self, prolog_engine=None, vector_engine=None):
        self.engines = {}
        if prolog_engine:
            self.engines['prolog'] = prolog_engine
        if vector_engine:
            self.engines['vector'] = vector_engine
        
        # Track shared knowledge
        self.knowledge_map = {}  # Maps facts to their representations in each engine
        
    def translate_fact(self, fact, source_formalism, target_formalism):
        """Translate a fact from one formalism to another."""
        # Implementation depends on the specific formalisms
        if source_formalism == 'prolog' and target_formalism == 'vector':
            return self._prolog_to_vector(fact)
        elif source_formalism == 'vector' and target_formalism == 'prolog':
            return self._vector_to_prolog(fact)
        # Add other translations as needed
        
    def _prolog_to_vector(self, prolog_fact):
        """Convert a Prolog fact to vector representation."""
        # Extract components from Prolog fact
        # Usually in form: predicate(subject, object, confidence)
        # Convert to vector format: {subject, predicate, object, confidence}
        # Implementation details depend on your specific formats
        
    def _vector_to_prolog(self, vector_fact):
        """Convert a vector fact to Prolog representation."""
        # Convert {subject, predicate, object, confidence} to 
        # predicate(subject, object, confidence)
        
    def assert_fact(self, fact, primary_formalism):
        """
        Assert a fact, propagating it to all relevant formalisms.
        
        Args:
            fact: The fact to assert
            primary_formalism: The formalism this fact is coming from
        """
        # Assert in the primary formalism first
        primary_engine = self.engines.get(primary_formalism)
        if not primary_engine:
            return False
            
        primary_result = primary_engine.assert_fact(fact)
        
        # Track in knowledge map
        fact_id = self._generate_fact_id(fact, primary_formalism)
        self.knowledge_map[fact_id] = {
            primary_formalism: fact
        }
        
        # Propagate to other formalisms
        for formalism, engine in self.engines.items():
            if formalism != primary_formalism:
                translated_fact = self.translate_fact(
                    fact, primary_formalism, formalism
                )
                if translated_fact:
                    engine.assert_fact(translated_fact)
                    self.knowledge_map[fact_id][formalism] = translated_fact
        
        return primary_result
        
    def query(self, query, preferred_formalism=None):
        """
        Query across all formalisms, using the preferred one first.
        
        Args:
            query: The query to execute
            preferred_formalism: The preferred formalism to try first
        """
        results = []
        attempted_formalisms = []
        
        # Try preferred formalism first
        if preferred_formalism and preferred_formalism in self.engines:
            engine = self.engines[preferred_formalism]
            try:
                formalism_results = engine.query(query)
                results.extend(formalism_results)
                attempted_formalisms.append(preferred_formalism)
            except Exception as e:
                print(f"Error querying {preferred_formalism}: {e}")
        
        # Try other formalisms
        for formalism, engine in self.engines.items():
            if formalism not in attempted_formalisms:
                try:
                    # Translate query if needed
                    if preferred_formalism:
                        translated_query = self.translate_fact(
                            query, preferred_formalism, formalism
                        )
                    else:
                        translated_query = query
                        
                    formalism_results = engine.query(translated_query)
                    results.extend(formalism_results)
                except Exception as e:
                    print(f"Error querying {formalism}: {e}")
        
        return results
        
    def _generate_fact_id(self, fact, formalism):
        """Generate a unique ID for a fact to track it across formalisms."""
        # Implementation depends on fact structure
        # Could be a hash of the key components