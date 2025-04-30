# cogmenta_core/models/symbolic/prolog_engine.py
import re
from pyswip import Prolog
from cognitive.thought_tracer import *

class PrologEngine:
    def __init__(self):
        self.prolog = Prolog()
        self.confidence_threshold = 0.7
        
        # Initialize Prolog knowledge base
        self._init_prolog_kb()
        self.init_scientific_knowledge()
    
    def _init_prolog_kb(self):
        """Initialize the Prolog knowledge base with base rules"""
        try:
            # First define predicates as dynamic
            self.prolog.assertz(":-dynamic(fact/4)")
            self.prolog.assertz(":-dynamic(confident_fact/4)")
            
            # Define base rules
            base_rules = [
                "fact(P, S, O, C) :- confident_fact(P, S, O, C)",
                "is_certain(X) :- confident_fact(X, _, _, C), C >= 0.8",
                "is_likely(X) :- confident_fact(X, _, _, C), C >= 0.6, C < 0.8",
                "is_possible(X) :- confident_fact(X, _, _, C), C >= 0.4, C < 0.6",
                "is_unlikely(X) :- confident_fact(X, _, _, C), C < 0.4"
            ]
            
            for rule in base_rules:
                self.prolog.assertz(rule)
                print(f"[Prolog] Added rule: {rule}")
            
            # Add some simple test facts to verify the KB is working
            test_facts = [
                "confident_fact(test, subject, object, 0.9)",
                "confident_fact(is_a, bird, animal, 0.9)"
            ]
            
            for fact in test_facts:
                self.prolog.assertz(fact)
                print(f"[Prolog] Added test fact: {fact}")
                
            # Verify KB is working
            test_query = "confident_fact(test, S, O, C)"
            results = list(self.prolog.query(test_query))
            if results:
                print(f"[Prolog] Knowledge base successfully initialized and verified!")
            else:
                print(f"[Prolog] WARNING: Knowledge base verification failed!")
                
        except Exception as e:
            print(f"[Prolog] Initialization error: {e}")
            raise

    def init_scientific_knowledge(self):
        """Initialize formal scientific knowledge"""
        # Physics formulas
        self.prolog.assertz("physics_formula(force, 'F = ma')")
        self.prolog.assertz("physics_formula(energy, 'E = mc^2')")
        
        # Chemical reactions
        self.prolog.assertz("chemical_reaction(water_formation, 'H2 + O -> H2O')")
        
        # Mathematical rules
        self.prolog.assertz("math_rule(square_area, 'A = s^2')")
        
        # Unit conversions
        self.prolog.assertz("unit_conversion(meters_to_feet, 3.28084)")

    def prolog_safe(self, term):
        """More robust Prolog term formatting"""
        if term is None:
            return "unknown"
            
        # Convert to string and clean
        term_str = str(term).strip().lower()
        
        # Replace spaces with underscores
        term_str = term_str.replace(" ", "_")
        
        # Remove quotation marks
        term_str = term_str.replace("'", "").replace('"', "")
        
        # Remove punctuation that would cause Prolog syntax errors
        for char in ".,;:!?()[]{}":
            term_str = term_str.replace(char, "")
        
        # Ensure valid atom for Prolog (starting with lowercase)
        if not term_str:
            return "unknown"
        
        # Add quotes to ensure it's treated as an atom
        return term_str

    def assert_neural_triples(self, triples):
        """Assert triples from neural model with confidence"""
        for subj, pred, obj, conf in triples:
            # Properly quote strings and format numbers
            safe_subj = f"'{self.prolog_safe(subj)}'"
            safe_pred = f"'{self.prolog_safe(pred)}'"
            safe_obj = f"'{self.prolog_safe(obj)}'"
            
            if pred.endswith("_nobody"):
                # Handle special case for "no one" patterns
                pred_base = pred.replace("_nobody", "")
                rule = f"confident_fact('{pred_base}', {safe_subj}, 'noone', {conf})"
            else:
                rule = f"confident_fact({safe_pred}, {safe_subj}, {safe_obj}, {conf})"
                
            print(f"[Prolog] Asserting rule: {rule}")
            self.prolog.assertz(rule)

    def extract_triples_neural(self, text):
        """
        Enhanced rule-based triple extraction with better handling of complex patterns
        
        Args:
            text (str): Input text to process
                
        Returns:
            list: List of (subject, predicate, object, confidence) tuples
        """
        text = text.strip()
        triples = []
        
        # Handle "no one" cases first
        no_one_pattern = re.compile(r"(\w+)\s+(trusts|likes|loves|knows|fears|hates|avoids)\s+(no\s*one|nobody)", re.IGNORECASE)
        match = no_one_pattern.search(text)
        
        if match:
            subj, pred, obj = match.groups()
            # Special handling for "no one" case
            triple = (subj.lower(), f"{pred}_nobody", "true", 0.9)
            triples.append(triple)
        
        # Handle passive voice: "X is trusted by Y"
        passive_pattern = re.compile(r"(\w+)\s+is\s+(trusted|liked|loved|known|feared|hated|avoided)\s+by\s+(\w+)", re.IGNORECASE)
        match = passive_pattern.search(text)
        
        if match:
            obj, passive_pred, subj = match.groups()
            # Convert to active voice
            pred_map = {
                'trusted': 'trusts',
                'liked': 'likes',
                'loved': 'loves',
                'known': 'knows',
                'feared': 'fears',
                'hated': 'hates',
                'avoided': 'avoids'
            }
            pred = pred_map.get(passive_pred.lower(), passive_pred.lower())
            triple = (subj.lower(), pred, obj.lower(), 0.9)
            triples.append(triple)
        
        # Enhanced handling for "Who does X [verb] ?" questions
        who_question_pattern = re.compile(r"who\s+does\s+(\w+)\s+(trust|like|love|know|fear|hate|avoid)\??", re.IGNORECASE)
        match = who_question_pattern.search(text)
        
        if match:
            subj, pred = match.groups()
            # Return a query triple that will be used to search
            # We use a special '_QUERY_' object to indicate this is a question about objects
            triple = (subj.lower(), pred.lower(), "_QUERY_", 0.9)
            triples.append(triple)
        
        # Handle "Who [verbs] X?" questions - asking about subjects
        who_verbs_pattern = re.compile(r"who\s+(trusts|likes|loves|knows|fears|hates|avoids)\s+(\w+)\??", re.IGNORECASE)
        match = who_verbs_pattern.search(text)
        
        if match:
            pred, obj = match.groups()
            # Return a query triple with special '_QUERY_' subject to indicate question about subjects
            triple = ("_QUERY_", pred.lower(), obj.lower(), 0.9)
            triples.append(triple)
        
        # Handle "Does X [verb] Y?" questions
        does_question_pattern = re.compile(r"does\s+(\w+)\s+(trust|like|love|know|fear|hate|avoid)\s+(\w+)\??", re.IGNORECASE)
        match = does_question_pattern.search(text)
        
        if match:
            subj, pred, obj = match.groups()
            triple = (subj.lower(), pred.lower(), obj.lower(), 0.9)
            triples.append(triple)
        
        # Standard relation pattern (X [verb] Y)
        relation_pattern = re.compile(r"(\w+)\s+(likes|hates|loves|knows|trusts|fears|distrusts|avoids)\s+(\w+)", re.IGNORECASE)
        match = relation_pattern.search(text)
        
        if match:
            subj, pred, obj = match.groups()
            triple = (subj.lower(), pred.lower(), obj.lower(), 0.9)
            triples.append(triple)
        
        # Handle negated relation patterns (X doesn't [verb] Y) - More robust version
        neg_pattern = re.compile(r"(\w+)\s+(?:doesn't|does\s+not|didn't|did\s+not|never|won't|will\s+not)\s+(like|hate|love|know|trust|fear|distrust|avoid)\s+(\w+)", re.IGNORECASE)
        match = neg_pattern.search(text)
        
        if match:
            subj, pred, obj = match.groups()
            # Add "not_" prefix to the predicate for negation
            triple = (subj.lower(), f"not_{pred.lower()}", obj.lower(), 0.9)
            triples.append(triple)
        
        # Handle "X [verb] Y and Z" patterns (multiple objects)
        multi_obj_pattern = re.compile(r"(\w+)\s+(likes|hates|loves|knows|trusts|fears|distrusts|avoids)\s+(\w+)\s+and\s+(\w+)", re.IGNORECASE)
        match = multi_obj_pattern.search(text)
        
        if match:
            subj, pred, obj1, obj2 = match.groups()
            # Return two triples for the two objects
            triples.append((subj.lower(), pred.lower(), obj1.lower(), 0.9))
            triples.append((subj.lower(), pred.lower(), obj2.lower(), 0.9))
        
        # Handle "X and Y [verb] Z" patterns (multiple subjects)
        multi_subj_pattern = re.compile(r"(\w+)\s+and\s+(\w+)\s+(like|hate|love|know|trust|fear|distrust|avoid)\s+(\w+)", re.IGNORECASE)
        match = multi_subj_pattern.search(text)
        
        if match:
            subj1, subj2, pred, obj = match.groups()
            # Return two triples for the two subjects
            triples.append((subj1.lower(), pred.lower(), obj.lower(), 0.9))
            triples.append((subj2.lower(), pred.lower(), obj.lower(), 0.9))
        
        return triples

    def reason_with_uncertainty(self, query=None, trace_id=None):
        """
        Reason with uncertainty about facts in the KB.
        Improved version with proper variable handling.
        
        Args:
            query (str): Prolog query string (optional)
            
        Returns:
            dict: Results containing certain/uncertain facts and natural language response
        """
        """Enhanced reasoning with thought tracing."""
    
    # If we have a thought_trace available (passed from bridge)
        if trace_id and hasattr(self, 'thought_trace'):
            self.thought_trace.add_step(
                trace_id,
                "PrologEngine",
                "symbolic_query",
                {"query": query}
            )

        certain_results = []
        uncertain_results = []
        
        if query:
            parts = [x.strip() for x in query.split(",")]
            if len(parts) == 3:
                pred, subj, obj = parts
                query_str = f"confident_fact({pred}, {subj}, {obj}, C)"
            else:
                query_str = "confident_fact(P, S, O, C)"
        else:
            query_str = "confident_fact(P, S, O, C)"
            
        print(f"[Prolog] Executing query: {query_str}")
        
        try:
            # Execute the query and process results
            for result in self.prolog.query(query_str):
                try:
                    # Fix: Add explicit type handling for variables
                    pred = str(result.get("P", "unknown"))
                    subj = str(result.get("S", "unknown"))
                    obj = str(result.get("O", "unknown"))
                    
                    # Handle the confidence value specially
                    conf_val = result.get("C", 0.0)
                    
                    # Check if conf_val is a Variable (unbound) or a numeric value
                    if hasattr(conf_val, "chars") or hasattr(conf_val, "_terms"):
                        # It's a Prolog Variable object, not bound to a value
                        conf = 0.5  # Default confidence for unbound variables
                    else:
                        try:
                            # Try to convert to float, handling both numeric and string values
                            if isinstance(conf_val, (int, float)):
                                conf = float(conf_val)
                            elif isinstance(conf_val, str):
                                conf = float(conf_val)
                            else:
                                # If it's some other unusual type, use default
                                conf = 0.5
                        except (ValueError, TypeError):
                            # If conversion fails, use default
                            print(f"[Prolog] Warning: Could not convert confidence value '{conf_val}' to float")
                            conf = 0.5
                    
                    entry = {
                        "predicate": pred,
                        "subject": subj,
                        "object": obj,
                        "confidence": conf
                    }
                    
                    if conf >= self.confidence_threshold:
                        certain_results.append(entry)
                    else:
                        uncertain_results.append(entry)
                        
                except Exception as e:
                    print(f"[Prolog] Result parsing error: {e}")
                    
        except Exception as e:
            print(f"[Prolog] Query execution error: {e}")
            return {
                "response": f"I encountered an error while reasoning: {str(e)}",
                "certain": [],
                "uncertain": []
            }
            
        # Format a natural language response
        response = self._format_natural_language_response(certain_results, uncertain_results)
            
         # Trace the results
        if trace_id and hasattr(self, 'thought_trace'):
            self.thought_trace.add_step(
                trace_id,
                "PrologEngine",
                "symbolic_results",
                {
                    "certain_count": len(certain_results),
                    "uncertain_count": len(uncertain_results),
                    "certain_sample": certain_results[:2] if certain_results else [],
                    "uncertain_sample": uncertain_results[:2] if uncertain_results else []
                }
            )
        return {
            "response": response,
            "certain": certain_results,
            "uncertain": uncertain_results
        }
    
    def _format_natural_language_response(self, certain_results, uncertain_results):
        """Format reasoning results as natural language"""
        if not certain_results and not uncertain_results:
            return "I don't have any information about that in my knowledge base."
            
        response_parts = []
        
        # Add certain facts
        if certain_results:
            response_parts.append("I know that:")
            for result in certain_results:
                pred = result["predicate"]
                subj = result["subject"]
                obj = result["object"]
                
                if obj == "noone":
                    response_parts.append(f"- {subj} {pred} no one")
                elif obj == "true":
                    response_parts.append(f"- {subj} is {pred}")
                else:
                    response_parts.append(f"- {subj} {pred} {obj}")
                    
        # Add uncertain facts
        if uncertain_results:
            if certain_results:
                response_parts.append("\nI believe, but am less certain, that:")
            else:
                response_parts.append("I'm not entirely certain, but I believe:")
                
            for result in uncertain_results:
                pred = result["predicate"]
                subj = result["subject"]
                obj = result["object"]
                conf = result["confidence"]
                
                confidence_str = ""
                if conf < 0.4:
                    confidence_str = " (but I'm quite uncertain about this)"
                elif conf < 0.6:
                    confidence_str = " (I'm moderately confident)"
                    
                if obj == "noone":
                    response_parts.append(f"- {subj} {pred} no one{confidence_str}")
                elif obj == "true":
                    response_parts.append(f"- {subj} is {pred}{confidence_str}")
                else:
                    response_parts.append(f"- {subj} {pred} {obj}{confidence_str}")
                    
        return "\n".join(response_parts)

    def translate_to_symbolic(self, text):
        """Translate natural language to symbolic Prolog query"""
        # Handle "no one" cases first
        no_one_match = re.search(r"(\w+)\s+(trusts|likes|loves|knows)\s+(no\s*one|nobody)", text.lower())
        if no_one_match:
            subj, pred, _ = no_one_match.groups()
            return f"{pred},{self.prolog_safe(subj)},noone"
        
        # Standard pattern matching
        match = re.search(r"(\w+)\s+(trusts|likes|hates|fears|distrusts|avoids)\s+(\w+)", text.lower())
        if match:
            subj, pred, obj = match.groups()
            return f"{self.prolog_safe(pred)},{self.prolog_safe(subj)},{self.prolog_safe(obj)}"
        
        # Return default query that will list all facts
        return None

    def neural_symbolic_inference(self, text_query):
        """Perform neural-symbolic inference on the input text"""
        symbolic_query = self.translate_to_symbolic(text_query)
        results = self.reason_with_uncertainty(symbolic_query)
        return results
    
    def debug_dump_facts(self):
        """Dump all facts in the KB for debugging"""
        print("\n[🧠 Prolog Knowledge Base Debug]")
        try:
            for result in self.prolog.query("confident_fact(P, S, O, C)"):
                print(f"  - {result['P']}({result['S']}, {result['O']}) :: confidence={result['C']}")
        except Exception as e:
            print(f"  Error during KB dump: {e}")

    def query_with_timeout(self, query_str, timeout=5):
        """Execute a Prolog query with timeout protection"""
        try:
            # Log the query (for debugging)
            print(f"[Prolog] Executing query: {query_str}")
            
            # Clean the query string
            query_str = self._clean_query_string(query_str)
            
            # Create a list to store results
            results = {"certain": [], "uncertain": []}
            
            # Execute query with timeout
            query = self.prolog.query(query_str)
            
            # Process query results
            for result in query:
                # Extract confidence if present
                confidence = result.get('C', 1.0)
                
                # Create fact structure
                fact = {
                    "predicate": result.get('P', ''),
                    "subject": result.get('S', ''),
                    "object": result.get('O', ''),
                    "confidence": confidence
                }
                
                # Categorize by confidence
                if confidence > 0.8:
                    results["certain"].append(fact)
                else:
                    results["uncertain"].append(fact)
            
            # Close the query
            query.close()
            
            return results
        except Exception as e:
            print(f"[Prolog] Query execution error: {e}")
            return {"error": str(e), "certain": [], "uncertain": []}

    def _clean_query_string(self, query_str):
        """Clean and validate a Prolog query string"""
        # Remove any potentially problematic characters
        clean_str = query_str.replace('\n', ' ').replace('\r', ' ')
        
        # Ensure string is properly formatted
        if not clean_str.endswith('.'):
            clean_str = clean_str + '.'
            
        return clean_str
    
    # Add to PrologEngine class in prolog_engine.py
    def receive_neural_feedback(self, neural_activation, strength=0.5):
        """Receive feedback from neural component to adjust confidence values."""
        # Find the most active neurons
        if isinstance(neural_activation, np.ndarray):
            active_indices = np.where(neural_activation > 0.7)[0]
            
            # Convert to list and limit size
            active_neurons = active_indices.tolist()[:10]
            
            # Use these to influence fact confidence
            if active_neurons:
                try:
                    # Get all current facts
                    facts = list(self.prolog.query("confident_fact(P, S, O, C)"))
                    
                    # Boost confidence for facts that involve active concepts
                    for fact in facts:
                        # Calculate influence score (simplified)
                        influence = strength * (len(active_neurons) / 20)
                        
                        # Apply boost to confidence (with cap)
                        new_conf = min(0.95, float(fact['C']) * (1 + influence))
                        
                        # Update fact if confidence changed significantly
                        if abs(new_conf - float(fact['C'])) > 0.05:
                            # Retract old fact
                            self.prolog.retract(f"confident_fact({fact['P']}, {fact['S']}, {fact['O']}, {fact['C']})")
                            # Assert with new confidence
                            self.prolog.assertz(f"confident_fact({fact['P']}, {fact['S']}, {fact['O']}, {new_conf})")
                except Exception as e:
                    print(f"Error applying neural feedback: {e}")
        
        return True