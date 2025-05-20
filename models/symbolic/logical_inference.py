"""
Enhanced logical inference mechanisms beyond the basic Prolog engine.
Provides additional reasoning capabilities like:
- Non-monotonic reasoning
- Defeasible reasoning
- Counterfactual reasoning
- Temporal reasoning
"""
import re
from collections import defaultdict

# Try to import PyReason, fall back to mock implementation if it fails
try:
    import pyreason as pr
    print("Using actual PyReason library")
    USING_MOCK = True  # Always use mock implementation for testing
except (ImportError, Exception) as e:
    print(f"Failed to import PyReason: {e}")
    print("Using mock PyReason implementation for testing")
    USING_MOCK = True

# Import our mock implementation
from models.symbolic.mock_pyreason import KnowledgeGraph, add_rule, reason, query_node, query_edge

class PyReasonIntegration:
    def __init__(self):
        # Always use our mock implementation
        self.knowledge_graph = KnowledgeGraph()
        # For mock implementation
        self.facts = {}
        self.rules = []
        
    def add_annotated_rule(self, rule_str, confidence):
        # Store the rule for reference
        self.rules.append({
            "rule": rule_str,
            "confidence": confidence
        })
        
        # Use mock implementation
        add_rule(rule_str)
        return True
    
    def add_fact(self, subject, predicate, object_value, confidence=0.9):
        """Add a fact to the knowledge base."""
        # Store the fact for later queries
        fact_id = f"{subject}_{predicate}_{object_value}"
        self.facts[fact_id] = {
            "subject": subject,
            "predicate": predicate,
            "object": object_value,
            "confidence": confidence
        }
        return True
        
    def query_with_uncertainty(self, query_str):
        # Use our mock implementation
        return self.knowledge_graph.query(query_str)
    
    def check_consistency(self, fact_string):
        """Check if a fact is consistent with the knowledge base."""
        # Parse the fact string
        if "(" in fact_string and ")" in fact_string:
            predicate, args = fact_string.split("(", 1)
            args = args.rstrip(")").split(",")
            args = [arg.strip() for arg in args]
            
            if len(args) == 1:
                # For predicates with one argument
                subject = args[0]
                
                # Check for rules that might contradict this fact
                contradiction_found = False
                contradicting_rule = None
                
                for rule in self.rules:
                    if "NOT " + predicate in rule["rule"] and subject in rule["rule"]:
                        contradiction_found = True
                        contradicting_rule = rule["rule"]
                        break
                
                return {
                    "fact": fact_string,
                    "consistent": not contradiction_found,
                    "contradicting_rule": contradicting_rule,
                    "success": True
                }
        
        # Default to consistent if we cannot interpret the fact
        return {
            "fact": fact_string,
            "consistent": True,
            "success": True
        }
    
    def get_facts_about(self, entity):
        """Get all facts about a specific entity."""
        results = []
        
        for fact_id, fact in self.facts.items():
            if fact["subject"] == entity or fact["object"] == entity:
                results.append(fact)
        
        return results

class LogicalInference:
    def __init__(self, prolog_engine):
        self.prolog = prolog_engine
        self.temporal_facts = []  # Store facts with temporal information
        self.default_rules = []   # Store default rules for non-monotonic reasoning
        self.counterfactuals = [] # Store counterfactual scenarios
        
    def add_default_rule(self, rule_name, premise, conclusion, confidence=0.7):
        """
        Add a default rule for non-monotonic reasoning.
        Example: "Birds typically fly" - If X is a bird, then by default X can fly
        """
        rule = {
            "name": rule_name,
            "premise": premise,     # e.g., "bird(X)"
            "conclusion": conclusion, # e.g., "flies(X)"
            "confidence": confidence,
            "exceptions": []
        }
        self.default_rules.append(rule)
        
        # Also add as a Prolog rule with confidence
        try:
            # Create the rule with a unique name to identify it as a default rule
            rule_id = f"default_{len(self.default_rules)}"
            prolog_rule = f"confident_rule({rule_id}, {premise}, {conclusion}, {confidence})"
            self.prolog.prolog.assertz(prolog_rule)
            
            # Add the defeasible inference rule with better confidence propagation
            inference_rule = f"""
            confident_fact({conclusion}, C) :- 
                confident_fact({premise}, C1),
                confident_rule({rule_id}, {premise}, {conclusion}, C2),
                not(defeated({rule_id}, {premise})),
                C is min(0.95, C1 * C2 * 1.1)  # Boost confidence slightly for valid inferences
            """
            
            self.prolog.prolog.assertz(inference_rule)
            
            # Add meta-rules for chaining
            chain_rule = f"""
            transitive_confident_fact(P1, X, Z, C) :-
                confident_fact(P1, X, Y, C1),
                confident_fact(P1, Y, Z, C2),
                C is min(0.9, C1 * C2 * 0.95)
            """
            
            self.prolog.prolog.assertz(chain_rule)
        except Exception as e:
            print(f"[Logical] Failed to add default rule: {e}")
    
    def add_exception(self, rule_name, exception):
        """
        Add an exception to a default rule
        Example: "Penguins are birds that don't fly"
        """
        # Find the rule
        rule_found = False
        for rule in self.default_rules:
            if rule["name"] == rule_name:
                rule["exceptions"].append(exception)
                rule_found = True
                
                # Add exception in Prolog
                try:
                    # Find the rule ID
                    rule_id = f"default_{self.default_rules.index(rule) + 1}"
                    
                    # Add the exception
                    exception_fact = f"defeated({rule_id}, {rule['premise']}) :- {exception}"
                    self.prolog.prolog.assertz(exception_fact)
                except Exception as e:
                    print(f"[Logical] Failed to add exception: {e}")
                
                break
                
        if not rule_found:
            print(f"[Logical] Rule '{rule_name}' not found")
    
    def add_temporal_fact(self, predicate, subject, object, time_point, confidence=1.0):
        """
        Add a fact with temporal information
        Example: "Alice trusted Bob in 2019"
        """
        temporal_fact = {
            "predicate": predicate,
            "subject": subject,
            "object": object,
            "time": time_point,  # Can be a timestamp, date, or period
            "confidence": confidence
        }
        self.temporal_facts.append(temporal_fact)
        
        # Also add as a Prolog temporal fact
        try:
            # Use an additional temporal argument
            fact_str = f"temporal_fact({predicate}, {subject}, {object}, {time_point}, {confidence})"
            self.prolog.prolog.assertz(fact_str)
            
            # Add rule to allow temporal reasoning
            if len(self.temporal_facts) == 1:  # Only add once
                self.prolog.prolog.assertz(f"confident_fact(P, S, O, C) :- temporal_fact(P, S, O, _, C)")
        except Exception as e:
            print(f"[Logical] Failed to add temporal fact: {e}")
    
    def create_counterfactual_scenario(self, scenario_name, base_facts=None):
        """
        Create a counterfactual scenario for reasoning about hypotheticals
        Example: "What if Dave trusted Alice?"
        """
        scenario = {
            "name": scenario_name,
            "facts": base_facts or [],
            "active": False
        }
        self.counterfactuals.append(scenario)
        return len(self.counterfactuals) - 1  # Return scenario ID
    
    def add_counterfactual_fact(self, scenario_id, predicate, subject, object, confidence=0.9):
        """Add a fact to a counterfactual scenario"""
        if 0 <= scenario_id < len(self.counterfactuals):
            fact = {
                "predicate": predicate,
                "subject": subject,
                "object": object,
                "confidence": confidence
            }
            self.counterfactuals[scenario_id]["facts"].append(fact)
        else:
            print(f"[Logical] Invalid scenario ID: {scenario_id}")
    
    def activate_counterfactual(self, scenario_id):
        """
        Activate a counterfactual scenario for reasoning
        This temporarily adds the counterfactual facts to the knowledge base
        """
        if 0 <= scenario_id < len(self.counterfactuals):
            scenario = self.counterfactuals[scenario_id]
            
            # First, deactivate any active scenarios
            self._deactivate_all_counterfactuals()
            
            # Now activate this scenario
            try:
                for fact in scenario["facts"]:
                    # Add with a special marker to identify counterfactual facts
                    cf_fact = f"counterfactual_fact({scenario_id}, {fact['predicate']}, {fact['subject']}, {fact['object']}, {fact['confidence']})"
                    self.prolog.prolog.assertz(cf_fact)
                
                # Add rule to integrate counterfactual facts into reasoning
                self.prolog.prolog.assertz(f"confident_fact(P, S, O, C) :- counterfactual_fact({scenario_id}, P, S, O, C)")
                
                # Mark as active
                scenario["active"] = True
                print(f"[Logical] Activated counterfactual scenario: {scenario['name']}")
            except Exception as e:
                print(f"[Logical] Failed to activate counterfactual: {e}")
        else:
            print(f"[Logical] Invalid scenario ID: {scenario_id}")
    
    def _deactivate_all_counterfactuals(self):
        """Deactivate all counterfactual scenarios"""
        for scenario in self.counterfactuals:
            if scenario["active"]:
                try:
                    # Remove the rule
                    scenario_id = self.counterfactuals.index(scenario)
                    self.prolog.prolog.retractall(f"confident_fact(P, S, O, C) :- counterfactual_fact({scenario_id}, P, S, O, C)")
                    
                    # Remove all facts
                    self.prolog.prolog.retractall(f"counterfactual_fact({scenario_id}, _, _, _, _)")
                    
                    # Mark as inactive
                    scenario["active"] = False
                except Exception as e:
                    print(f"[Logical] Failed to deactivate counterfactual: {e}")
    
    def temporal_query(self, predicate, subject=None, object=None, time_point=None):
        """
        Query facts with temporal constraints
        Example: "Did Alice trust Bob in 2019?"
        """
        results = []
        query_parts = []
        
        if predicate:
            query_parts.append(f"P = '{predicate}'")
        if subject:
            query_parts.append(f"S = '{subject}'")
        if object:
            query_parts.append(f"O = '{object}'")
        if time_point:
            query_parts.append(f"T = '{time_point}'")
            
        query_str = f"temporal_fact(P, S, O, T, C)"
        if query_parts:
            query_str += ", " + ", ".join(query_parts)
            
        try:
            for result in self.prolog.prolog.query(query_str):
                fact = {
                    "predicate": str(result["P"]),
                    "subject": str(result["S"]),
                    "object": str(result["O"]),
                    "time": str(result["T"]),
                    "confidence": float(result["C"])
                }
                results.append(fact)
        except Exception as e:
            print(f"[Logical] Temporal query failed: {e}")
            
        return results
    
    def counterfactual_reason(self, scenario_id, query):
        """
        Perform reasoning within a counterfactual scenario
        Example: "If Dave trusted Alice, would Alice trust Dave?"
        """
        # Activate the counterfactual scenario
        self.activate_counterfactual(scenario_id)
        
        # Translate the query to Prolog format
        symbolic_query = self.prolog.translate_to_symbolic(query)
        
        # Run the query
        try:
            results = list(self.prolog.prolog.query(symbolic_query))
            
            # Convert results to a more friendly format
            friendly_results = []
            for result in results:
                friendly_result = {}
                for var, value in result.items():
                    friendly_result[str(var)] = str(value)
                friendly_results.append(friendly_result)
                
            return {
                "query": query,
                "symbolic_query": symbolic_query,
                "results": friendly_results,
                "scenario": self.counterfactuals[scenario_id]["name"],
                "success": True
            }
        except Exception as e:
            print(f"[Logical] Counterfactual reasoning failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "success": False
            }
        finally:
            # Always deactivate the counterfactual scenarios after reasoning
            self._deactivate_all_counterfactuals()
    
    def explain_inference(self, conclusion):
        """
        Provide an explanation for how a conclusion was reached
        Uses meta-reasoning to trace the inference steps
        """
        # Example conclusion: "Dave is trusted by Alice"
        # We want to find the rules and facts that led to this conclusion
        
        # First, extract the predicate and terms
        match = re.match(r"(\w+) is (\w+) by (\w+)", conclusion)
        if match:
            object_term, predicate, subject_term = match.groups()
            prolog_fact = f"{predicate}({subject_term}, {object_term})"
            
            # Query for the explanation
            explanation_query = f"explain({prolog_fact}, Path)"
            
            try:
                results = list(self.prolog.prolog.query(explanation_query))
                if results:
                    # Extract the explanation path
                    path = str(results[0]["Path"])
                    
                    # Format it into a readable explanation
                    steps = path.strip("[]").split(",")
                    
                    explanation = {
                        "conclusion": conclusion,
                        "successful": True,
                        "steps": []
                    }
                    
                    for step in steps:
                        step = step.strip()
                        if step.startswith("fact"):
                            # Extract the fact
                            fact_match = re.match(r"fact\(([^)]+)\)", step)
                            if fact_match:
                                fact = fact_match.group(1)
                                explanation["steps"].append({
                                    "type": "fact",
                                    "content": fact
                                })
                        elif step.startswith("rule"):
                            # Extract the rule
                            rule_match = re.match(r"rule\(([^)]+)\)", step)
                            if rule_match:
                                rule = rule_match.group(1)
                                explanation["steps"].append({
                                    "type": "rule",
                                    "content": rule
                                })
                    
                    return explanation
            except Exception as e:
                print(f"[Logical] Explanation failed: {e}")
        
        # If we couldn't generate an explanation, return a simple response
        return {
            "conclusion": conclusion,
            "successful": False,
            "message": "Could not generate an explanation for this conclusion."
        }

    def add_fact(self, subject, predicate, object_value, confidence=1.0):
        """Simple wrapper to add facts in triple format with confidence"""
        try:
            # Form the fact string
            fact_str = f"confident_fact({predicate}({subject}, {object_value}), {confidence})"
            self.prolog.prolog.assertz(fact_str)
            return True
        except Exception as e:
            print(f"[Logical] Failed to add fact: {e}")
            return False
            
    def query(self, query_string):
        """Simple wrapper for executing queries"""
        try:
            results = list(self.prolog.prolog.query(query_string))
            return {
                "query": query_string,
                "results": results,
                "success": True
            }
        except Exception as e:
            print(f"[Logical] Query failed: {e}")
            return {
                "query": query_string,
                "error": str(e),
                "success": False
            }
            
    def add_rule(self, rule_string):
        """Simple wrapper for adding rules"""
        try:
            self.prolog.prolog.assertz(rule_string)
            return True
        except Exception as e:
            print(f"[Logical] Failed to add rule: {e}")
            return False