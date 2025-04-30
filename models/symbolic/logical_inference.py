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
        
        # Execute the query
        results = self.prolog.reason_with_uncertainty(symbolic_query)
        
        # Deactivate the counterfactual
        self._deactivate_all_counterfactuals()
        
        return results
    
    def explain_inference(self, conclusion):
        """
        Explain how a conclusion was reached
        Example: "Why do you think Alice trusts Charlie?"
        """
        explanation = {
            "conclusion": conclusion,
            "direct_evidence": [],
            "inferred_from": [],
            "default_rules": [],
            "confidence": 0.0
        }
        
        # Parse the conclusion into predicate, subject, object
        match = re.match(r"(\w+)\((\w+),\s*(\w+)\)", conclusion)
        if not match:
            return explanation
            
        pred, subj, obj = match.groups()
        
        try:
            # Check for direct evidence
            direct_query = f"confident_fact({pred}, {subj}, {obj}, C)"
            for result in self.prolog.prolog.query(direct_query):
                explanation["direct_evidence"].append({
                    "fact": f"{pred}({subj}, {obj})",
                    "confidence": float(result["C"])
                })
                explanation["confidence"] = max(explanation["confidence"], float(result["C"]))
            
            # Check for inferences through rules
            rule_query = f"confident_rule(R, P, {pred}({subj}, {obj}), C), confident_fact(P, C2)"
            for result in self.prolog.prolog.query(rule_query):
                rule_id = str(result["R"])
                premise = str(result["P"])
                rule_conf = float(result["C"])
                fact_conf = float(result["C2"])
                
                # Compute combined confidence
                combined_conf = rule_conf * fact_conf
                
                explanation["inferred_from"].append({
                    "rule": f"{premise} -> {pred}({subj}, {obj})",
                    "rule_confidence": rule_conf,
                    "premise_confidence": fact_conf,
                    "combined_confidence": combined_conf
                })
                explanation["confidence"] = max(explanation["confidence"], combined_conf)
            
            # Check if default rules were used
            for rule in self.default_rules:
                # This is a simplified check - would need more complex unification in a real system
                if rule["conclusion"] == f"{pred}({subj}, {obj})" or f"{pred}({subj}" in rule["conclusion"]:
                    explanation["default_rules"].append({
                        "rule": f"{rule['premise']} -> {rule['conclusion']}",
                        "confidence": rule["confidence"],
                        "exceptions": rule["exceptions"]
                    })
        except Exception as e:
            print(f"[Logical] Explanation generation failed: {e}")
        
        return explanation