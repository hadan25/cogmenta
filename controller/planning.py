"""
Planning system for goal-directed behavior in the cognitive architecture.
Implements plan generation, execution, monitoring, and adjustment.
"""

import time
import re
import random
from collections import defaultdict

class PlanningSystem:
    def __init__(self, symbolic_engine=None):
        """Initialize the planning system"""
        self.symbolic = symbolic_engine  # Prolog reasoning engine
        
        # Goal and plan state
        self.current_goal = None        # Current active goal
        self.current_plan = []          # Sequence of actions to achieve goal
        self.plan_index = 0             # Current position in plan
        self.success_criteria = {}      # Criteria for goal success
        self.alternative_plans = []     # Backup plans
        
        # Planning history
        self.completed_goals = []       # Previously completed goals
        self.failed_goals = []          # Goals that could not be achieved
        self.executed_actions = []      # History of executed actions
        
        # Action templates for basic operations
        self.action_templates = self._init_action_templates()
        
        # Plan monitoring
        self.plan_start_time = None     # When the current plan started
        self.action_timeouts = {}       # Maximum time for each action type
        self.monitoring_active = False  # Whether monitoring is active
    
    def _init_action_templates(self):
        """Initialize action templates that can be used in plans"""
        templates = {
            # Knowledge retrieval actions
            "retrieve_fact": {
                "params": ["subject", "predicate", "object"],
                "preconditions": [],
                "effects": ["knowledge_retrieved"],
                "execution": self._execute_retrieve_fact
            },
            "check_relation": {
                "params": ["entity1", "relation", "entity2"],
                "preconditions": [],
                "effects": ["relation_checked"],
                "execution": self._execute_check_relation
            },
            
            # Reasoning actions
            "infer_from_facts": {
                "params": ["premises", "conclusion"],
                "preconditions": ["knowledge_retrieved"],
                "effects": ["inference_made"],
                "execution": self._execute_infer
            },
            "generate_hypothesis": {
                "params": ["observation", "explanation_type"],
                "preconditions": [],
                "effects": ["hypothesis_generated"],
                "execution": self._execute_generate_hypothesis
            },
            
            # Memory operations
            "store_in_memory": {
                "params": ["content", "memory_type"],
                "preconditions": [],
                "effects": ["memory_stored"],
                "execution": self._execute_store_memory
            },
            "recall_from_memory": {
                "params": ["query", "memory_type"],
                "preconditions": [],
                "effects": ["memory_recalled"],
                "execution": self._execute_recall_memory
            },
            
            # Natural language processing
            "analyze_text": {
                "params": ["text"],
                "preconditions": [],
                "effects": ["text_analyzed"],
                "execution": self._execute_analyze_text
            },
            "extract_entities": {
                "params": ["text"],
                "preconditions": [],
                "effects": ["entities_extracted"],
                "execution": self._execute_extract_entities
            },
            
            # Response generation
            "formulate_response": {
                "params": ["query", "knowledge"],
                "preconditions": ["inference_made"],
                "effects": ["response_formulated"],
                "execution": self._execute_formulate_response
            }
        }
        
        return templates
    
    def set_goal(self, goal_description, goal_type="answer_query", success_criteria=None):
        """
        Set a new goal with success criteria
        
        Args:
            goal_description: Description of the goal
            goal_type: Type of goal (answer_query, solve_problem, etc.)
            success_criteria: Dict of criteria for goal success
            
        Returns:
            The new goal
        """
        # Save current goal if not completed
        if self.current_goal and not self.is_goal_achieved():
            # Store as a failed goal
            self.failed_goals.append(self.current_goal)
            
        # Create new goal
        self.current_goal = {
            "description": goal_description,
            "type": goal_type,
            "timestamp": time.time(),
            "achieved": False
        }
        
        # Set success criteria
        self.success_criteria = success_criteria or {}
        
        # Reset plan
        self.current_plan = []
        self.plan_index = 0
        self.alternative_plans = []
        
        # Generate initial plan
        self._generate_plan()
        
        # Start monitoring
        self.plan_start_time = time.time()
        self.monitoring_active = True
        
        return self.current_goal
    
    def _generate_plan(self):
        """
        Generate a plan to achieve the current goal
        
        Returns:
            The generated plan (list of actions)
        """
        if not self.current_goal:
            return []
            
        goal_type = self.current_goal["type"]
        goal_text = self.current_goal["description"]
        
        # Generate plan based on goal type
        if goal_type == "answer_query":
            self.current_plan = self._generate_query_plan(goal_text)
        elif goal_type == "solve_problem":
            self.current_plan = self._generate_problem_solving_plan(goal_text)
        elif goal_type == "learn_concept":
            self.current_plan = self._generate_learning_plan(goal_text)
        else:
            # Default generic plan
            self.current_plan = self._generate_generic_plan(goal_text)
            
        return self.current_plan
    
    def _generate_query_plan(self, query_text):
        """Generate a plan to answer a query"""
        # Extract key entities and relations from the query
        plan = [
            {
                "action": "analyze_text",
                "params": {"text": query_text}
            },
            {
                "action": "extract_entities",
                "params": {"text": query_text}
            }
        ]
        
        # Determine query type
        if re.search(r'why|how come|reason', query_text.lower()):
            # Explanation query
            plan.extend([
                {
                    "action": "retrieve_fact",
                    "params": {"subject": None, "predicate": None, "object": None}
                },
                {
                    "action": "recall_from_memory",
                    "params": {"query": query_text, "memory_type": "episodic"}
                },
                {
                    "action": "generate_hypothesis",
                    "params": {"observation": query_text, "explanation_type": "causal"}
                },
                {
                    "action": "infer_from_facts",
                    "params": {"premises": None, "conclusion": None}
                }
            ])
        elif re.search(r'what|who|which', query_text.lower()):
            # Factual query
            plan.extend([
                {
                    "action": "retrieve_fact",
                    "params": {"subject": None, "predicate": None, "object": None}
                },
                {
                    "action": "check_relation",
                    "params": {"entity1": None, "relation": None, "entity2": None}
                }
            ])
        else:
            # General query
            plan.extend([
                {
                    "action": "retrieve_fact",
                    "params": {"subject": None, "predicate": None, "object": None}
                },
                {
                    "action": "recall_from_memory",
                    "params": {"query": query_text, "memory_type": "semantic"}
                }
            ])
            
        # Add response formulation
        plan.append({
            "action": "formulate_response",
            "params": {"query": query_text, "knowledge": None}
        })
        
        return plan
    
    def _generate_problem_solving_plan(self, problem_text):
        """Generate a plan to solve a problem"""
        # Basic problem-solving plan
        return [
            {
                "action": "analyze_text",
                "params": {"text": problem_text}
            },
            {
                "action": "extract_entities",
                "params": {"text": problem_text}
            },
            {
                "action": "retrieve_fact",
                "params": {"subject": None, "predicate": None, "object": None}
            },
            {
                "action": "generate_hypothesis",
                "params": {"observation": problem_text, "explanation_type": "solution"}
            },
            {
                "action": "infer_from_facts",
                "params": {"premises": None, "conclusion": None}
            },
            {
                "action": "formulate_response",
                "params": {"query": problem_text, "knowledge": None}
            }
        ]
    
    def _generate_learning_plan(self, concept_text):
        """Generate a plan to learn a concept"""
        # Basic concept learning plan
        return [
            {
                "action": "analyze_text",
                "params": {"text": concept_text}
            },
            {
                "action": "extract_entities",
                "params": {"text": concept_text}
            },
            {
                "action": "retrieve_fact",
                "params": {"subject": None, "predicate": "is_a", "object": None}
            },
            {
                "action": "retrieve_fact",
                "params": {"subject": None, "predicate": "has_property", "object": None}
            },
            {
                "action": "store_in_memory",
                "params": {"content": None, "memory_type": "semantic"}
            }
        ]
    
    def _generate_generic_plan(self, goal_text):
        """Generate a generic plan for any goal"""
        # Basic generic plan
        return [
            {
                "action": "analyze_text",
                "params": {"text": goal_text}
            },
            {
                "action": "extract_entities",
                "params": {"text": goal_text}
            },
            {
                "action": "retrieve_fact",
                "params": {"subject": None, "predicate": None, "object": None}
            },
            {
                "action": "recall_from_memory",
                "params": {"query": goal_text, "memory_type": "episodic"}
            },
            {
                "action": "infer_from_facts",
                "params": {"premises": None, "conclusion": None}
            },
            {
                "action": "formulate_response",
                "params": {"query": goal_text, "knowledge": None}
            }
        ]
    
    def get_next_action(self):
        """
        Get the next action in the current plan
        
        Returns:
            Next action dict or None if plan is complete
        """
        if not self.current_plan or self.plan_index >= len(self.current_plan):
            return None
            
        next_action = self.current_plan[self.plan_index]
        self.plan_index += 1
        return next_action
    
    def execute_action(self, action, action_context=None):
        """
        Execute an action from the plan
        
        Args:
            action: Action dict with 'action' and 'params' keys
            action_context: Additional context for action execution
            
        Returns:
            Result of the action execution
        """
        if not action:
            return {"status": "failure", "error": "No action provided"}
            
        action_name = action.get("action")
        params = action.get("params", {})
        
        # Record action start time
        start_time = time.time()
        
        # Get action template
        template = self.action_templates.get(action_name)
        if not template:
            return {"status": "failure", "error": f"Unknown action: {action_name}"}
            
        # Execute action using template's execution function
        execution_func = template.get("execution")
        if execution_func:
            try:
                result = execution_func(params, action_context)
            except Exception as e:
                result = {"status": "failure", "error": str(e)}
        else:
            result = {"status": "failure", "error": f"No execution function for {action_name}"}
            
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Record action execution
        executed_action = {
            "action": action_name,
            "params": params,
            "result": result,
            "execution_time": execution_time,
            "timestamp": time.time()
        }
        self.executed_actions.append(executed_action)
        
        return result
    
    def update_plan(self, action_result):
        """
        Update plan based on action results
        
        Args:
            action_result: Result dict from action execution
            
        Returns:
            Updated plan
        """
        # If action failed, consider revising the plan
        if action_result.get("status") == "failure":
            # Check if we have alternative plans
            if self.alternative_plans:
                # Switch to the next alternative plan
                self.current_plan = self.alternative_plans.pop(0)
                self.plan_index = 0
                print(f"[Planning] Switching to alternative plan")
            else:
                # Generate a new plan
                self._generate_plan()
                self.plan_index = 0
                print(f"[Planning] Replanning after action failure")
                
        # If the action returned specific information to update the plan
        plan_update = action_result.get("plan_update")
        if plan_update:
            if plan_update.get("type") == "insert_actions":
                # Insert new actions at current position
                new_actions = plan_update.get("actions", [])
                self.current_plan[self.plan_index:self.plan_index] = new_actions
                print(f"[Planning] Inserted {len(new_actions)} new actions into plan")
                
            elif plan_update.get("type") == "replace_actions":
                # Replace a portion of the plan
                start = plan_update.get("start", self.plan_index)
                end = plan_update.get("end", self.plan_index)
                new_actions = plan_update.get("actions", [])
                self.current_plan[start:end] = new_actions
                print(f"[Planning] Replaced actions from index {start} to {end}")
                
            elif plan_update.get("type") == "skip_actions":
                # Skip ahead in the plan
                skip_count = plan_update.get("count", 1)
                self.plan_index += skip_count
                print(f"[Planning] Skipped {skip_count} actions")
                
        return self.current_plan
    
    def is_goal_achieved(self):
        """
        Check if the current goal has been achieved
        
        Returns:
            Boolean indicating goal achievement
        """
        if not self.current_goal:
            return False
            
        # If already marked as achieved
        if self.current_goal.get("achieved", False):
            return True
            
        # Check against success criteria
        criteria_met = True
        
        for criterion, value in self.success_criteria.items():
            if criterion == "required_facts":
                # Check if all required facts are known
                for fact in value:
                    if not self._check_fact_known(fact):
                        criteria_met = False
                        break
                        
            elif criterion == "required_inferences":
                # Check if all required inferences have been made
                for inference in value:
                    if not self._check_inference_made(inference):
                        criteria_met = False
                        break
                        
            elif criterion == "required_actions":
                # Check if all required actions have been performed
                required_actions = set(value)
                performed_actions = {action["action"] for action in self.executed_actions}
                if not required_actions.issubset(performed_actions):
                    criteria_met = False
                    
        # Also consider plan completion as a success indicator
        if self.current_plan and self.plan_index >= len(self.current_plan):
            # All actions in plan have been executed
            if criteria_met:
                self.current_goal["achieved"] = True
                self.completed_goals.append(self.current_goal)
                return True
                
        return False
    
    def _check_fact_known(self, fact):
        """Check if a fact is known in the knowledge base"""
        if not self.symbolic:
            return False
            
        # Extract components from fact string
        match = re.match(r'(\w+)\(([^,]+),\s*([^)]+)\)', fact)
        if not match:
            return False
            
        pred, subj, obj = match.groups()
        
        # Query the knowledge base
        query = f"{pred},{subj},{obj}"
        results = self.symbolic.reason_with_uncertainty(query)
        
        return bool(results.get("certain"))
    
    def _check_inference_made(self, inference):
        """Check if an inference has been recorded in executed actions"""
        for action in self.executed_actions:
            if action["action"] == "infer_from_facts":
                if action["result"].get("status") == "success":
                    # Check if the inference matches
                    inferred = action["result"].get("inference", "")
                    if inference in inferred:
                        return True
        return False
    
    def monitor_plan_execution(self):
        """
        Monitor plan execution for timeouts and failures
        
        Returns:
            Monitoring status dict
        """
        if not self.monitoring_active or not self.current_plan:
            return {"status": "inactive"}
            
        current_time = time.time()
        plan_duration = current_time - self.plan_start_time
        
        # Check for plan timeout
        plan_timeout = self.success_criteria.get("timeout", 300)  # 5 minutes default
        if plan_duration > plan_timeout:
            # Plan has timed out
            self.monitoring_active = False
            self.failed_goals.append(self.current_goal)
            
            return {
                "status": "timeout",
                "message": f"Plan execution timed out after {plan_duration:.1f} seconds"
            }
            
        # Check for action timeouts
        if self.plan_index > 0 and self.plan_index <= len(self.current_plan):
            current_action = self.current_plan[self.plan_index - 1]
            action_type = current_action.get("action")
            
            # Get the timeout for this action type
            action_timeout = self.action_timeouts.get(action_type, 30)  # 30 second default
            
            # Check if the last executed action was started but never completed
            if self.executed_actions and self.executed_actions[-1]["action"] == action_type:
                last_action_time = self.executed_actions[-1]["timestamp"]
                action_duration = current_time - last_action_time
                
                if action_duration > action_timeout:
                    # Action has timed out
                    return {
                        "status": "action_timeout",
                        "action": action_type,
                        "message": f"Action {action_type} timed out after {action_duration:.1f} seconds"
                    }
                    
        return {"status": "active", "plan_duration": plan_duration}
    
    def reset(self):
        """Reset the planning system"""
        self.current_goal = None
        self.current_plan = []
        self.plan_index = 0
        self.success_criteria = {}
        self.alternative_plans = []
        self.monitoring_active = False
        
    # Action execution methods
    
    def _execute_retrieve_fact(self, params, context):
        """Execute a fact retrieval action"""
        if not self.symbolic:
            return {"status": "failure", "error": "No symbolic reasoning engine"}
            
        subject = params.get("subject")
        predicate = params.get("predicate")
        object = params.get("object")
        
        # If params are None, use context to fill them in if available
        if context and "entities" in context:
            if subject is None and "subject" in context["entities"]:
                subject = context["entities"]["subject"]
            if object is None and "object" in context["entities"]:
                object = context["entities"]["object"]
                
        # Build query
        query_parts = []
        if predicate:
            query_parts.append(predicate)
        if subject:
            query_parts.append(subject)
        if object:
            query_parts.append(object)
            
        query = ",".join(query_parts) if query_parts else None
            
        # Execute query
        try:
            results = self.symbolic.reason_with_uncertainty(query)
            return {
                "status": "success",
                "facts": results,
                "query": query
            }
        except Exception as e:
            return {"status": "failure", "error": str(e)}
    
    def _execute_check_relation(self, params, context):
        """Execute a relation check action"""
        entity1 = params.get("entity1")
        relation = params.get("relation")
        entity2 = params.get("entity2")
        
        # Fill in from context if available
        if context and "entities" in context:
            if entity1 is None and "entity1" in context["entities"]:
                entity1 = context["entities"]["entity1"]
            if entity2 is None and "entity2" in context["entities"]:
                entity2 = context["entities"]["entity2"]
                
        # If we're still missing parameters, use the first two entities
        if context and "entities" in context and isinstance(context["entities"], list):
            entities = context["entities"]
            if entity1 is None and len(entities) > 0:
                entity1 = entities[0]
            if entity2 is None and len(entities) > 1:
                entity2 = entities[1]
        
        # If relation is not specified, check common relations
        relations_to_check = [relation] if relation else ["trusts", "likes", "knows", "fears"]
        
        results = {}
        if self.symbolic:
            for rel in relations_to_check:
                query = f"{rel},{entity1},{entity2}"
                rel_results = self.symbolic.reason_with_uncertainty(query)
                if rel_results.get("certain") or rel_results.get("uncertain"):
                    results[rel] = rel_results
        
        return {
            "status": "success" if results else "failure",
            "relations": results,
            "entity1": entity1,
            "entity2": entity2
        }
    
    def _execute_infer(self, params, context):
        """Execute an inference action"""
        premises = params.get("premises")
        conclusion = params.get("conclusion")
        
        # In a real system, this would use formal reasoning
        # Here we use a simple simulation
        
        # If premises not provided, use facts from context
        if not premises and context and "facts" in context:
            premises = context["facts"]
            
        # Simulate inference
        if premises:
            # Generate a plausible inference (simplified)
            inference = f"Inferred from {len(premises)} premises"
            confidence = random.uniform(0.6, 0.9)
            
            return {
                "status": "success",
                "inference": inference,
                "confidence": confidence
            }
        
        return {"status": "failure", "error": "Insufficient premises"}
    
    def _execute_generate_hypothesis(self, params, context):
        """Execute a hypothesis generation action"""
        observation = params.get("observation")
        explanation_type = params.get("explanation_type", "causal")
        
        # In a real system, this would use abductive reasoning
        # Here we simulate hypothesis generation
        
        hypotheses = []
        if observation:
            # Generate dummy hypotheses
            hypotheses = [
                {"hypothesis": f"Hypothesis 1 for {observation}", "confidence": 0.8},
                {"hypothesis": f"Hypothesis 2 for {observation}", "confidence": 0.6},
                {"hypothesis": f"Hypothesis 3 for {observation}", "confidence": 0.4}
            ]
            
        return {
            "status": "success" if hypotheses else "failure",
            "hypotheses": hypotheses,
            "explanation_type": explanation_type
        }
    
    def _execute_store_memory(self, params, context):
        """Execute a memory storage action"""
        content = params.get("content")
        memory_type = params.get("memory_type", "episodic")
        
        # In a real system, this would interface with memory modules
        # Here we simulate memory storage
        
        return {
            "status": "success",
            "message": f"Stored in {memory_type} memory",
            "content_hash": hash(str(content)) if content else None
        }
    
    def _execute_recall_memory(self, params, context):
        """Execute a memory recall action"""
        query = params.get("query")
        memory_type = params.get("memory_type", "episodic")
        
        # In a real system, this would interface with memory modules
        # Here we simulate memory recall
        
        memories = []
        if query:
            # Generate dummy recalled memories
            memories = [
                {"content": f"Memory 1 related to {query}", "relevance": 0.9},
                {"content": f"Memory 2 related to {query}", "relevance": 0.7},
                {"content": f"Memory 3 related to {query}", "relevance": 0.5}
            ]
            
        return {
            "status": "success" if memories else "failure",
            "memories": memories,
            "memory_type": memory_type
        }
    
    def _execute_analyze_text(self, params, context):
        """Execute a text analysis action"""
        text = params.get("text")
        
        if not text:
            return {"status": "failure", "error": "No text provided"}
            
        # In a real system, this would use NLP processing
        # Here we simulate text analysis
        
        analysis = {
            "tokens": text.split(),
            "sentence_count": text.count('.') + text.count('?') + text.count('!'),
            "sentiment": random.uniform(-1, 1)
        }
        
        return {
            "status": "success",
            "analysis": analysis,
            "text": text
        }
    
    def _execute_extract_entities(self, params, context):
        """Execute an entity extraction action"""
        text = params.get("text")
        
        if not text:
            return {"status": "failure", "error": "No text provided"}
            
        # In a real system, this would use NER
        # Here we use simple regex patterns
        
        entities = []
        
        # Extract capitalized words as potential entities
        for match in re.finditer(r'\b([A-Z][a-z]+)\b', text):
            entities.append({
                "text": match.group(1),
                "type": "PERSON",  # Assume all capitalized words are people
                "position": match.start()
            })
            
        # Extract potential relations
        relations = []
        relation_pattern = r'(\w+)\s+(trusts|likes|knows|fears|hates|avoids)\s+(\w+)'
        for match in re.finditer(relation_pattern, text, re.IGNORECASE):
            relations.append({
                "subject": match.group(1),
                "predicate": match.group(2),
                "object": match.group(3)
            })
            
        return {
            "status": "success",
            "entities": entities,
            "relations": relations,
            "text": text
        }
    
    def _execute_formulate_response(self, params, context):
        """Execute a response formulation action"""
        query = params.get("query")
        knowledge = params.get("knowledge")
        
        # Use context if knowledge not provided
        if not knowledge and context:
            knowledge = context
            
        # In a real system, this would generate a natural language response
        # Here we simulate response generation
        
        if not query:
            return {"status": "failure", "error": "No query provided"}
            
        # Generate a simple response based on available knowledge
        response_parts = ["Response to query: " + query]
        
        if isinstance(knowledge, dict):
            if "facts" in knowledge:
                facts = knowledge["facts"]
                if isinstance(facts, dict):
                    if facts.get("certain"):
                        response_parts.append(f"Found {len(facts['certain'])} certain facts.")
                    if facts.get("uncertain"):
                        response_parts.append(f"Found {len(facts['uncertain'])} uncertain facts.")
                        
            if "relations" in knowledge:
                relations = knowledge["relations"]
                if relations:
                    response_parts.append(f"Found information about {len(relations)} relations.")
                    
            if "hypotheses" in knowledge:
                hypotheses = knowledge["hypotheses"]
                if hypotheses:
                    response_parts.append(f"Generated {len(hypotheses)} hypotheses.")
                    
            if "memories" in knowledge:
                memories = knowledge["memories"]
                if memories:
                    response_parts.append(f"Retrieved {len(memories)} memories.")
                    
        response = " ".join(response_parts)
        
        return {
            "status": "success",
            "response": response,
            "query": query
        }