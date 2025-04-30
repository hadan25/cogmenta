# cogmenta_core/models/hybrid/enhanced_neuro_symbolic_bridge.py
import re
import random
import time
import math
from models.symbolic.prolog_engine import PrologEngine
from models.snn.enhanced_spiking_core import EnhancedSpikingCore
from models.hybrid.dual_snn_coordinator import DualSNNCoordinator
from models.memory.episodic_memory import EpisodicMemory
from models.memory.semantic_memory import SemanticMemory
from models.symbolic.knowledge_abstraction import KnowledgeAbstraction
from processing.nlp_pipeline import NLPPipeline
from processing.context_manager import ContextManager
from models.symbolic.vector_symbolic import VectorSymbolicEngine
from models.hybrid.formalism_router import FormalismRouter, FormalismType
from cognitive.thought_tracer import ThoughtTrace

class EnhancedNeuroSymbolicBridge:
    """
    Enhanced Neuro-Symbolic Bridge with Integrated Information Theory concepts.
    Implements consciousness-like properties through integration and recurrent processing.
    """
    def __init__(self, use_enhanced_snn=True, thought_trace=None):
        """Initialize the bridge with required components"""
        # Initialize all components
        self.symbolic = PrologEngine()
        self.vector_symbolic = VectorSymbolicEngine(dimension=300, sparsity=0.1)
        self.thought_trace = thought_trace  # Store thought trace

         # Add thought trace system
        self.thought_trace = thought_trace or ThoughtTrace()

        # Create formalism router
        self.formalism_router = FormalismRouter(
        prolog_engine=self.symbolic,
        vector_engine=self.vector_symbolic
        )

        # Use dual SNN coordinator instead of a single SNN
        if use_enhanced_snn:
            from models.snn.enhanced_spiking_core import EnhancedSpikingCore
            from models.snn.statistical_snn import StatisticalSNN
            
            # Create the individual SNNs
            self.abductive_snn = EnhancedSpikingCore()
            self.statistical_snn = StatisticalSNN()
            
            # Create the coordinator
            self.snn = DualSNNCoordinator(
                statistical_snn=self.statistical_snn,
                abductive_snn=self.abductive_snn
            )
        else:
            # Fall back to regular SpikingCore if enhanced version not available
            from models.snn.spiking_core import SpikingCore
            self.snn = SpikingCore()

        # Memory systems
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        
        # Processing components
        self.abstraction = KnowledgeAbstraction(self.symbolic)
        self.nlp = NLPPipeline()
        self.context = ContextManager()
        
        # IIT integration parameters
        self.integration_level = 0.0  # Phi (Φ) value - integration measure
        self.subsystem_activities = {
            'symbolic': 0.0,
            'neural': 0.0,
            'abductive': 0.0,
            'memory': 0.0,
            'reasoning': 0.0,
            'vector_symbolic': 0.0  # Add the vector-symbolic subsystem
        }
        
        # Global Workspace parameters (for consciousness-like broadcasting)
        self.global_workspace = {
            'current_focus': None,
            'broadcast_strength': 0.0,
            'active_elements': [],
            'last_update': time.time()
        }
        
        # Recurrent Processing (for consciousness emergence)
        self.recurrent_loops = 0
        self.recurrent_threshold = 3  # Minimum loops for conscious-like processing
        self.recurrent_history = []
        
        # Integration metrics history
        self.integration_history = []
        
        print("[Bridge] Enhanced Neuro-Symbolic Bridge initialized with IIT consciousness concepts")
    
    def _calculate_phi(self):
        """
        Calculate the system's integration level (Phi/Φ value) with improved sensitivity.
        Higher values indicate more consciousness-like behavior.
        """
        # Get activity values of all subsystems with minimum values to avoid zeros
        activities = [max(0.1, v) for v in self.subsystem_activities.values()]
        
        # Ensure we have some activity to measure
        if not activities or all(a <= 0.1 for a in activities):
            return 0.0  # No integration if minimal activity
        
        # Calculate mean activity
        mean_activity = sum(activities) / len(activities)
        
        # Calculate variance (differentiation)
        variance = sum((a - mean_activity) ** 2 for a in activities) / len(activities)
        differentiation = math.sqrt(variance)  # Use square root for better scaling
        
        # Calculate connectivity (integration)
        # Count how many subsystems are active above threshold
        active_subsystems = sum(1 for a in activities if a > 0.3)
        integration = active_subsystems / len(activities)
        
        # Factor in recurrent processing
        recurrent_factor = min(1.0, self.recurrent_loops / 5)  # Cap at 1.0
        
        # Phi calculation (enhanced version of IIT's Φ)
        # Phi is high when both differentiated (high variance) and integrated (high connectivity)
        phi = differentiation * integration * mean_activity
        
        # Boost phi based on recurrent processing
        phi = phi * (1.0 + 0.5 * recurrent_factor)  # Boost phi by up to 50% based on recurrence
        
        # Update the integration level
        self.integration_level = phi
        
        return phi
    
    def _update_global_workspace(self, focus_element, strength):
        """
        Update the global workspace with the current focus of attention.
        This implements aspects of Global Workspace Theory.
        """
        # Update workspace
        self.global_workspace['current_focus'] = focus_element
        self.global_workspace['broadcast_strength'] = strength
        self.global_workspace['last_update'] = time.time()
        
        # Add to active elements if strong enough (broadcast)
        if strength > 0.5:
            if focus_element not in self.global_workspace['active_elements']:
                self.global_workspace['active_elements'].append(focus_element)
                
        # Decay old elements
        self.global_workspace['active_elements'] = [
            e for e in self.global_workspace['active_elements']
            if e == focus_element or self.global_workspace['last_update'] - time.time() < 60
        ]
        
    def _handle_prolog_error(self, error_msg, input_text):
        """Handle cases where Prolog processing fails"""
        print(f"[Bridge] Handling Prolog error: {error_msg}")
        
        # Update subsystem activity - symbolic reasoning failed
        self.subsystem_activities['symbolic'] = 0.1
        
        # Extract any meaningful information from the input
        lower_text = input_text.lower()
        
        # Handle basic conversation starters
        if any(greeting in lower_text for greeting in ["hi", "hello", "hey", "greetings"]):
            return {"response": "Hello! I'm still learning. How can I help you today?"}
        
        if "how are you" in lower_text:
            return {"response": "I'm functioning well, thank you for asking. How can I assist you?"}
        
        # Default response for errors
        return {"response": "I'm having trouble processing that with my knowledge base. Could you try asking about something specific, like relationships between people?"}

    def process_text_and_reason(self, text):
        """Process text input and perform neural-symbolic reasoning with dual SNN integration"""
        # Set up thought tracing
        trace_id = None
        if hasattr(self, 'thought_trace') and self.thought_trace:
            trace_id = self.thought_trace.start_trace(
                trigger=f"Process: '{text}'",
                source_component="EnhancedNeuroSymbolicBridge"
            )

        # Create default result structure
        results = {
            'neural': {
                'success': False,
                'data': {},
                'error': None
            },
            'symbolic': {
                'success': False,
                'data': {},
                'error': None
            },
            'integrated': {
                'success': False,
                'data': {},
                'error': None
            },
            'response': None,
            'trace_id': trace_id
        }

        print(f"\n[Bridge] Processing input: '{text}'")
        
        # Check for conversation first
        conversation_response = self.handle_conversation(text)
        if conversation_response:
            if trace_id:
                self.thought_trace.end_trace(trace_id, "Conversation handled")
                conversation_response['trace_id'] = trace_id
            return conversation_response

        # Determine formalism type for processing
        formalism = self.formalism_router.determine_formalism(text)
        
        try:
            # STEP 1: Neural processing with error handling
            try:
                # Process text with NLP pipeline
                nlp_result = self.nlp.process(text)
                
                # Update context
                self.context.add_utterance(text, nlp_result)
                
                # Store in episodic memory
                self.episodic.store_episode(text, importance=0.7)
                
                # Update neural subsystem activity based on NLP success
                self.subsystem_activities['neural'] = 0.5
                
                # Extract relations using NLP
                relations = nlp_result.get('relations', [])
                triples = []
                
                # Convert relations to triples
                for relation in relations:
                    if relation.get('negated'):
                        # Handle negated relations differently
                        triple = (relation['subject'], f"not_{relation['predicate']}", relation['object'], relation['confidence'])
                    else:
                        triple = (relation['subject'], relation['predicate'], relation['object'], relation['confidence'])
                    triples.append(triple)
                
                # Mark neural processing as successful
                results['neural'] = {
                    'success': True,
                    'data': {
                        'nlp_result': nlp_result,
                        'relations': relations,
                        'triples': triples
                    },
                    'error': None
                }
            except Exception as e:
                # Neural processing failed
                error_msg = f"Neural processing error: {str(e)}"
                print(f"[Bridge] {error_msg}")
                results['neural'] = {
                    'success': False,
                    'data': {},
                    'error': error_msg
                }
                # Continue with Symbolic processing

            # STEP 2: Determine if we need Dual SNN processing
            # Look for task type indicators in the text for specialized processing
            if "why" in text.lower() or "explain" in text.lower() or "reason" in text.lower():
                # This is likely an abductive reasoning task
                print("[Bridge] Detected abductive reasoning task")
                task_type = "abduction"
            elif "similar" in text.lower() or "like" in text.lower() or "related" in text.lower():
                # This is likely a similarity task
                print("[Bridge] Detected similarity task")
                task_type = "similarity"
            elif "generalize" in text.lower() or "pattern" in text.lower():
                # This is likely a generalization task
                print("[Bridge] Detected generalization task")
                task_type = "generalization"
            elif "analogy" in text.lower() or "is to" in text.lower() or "as" in text.lower():
                # This is likely an analogy task
                print("[Bridge] Detected analogy task")
                task_type = "analogy"
            else:
                # Default to balanced processing
                task_type = "balanced"

            # STEP 3: Process based on selected formalism
            if formalism == FormalismType.VECTOR_SYMBOLIC:
                # STEP 3A: Vector Symbolic processing
                try:
                    # Process with vector symbolic engine
                    vector_results = self.vector_symbolic.process_text(text)
                    
                    # Update subsystem activity
                    self.subsystem_activities['vector_symbolic'] = 0.7
                    
                    # Mark vector symbolic processing as successful
                    results['vector_symbolic'] = {
                        'success': True,
                        'data': vector_results,
                        'error': None
                    }
                    
                    # Set response
                    if 'response' in vector_results:
                        results['response'] = vector_results['response']
                        
                    # Store facts in memory
                    for fact in vector_results.get('facts', []):
                        self.semantic.add_relation(
                            fact['subject'], fact['predicate'], fact['object'], fact['confidence']
                        )
                except Exception as e:
                    # Vector symbolic processing failed
                    error_msg = f"Vector symbolic processing error: {str(e)}"
                    print(f"[Bridge] {error_msg}")
                    results['vector_symbolic'] = {
                        'success': False,
                        'data': {},
                        'error': error_msg
                    }
                    
                    # Fall back to symbolic processing
                    formalism = FormalismType.PROLOG
                    print(f"[Bridge] Falling back to {formalism.value} processing")
            
            if formalism == FormalismType.PROLOG:
                # STEP 3B: Symbolic processing with Dual SNN integration
                try:
                    # If no triples from NLP, try pattern matching if not already tried
                    if not triples and not results['neural'].get('data', {}).get('fallback', False):
                        triples = self.symbolic.extract_triples_neural(text)
                        
                        if triples:
                            print(f"[Bridge] Pattern matching extracted {len(triples)} triples")
                            
                            # Update neural results with the pattern matching triples
                            if results['neural']['success']:
                                results['neural']['data']['triples'] = triples
                            else:
                                results['neural'] = {
                                    'success': True,
                                    'data': {
                                        'nlp_result': None,
                                        'relations': [],
                                        'triples': triples,
                                        'fallback': True
                                    },
                                    'error': results['neural']['error']
                                }
                        else:
                            print("[Bridge] No triples extracted using patterns")

                    # Assert triples to Prolog knowledge base if available
                    if triples:
                        print(f"[Bridge] Processing {len(triples)} triples:")
                        for t in triples:
                            print(f"  - {t}")
                        
                        # Assert triples to Prolog knowledge base
                        self.symbolic.assert_neural_triples(triples)
                        
                        # Boost symbolic activity when triples are found
                        self.subsystem_activities['symbolic'] = 0.7
                    else:
                        print(f"[Bridge] No triples extracted")
                        self.subsystem_activities['symbolic'] = 0.3
                    
                    # Try symbolic reasoning first
                    symbolic_results = self.symbolic.neural_symbolic_inference(text)
                    
                    # If we have confident results, boost reasoning activity
                    if symbolic_results.get('certain'):
                        self.subsystem_activities['reasoning'] = 0.7
                        
                        # Update global workspace with a certain fact
                        if symbolic_results['certain']:
                            fact = symbolic_results['certain'][0]
                            triple_str = f"{fact['predicate']}({fact['subject']}, {fact['object']})"
                            self._update_global_workspace(triple_str, 0.8)
                    else:
                        # No confident results - trigger dual SNN processing
                        print("[Bridge] No confident symbolic answer. Triggering dual SNN processing.")
                        self.subsystem_activities['reasoning'] = 0.4
                        
                        # Use the dual SNN coordinator for processing
                        snn_result = self.snn.process_input(text, task_type=task_type)
                        
                        # Update neural activity based on SNN result
                        self.subsystem_activities['neural'] = 0.7
                        self.subsystem_activities['abductive'] = 0.6
                        
                        # Extract and process results based on task type
                        if task_type == "abduction":
                            # Extract hypotheses for abductive reasoning
                            hypotheses = []
                            # Check for hypotheses in structured format
                            if 'enhanced_hypotheses' in snn_result:
                                for hypo_data in snn_result['enhanced_hypotheses']:
                                    hypothesis = hypo_data['hypothesis']
                                    confidence = hypo_data.get('confidence', 0.5)
                                    hypotheses.append((hypothesis, confidence))
                            # Check for primary hypothesis
                            elif 'primary_hypothesis' in snn_result and snn_result['primary_hypothesis']:
                                hypothesis = snn_result['primary_hypothesis']['hypothesis']
                                confidence = snn_result['primary_hypothesis'].get('confidence', 0.5)
                                hypotheses.append((hypothesis, confidence))
                            # Fallback to hypotheses list
                            elif 'hypotheses' in snn_result:
                                for i, hypo in enumerate(snn_result['hypotheses']):
                                    # Assign decreasing confidence to later hypotheses
                                    confidence = max(0.3, 0.8 - (i * 0.1))
                                    hypotheses.append((hypo, confidence))
                            
                            # Inject hypotheses into Prolog knowledge base
                            for hypothesis, confidence in hypotheses:
                                try:
                                    # Parse the hypothesis for structured format
                                    match = re.match(r"(\w+)\((\w+)(?:,\s*(\w+))?\)", hypothesis)
                                    if match:
                                        groups = match.groups()
                                        pred = groups[0]
                                        subj = groups[1]
                                        
                                        if len(groups) > 2 and groups[2]:
                                            # Binary predicate
                                            obj = groups[2]
                                            self.symbolic.prolog.assertz(f"confident_fact({pred}, {subj}, {obj}, {confidence})")
                                        else:
                                            # Unary predicate
                                            self.symbolic.prolog.assertz(f"confident_fact({pred}, {subj}, true, {confidence})")
                                        
                                        print(f"[Bridge] Hypothesis injected: {hypothesis}")
                                    else:
                                        print(f"[Bridge] Failed to parse hypothesis: {hypothesis}")
                                except Exception as e:
                                    print(f"[Bridge] Failed to inject hypothesis: {hypothesis} → {e}")
                        
                        elif task_type == "similarity":
                            # Extract similar concepts
                            if 'similar_concepts' in snn_result:
                                similar_concepts = snn_result['similar_concepts']
                                print(f"[Bridge] Found similar concepts: {similar_concepts}")
                                
                                # Use these concepts to enhance symbolic reasoning
                                for concept, similarity in similar_concepts:
                                    if similarity > 0.6:  # Only use reasonably confident similarities
                                        # Extract subject from input using simple heuristics
                                        subject = None
                                        for word in text.lower().split():
                                            if len(word) > 3 and word.isalpha():
                                                subject = word
                                                break
                                        
                                        if subject:
                                            # Assert similarity relation
                                            self.symbolic.prolog.assertz(
                                                f"confident_fact('similar_to', '{subject}', '{concept}', {similarity})"
                                            )
                                            print(f"[Bridge] Added similarity: {subject} similar_to {concept} ({similarity:.2f})")
                        
                        elif task_type == "generalization":
                            # Extract generalizations
                            if 'generalizations' in snn_result and 'generalized_concepts' in snn_result['generalizations']:
                                generalizations = snn_result['generalizations']['generalized_concepts']
                                print(f"[Bridge] Found generalizations: {generalizations}")
                                
                                # Assert generalizations to knowledge base
                                for concept, confidence in generalizations:
                                    if confidence > 0.5:
                                        self.symbolic.prolog.assertz(
                                            f"confident_fact('generalization_of', '{concept}', 'input', {confidence})"
                                        )
                                        print(f"[Bridge] Added generalization: {concept} ({confidence:.2f})")
                        
                        elif task_type == "analogy":
                            # Process analogy results
                            if 'result' in snn_result:
                                analogy_result = snn_result['result']
                                if 'answer' in analogy_result:
                                    answer = analogy_result['answer']
                                    print(f"[Bridge] Analogy result: {answer}")
                                    
                                    # Add analogy result to semantic memory
                                    if 'input_terms' in analogy_result:
                                        a, b, c = analogy_result['input_terms']
                                        self.semantic.add_relation(
                                            f"{a}_to_{b}", 'analogous_to', f"{c}_to_{answer}", 0.7
                                        )
                        
                        # Re-run reasoning with new information from dual SNN
                        query = self.symbolic.translate_to_symbolic(text)
                        updated_results = self.symbolic.reason_with_uncertainty(query)
                        
                        # Update symbolic results with enhanced information
                        if updated_results.get('certain') or len(updated_results.get('uncertain', [])) > len(symbolic_results.get('uncertain', [])):
                            print(f"[Bridge] Updated reasoning with dual SNN enhanced information")
                            symbolic_results = updated_results
                        
                        # Add dual SNN results to output
                        symbolic_results['snn_results'] = snn_result
                        
                        # Get integration metrics
                        integration_metrics = self.snn.get_integration_metrics()
                        self.integration_level = integration_metrics.get('integration_level', 0.3)
                        
                        # Increase recurrent loops for more complex processing
                        self.recurrent_loops = min(5, self.recurrent_loops + 1)
                    
                    # Prepare result data
                    results['symbolic'] = {
                        'success': True,
                        'data': symbolic_results,
                        'error': None
                    }
                    
                    # Set response field if available in results
                    if symbolic_results.get('response'):
                        results['response'] = symbolic_results['response']
                    elif symbolic_results.get('certain'):
                        # Format response from certain facts
                        facts = symbolic_results['certain']
                        response_parts = []
                        for fact in facts[:3]:  # Limit to top 3 facts
                            msg = f"I know that {fact['subject']} {fact['predicate']} {fact['object']}."
                            response_parts.append(msg)
                        results['response'] = " ".join(response_parts)
                    elif symbolic_results.get('uncertain'):
                        # Format response from uncertain facts
                        facts = symbolic_results['uncertain']
                        if facts:
                            top_fact = facts[0]
                            results['response'] = f"I think that {top_fact['subject']} might {top_fact['predicate']} {top_fact['object']}, but I'm not certain."
                    
                    # Ensure we have a response
                    if not results['response'] and 'snn_results' in symbolic_results:
                        # Try to generate a response from SNN results
                        snn_result = symbolic_results['snn_results']
                        
                        if task_type == 'similarity' and 'similar_concepts' in snn_result:
                            concepts = [c[0] for c in snn_result['similar_concepts'][:3]]
                            if concepts:
                                results['response'] = f"I found similar concepts: {', '.join(concepts)}."
                        
                        elif task_type == 'abduction' and 'enhanced_hypotheses' in snn_result:
                            hypos = [h['hypothesis'] for h in snn_result['enhanced_hypotheses'][:2]]
                            if hypos:
                                results['response'] = f"My hypotheses are: {'; '.join(hypos)}."
                        
                        elif task_type == 'generalization' and 'generalizations' in snn_result:
                            if 'generalized_concepts' in snn_result['generalizations']:
                                gen_concepts = [g[0] for g in snn_result['generalizations']['generalized_concepts'][:3]]
                                if gen_concepts:
                                    results['response'] = f"I can generalize these concepts: {', '.join(gen_concepts)}."
                    
                    # Further integration with memory systems
                    if 'uncertain' in symbolic_results and len(symbolic_results['uncertain']) > 0:
                        # Try to recall related memories to refine uncertain results
                        memories = self.episodic.retrieve_relevant(text)
                        if memories:
                            self.subsystem_activities['memory'] = 0.8
                            print(f"[Bridge] Retrieved {len(memories)} relevant memories")
                            
                            memory_insights = []
                            for memory_text, relevance in memories[:2]:  # Use top 2 memories
                                if relevance > 0.6:  # Only use relevant memories
                                    memory_insights.append(f"I recall: {memory_text}")
                            
                            # Add memory insights to response if we have any
                            if memory_insights and results['response']:
                                results['response'] += f" {memory_insights[0]}"
                
                except Exception as e:
                    # Symbolic processing failed
                    error_msg = f"Symbolic processing error: {str(e)}"
                    print(f"[Bridge] {error_msg}")
                    results['symbolic'] = {
                        'success': False,
                        'data': {},
                        'error': error_msg
                    }
            
            # Calculate integration metrics
            phi = self._calculate_phi()
            
            # Record system state
            results['system_state'] = {
                'integration_level': phi,
                'recurrent_loops': self.recurrent_loops,
                'subsystem_activities': self.subsystem_activities
            }
            
            # STEP 4: Knowledge abstraction
            try:
                # Periodically trigger knowledge abstraction (20% chance)
                if random.random() < 0.2:
                    new_abstractions = self.abstraction.apply_abstractions()
                    if new_abstractions > 0:
                        print(f"[Bridge] Created {new_abstractions} new knowledge abstractions")
                        self.subsystem_activities['reasoning'] += 0.1  # Small boost for abstraction
            except Exception as abstraction_e:
                print(f"[Bridge] Error in knowledge abstraction: {str(abstraction_e)}")
            
            # Ensure we have a response
            if not results.get('response'):
                results['response'] = "I processed your input but couldn't form a specific response."
            
            # Add trace ID if available
            if trace_id:
                self.thought_trace.end_trace(trace_id, results.get('response', 'Processing complete'))
                results['trace_id'] = trace_id
            
            return results
            
        except Exception as e:
            # Top-level error handler
            error_result = self._handle_prolog_error(str(e), text)
            if trace_id:
                self.thought_trace.end_trace(trace_id, f"Error: {str(e)}")
                error_result['trace_id'] = trace_id
            return error_result
    
    def _integrate_results(self, neural_data, symbolic_data):
        """Enhanced integration between neural and symbolic processing results."""
        # Create integrated result structure
        integrated = {
            'source_text': neural_data.get('nlp_result', {}).get('text', ''),
            'entities': neural_data.get('nlp_result', {}).get('entities', []),
            'facts': []
        }
        
        # IMPORTANT: Enhanced bidirectional communication between components
        
        # 1. Feed symbolic results to neural component with more detailed feedback
        if hasattr(self.snn, 'process_symbolic_result') and symbolic_data.get('results'):
            symbolic_facts = []
            if symbolic_data.get('results', {}).get('certain'):
                symbolic_facts.extend(symbolic_data['results']['certain'])
            if symbolic_data.get('results', {}).get('uncertain'):
                symbolic_facts.extend(symbolic_data['results']['uncertain'])
                    
            if symbolic_facts:
                # Try to process symbolic facts with neural network
                try:
                    # More detailed feedback with confidence information
                    enhanced_facts = []
                    for fact in symbolic_facts:
                        enhanced_fact = {
                            'subject': str(fact.get('subject', '')),
                            'predicate': str(fact.get('predicate', '')),
                            'object': str(fact.get('object', '')),
                            'confidence': float(fact.get('confidence', 0.5)),
                            'source': 'symbolic'
                        }
                        enhanced_facts.append(enhanced_fact)
                    
                    # Send more structured feedback
                    self.snn.process_symbolic_result(enhanced_facts)
                    
                    # Update neural subsystem activity with more accurate value
                    self.subsystem_activities['neural'] = max(0.7, self.subsystem_activities.get('neural', 0))
                    
                    # If SNN has a phi value, incorporate it
                    if hasattr(self.snn, 'phi'):
                        snn_phi = self.snn.phi
                        # Use it to influence our overall integration
                        self.integration_level = max(self.integration_level, snn_phi * 0.8)
                except Exception as e:
                    print(f"[Bridge] Error in neural-symbolic integration: {e}")
        
        # 2. Feed neural state back to symbolic system with enhanced state information
        if hasattr(self.snn, 'get_current_activation') and hasattr(self.symbolic, 'receive_neural_feedback'):
            try:
                # Get neural activation pattern
                neural_activation = self.snn.get_current_activation()
                
                # Also get region activations for more detailed feedback
                region_activations = {}
                if hasattr(self.snn, 'regions'):
                    for region_name, region in self.snn.regions.items():
                        region_activations[region_name] = region.get('activation', 0)
                
                # Send more detailed feedback to symbolic system
                self.symbolic.receive_neural_feedback(neural_activation, strength=0.8)
                
                # Update symbolic subsystem activity based on region activations
                # Higher activation for cognitive and reasoning regions
                if region_activations:
                    cognitive_activation = max(
                        region_activations.get('conceptual', 0),
                        region_activations.get('metacognition', 0),
                        region_activations.get('reasoning', 0)
                    )
                    self.subsystem_activities['symbolic'] = max(0.7, cognitive_activation)
                else:
                    self.subsystem_activities['symbolic'] = max(0.7, self.subsystem_activities.get('symbolic', 0))
            except Exception as e:
                print(f"[Bridge] Error in symbolic-neural integration: {e}")
        
        # 3. Vector symbolic integration
        if hasattr(self, 'vector_symbolic'):
            try:
                # Get facts from symbolic system
                facts = []
                if symbolic_data.get('results', {}).get('certain'):
                    facts.extend(symbolic_data['results']['certain'])
                if symbolic_data.get('results', {}).get('uncertain'):
                    facts.extend(symbolic_data['results']['uncertain'])
                
                # Integrate with vector symbolic system
                for fact in facts:
                    subj = fact.get('subject', '')
                    pred = fact.get('predicate', '')
                    obj = fact.get('object', '')
                    conf = fact.get('confidence', 0.8)
                    
                    if subj and pred and obj:
                        self.vector_symbolic.create_fact(subj, pred, obj, conf)
                
                # Update vector symbolic activity
                self.subsystem_activities['vector_symbolic'] = 0.8
            except Exception as e:
                print(f"[Bridge] Error in vector symbolic integration: {e}")
        
        # Extract facts from symbolic results
        if symbolic_data.get('results', {}).get('certain'):
            for fact in symbolic_data['results']['certain']:
                integrated['facts'].append({
                    'type': 'certain',
                    'predicate': fact['predicate'],
                    'subject': fact['subject'],
                    'object': fact['object'],
                    'confidence': fact['confidence']
                })
        
        if symbolic_data.get('results', {}).get('uncertain'):
            for fact in symbolic_data['results']['uncertain']:
                integrated['facts'].append({
                    'type': 'uncertain',
                    'predicate': fact['predicate'],
                    'subject': fact['subject'],
                    'object': fact['object'],
                    'confidence': fact['confidence']
                })
        
        # 4. Calculate new combined integration level based on more factors
        # This computation will increase phi when multiple subsystems are active
        active_subsystems = sum(1 for v in self.subsystem_activities.values() if v > 0.6)
        subsystem_variance = np.var(list(self.subsystem_activities.values()))
        
        # Phi is higher when:
        # 1. Multiple subsystems are active (integration)
        # 2. Their activations vary (differentiation)
        # 3. At least one subsystem is highly active
        max_activation = max(self.subsystem_activities.values())
        avg_activation = sum(self.subsystem_activities.values()) / len(self.subsystem_activities)
        
        # Formula that balances these factors
        self.integration_level = (
            (active_subsystems / len(self.subsystem_activities)) *  # Integration term
            (0.5 + 0.5 * subsystem_variance) *                      # Differentiation term
            (0.5 * avg_activation + 0.5 * max_activation)           # Activation term
        )
        
        # Ensure a minimum phi value
        self.integration_level = max(0.3, self.integration_level)
        
        # Recalculate phi after integration
        phi = self._calculate_phi()
        
        # Add system metrics
        integrated['metrics'] = {
            'phi': self.integration_level or phi,
            'loops': self.recurrent_loops,
            'subsystems': {k: round(v, 2) for k, v in self.subsystem_activities.items()}
        }

        return integrated
    
    def _calculate_information_transfer(self, neural_data, symbolic_data):
        """
        Calculate the amount of information transferred between systems.
        
        Args:
            neural_data: Results from neural processing
            symbolic_data: Results from symbolic processing
            
        Returns:
            dict: Metrics of information transfer
        """
        metrics = {
            'neural_to_symbolic': 0.0,
            'symbolic_to_neural': 0.0,
            'neural_to_vector': 0.0,
            'symbolic_to_vector': 0.0,
            'total_transfer': 0.0
        }
        
        # Extract neural information
        neural_facts = []
        if isinstance(neural_data, dict) and 'triples' in neural_data:
            neural_facts = neural_data['triples']
        
        # Extract symbolic information
        symbolic_facts = []
        if (isinstance(symbolic_data, dict) and 'results' in symbolic_data 
                and isinstance(symbolic_data['results'], dict)):
            if 'certain' in symbolic_data['results']:
                symbolic_facts.extend(symbolic_data['results']['certain'])
            if 'uncertain' in symbolic_data['results']:
                symbolic_facts.extend(symbolic_data['results']['uncertain'])
        
        # Calculate neural -> symbolic transfer
        symbolic_count = len(symbolic_facts)
        if neural_facts and symbolic_count > 0:
            # Estimate how many facts came from neural processing
            overlap = 0
            for neural_fact in neural_facts:
                for symbolic_fact in symbolic_facts:
                    if (isinstance(neural_fact, tuple) and len(neural_fact) >= 3 and
                        symbolic_fact.get('subject', '').lower() == neural_fact[0].lower() and
                        symbolic_fact.get('predicate', '').lower() == neural_fact[1].lower() and
                        symbolic_fact.get('object', '').lower() == neural_fact[2].lower()):
                        overlap += 1
                        break
            
            if len(neural_facts) > 0:
                metrics['neural_to_symbolic'] = overlap / len(neural_facts)
        
        # Calculate symbolic -> neural transfer
        if hasattr(self.snn, 'active_neurons_cache'):
            # More active neurons indicate more information transferred
            active_count = len(self.snn.active_neurons_cache)
            total_neurons = getattr(self.snn, 'neuron_count', 1000)
            activity_ratio = active_count / total_neurons
            
            # Higher activation with symbolic facts indicates transfer
            if symbolic_count > 0 and activity_ratio > 0.1:
                metrics['symbolic_to_neural'] = min(0.9, activity_ratio * symbolic_count * 0.1)
        
        # Calculate overall transfer
        metrics['total_transfer'] = (metrics['neural_to_symbolic'] + 
                                    metrics['symbolic_to_neural'] + 
                                    metrics['neural_to_vector'] + 
                                    metrics['symbolic_to_vector']) / 4
        
        return metrics
            
    def _inject_hypothesis(self, hypo, confidence=0.4):
        """Inject a hypothesis into the symbolic KB with specified confidence"""
        try:
            # Extract predicate and arguments
            match = re.match(r"(\w+)\((\w+)(?:,\s*(\w+))?\)", hypo)
            if match:
                groups = match.groups()
                pred = groups[0]
                subj = groups[1]
                
                if len(groups) > 2 and groups[2]:
                    # Binary predicate
                    obj = groups[2]
                    self.symbolic.prolog.assertz(f"confident_fact({pred}, {subj}, {obj}, {confidence})")
                    # Also add to semantic memory
                    self.semantic.add_relation(subj, pred, obj, confidence)
                else:
                    # Unary predicate
                    self.symbolic.prolog.assertz(f"confident_fact({pred}, {subj}, true, {confidence})")
                    # Add to semantic memory
                    self.semantic.add_relation(subj, pred, "true", confidence)
                    
                print(f"[Bridge] Hypothesis injected: {hypo} (conf={confidence})")
            else:
                print(f"[Bridge] Failed to parse hypothesis: {hypo}")
                    
        except Exception as e:
            print(f"[Bridge] Failed to inject hypothesis: {hypo} → {e}")
    
    def handle_conversation(self, text):
        """Handle basic conversational inputs"""
        text_lower = text.lower()
        
        if any(greeting in text_lower for greeting in ["hello", "hi", "hey", "greetings"]):
            return {
                "response": "Hello! How can I help you today?",
                "success": True
            }
            
        if "how are you" in text_lower:
            # Include some system introspection if integration level is high
            if self.integration_level > 0.5:
                phi_str = f"{self.integration_level:.2f}"
                return {
                    "response": f"I'm functioning well with an integration level of Φ={phi_str}, thank you for asking. How can I assist you today?",
                    "success": True
                }
            else:
                return {
                    "response": "I'm functioning well, thank you for asking. How can I assist you?",
                    "success": True
                }
            
        if any(farewell in text_lower for farewell in ["goodbye", "bye", "see you"]):
            return {
                "response": "Goodbye! Feel free to chat again later.",
                "success": True
            }
            
        if "thank" in text_lower:
            return {
                "response": "You're welcome! Is there anything else you'd like to know?",
                "success": True
            }
        
        # Check for questions about the system's state/consciousness
        if any(term in text_lower for term in ["consciousness", "aware", "phi", "integration level", "self aware"]):
            phi_str = f"{self.integration_level:.2f}"
            active_systems = [name for name, activity in self.subsystem_activities.items() if activity > 0.5]
            active_str = ", ".join(active_systems) if active_systems else "minimal activity across subsystems"
            
            response = (
                f"My current integration level (Φ) is {phi_str}, with {active_str}. "
                f"I've processed through {self.recurrent_loops} recurrent loops in this interaction. "
                f"According to Integrated Information Theory, a higher Φ value suggests more consciousness-like properties."
            )
            
            return {
                "response": response,
                "success": True
            }
        
        # If no pattern matches, try the symbolic reasoning
        return None  # Let the main processing handle it
    
    def get_current_context(self):
        """Get current conversation context"""
        return {
            'topic': self.context.current_topic,
            'entities': list(self.context.current_entities),
            'history_length': len(self.context.conversation_history)
        }
        
    def extract_kg_triples(self):
        """Extract knowledge graph triples for visualization"""
        triples = []
        try:
            # Query all facts from Prolog
            for result in self.symbolic.prolog.query("confident_fact(P, S, O, C)"):
                triple = {
                    'subject': str(result['S']),
                    'predicate': str(result['P']),
                    'object': str(result['O']),
                    'confidence': float(result['C'])
                }
                triples.append(triple)
        except Exception as e:
            print(f"[Bridge] Error extracting KG triples: {e}")
            
        return triples
    
    def get_integration_metrics(self):
        """Return metrics about the system's integration level and subsystem activities"""
        return {
            'phi': self.integration_level,
            'recurrent_loops': self.recurrent_loops,
            'subsystem_activities': self.subsystem_activities,
            'integration_history': self.integration_history[-10:] if self.integration_history else [],
            'global_workspace': {
                'current_focus': self.global_workspace['current_focus'],
                'broadcast_strength': self.global_workspace['broadcast_strength'],
                'active_elements_count': len(self.global_workspace['active_elements'])
            }
        }
    
    def ensure_minimal_metrics(self):
        """Ensure that integration metrics have at least minimal values for training"""
        # Force minimum integration level for training if it's zero
        if self.integration_level == 0:
            # Calculate a minimal phi value
            if any(activity > 0 for activity in self.subsystem_activities.values()):
                # Recalculate phi from subsystem activities
                self._calculate_phi()
            else:
                # Ensure at least minimal activity for learning
                self.subsystem_activities = {
                    'symbolic': 0.3,
                    'neural': 0.3,
                    'abductive': 0.1,
                    'memory': 0.2,
                    'reasoning': 0.3,
                    'vector_symbolic': 0.2
                }
                self.integration_level = 0.1
        
        # Ensure recurrent_loops has at least value 1
        if self.recurrent_loops == 0:
            self.recurrent_loops = 1

    def ensure_phi_calculation(self):
        """Force Phi calculation even if not triggered during processing."""
        # Set minimum values for subsystem activities if they're zero
        for key in self.subsystem_activities:
            if self.subsystem_activities[key] == 0:
                self.subsystem_activities[key] = 0.3  # Minimal activity
        
        # Force a recalculation of Phi
        phi = self._calculate_phi()
        
        # Ensure at least one recurrent loop is counted
        if self.recurrent_loops == 0:
            self.recurrent_loops = 1
        
        return phi