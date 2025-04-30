import re
from models.symbolic.prolog_engine import PrologEngine
from models.snn.spiking_core import SpikingCore

class NeuroSymbolicBridge:
    def __init__(self):  # Corrected from 'init' to '__init__'
        self.symbolic = PrologEngine()
        self.snn = SpikingCore()
        self.thought_trace = None  # Add thought trace field

    def _handle_prolog_error(self, error_msg, input_text):
        """Handle cases where Prolog processing fails"""
        print(f"[Bridge] Handling Prolog error: {error_msg}")
        
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
        """Process text input and perform neural-symbolic reasoning"""
        trace_id = None
        if hasattr(self, 'thought_trace') and self.thought_trace:
            trace_id = self.thought_trace.start_trace(
                trigger=f"Process: '{text}'",
                source_component="NeuroSymbolicBridge"
            )

        print(f"\n[Bridge] Processing input: '{text}'")
        
        # Check for conversation first
        conversation_response = self.handle_conversation(text)
        if conversation_response:
            if trace_id:
                self.thought_trace.end_trace(trace_id, "Conversation handled")
                conversation_response['trace_id'] = trace_id
            return conversation_response

        try:
        # Extract triples using pattern matching (neural extraction in real system)
            triples = self.symbolic.extract_triples_neural(text)

            if triples:
                print(f"[Bridge] Extracted {len(triples)} triples:")
                for t in triples:
                    print(f"  - {t}")

                # Assert triples to Prolog knowledge base
                self.symbolic.assert_neural_triples(triples)
            else:
                print("[Bridge] No triples extracted using patterns")

            # Try symbolic reasoning first
            results = self.symbolic.neural_symbolic_inference(text)

            # If no confident results, trigger abductive fallback
            if not results.get("certain"):
                print("[Bridge] No confident symbolic answer. Triggering abductive reasoning.")

                # Use spiking neural network for abductive reasoning
                hypotheses = self.snn.abductive_reasoning(text)

                print(f"[Bridge] Generated {len(hypotheses)} hypotheses:")
                for i, hypo in enumerate(hypotheses):
                    print(f"  {i+1}. {hypo}")

                # Assert hypotheses into Prolog with low confidence
                for hypo in hypotheses:
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
                                self.symbolic.prolog.assertz(f"confident_fact({pred}, {subj}, {obj}, 0.4)")
                            else:
                                # Unary predicate
                                self.symbolic.prolog.assertz(f"confident_fact({pred}, {subj}, true, 0.4)")

                            print(f"[Bridge] Hypothesis injected: {hypo}")
                        else:
                            print(f"[Bridge] Failed to parse hypothesis: {hypo}")

                    except Exception as e:
                        print(f"[Bridge] Failed to inject hypothesis: {hypo} → {e}")

                # Re-run reasoning with new hypotheses in place
                query = self.symbolic.translate_to_symbolic(text)
                results = self.symbolic.reason_with_uncertainty(query)

            if trace_id:
                self.thought_trace.end_trace(trace_id, results.get('response', 'Processing complete'))
                results['trace_id'] = trace_id

            return results
        
        except Exception as e:
            error_result = self._handle_prolog_error(str(e), text)
            if trace_id:
                self.thought_trace.end_trace(trace_id, f"Error: {str(e)}")
                error_result['trace_id'] = trace_id
            return error_result

        return results
    
        # basic convos
    def handle_conversation(self, text):
        """Handle basic conversational inputs"""
        text_lower = text.lower()
        
        # Simple pattern matching for conversation
        if any(greeting in text_lower for greeting in ["hello", "hi", "hey", "greetings"]):
            return {"response": "Hello! How can I help you today?"}
            
        if "how are you" in text_lower:
            return {"response": "I'm functioning well, thank you for asking. How can I assist you?"}
            
        if any(farewell in text_lower for farewell in ["goodbye", "bye", "see you"]):
            return {"response": "Goodbye! Feel free to chat again later."}
            
        if "thank" in text_lower:
            return {"response": "You're welcome! Is there anything else you'd like to know?"}
        
        # If no pattern matches, try the symbolic reasoning
        return None  # Let the main processing handle it