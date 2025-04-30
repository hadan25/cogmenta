# cogmenta_core/models/hybrid/temporal_bridge.py
import logging
from datetime import datetime, timedelta
from models.symbolic.prolog_engine import PrologEngine
from models.symbolic.temporal_reasoning import TemporalReasoner
from models.hybrid.enhanced_neuro_symbolic_bridge import EnhancedNeuroSymbolicBridge

class TemporalBridge:
    """
    Bridge class that integrates temporal reasoning with the core neuro-symbolic architecture.
    Handles time-based knowledge representation and reasoning.
    """
    
    def __init__(self, neuro_symbolic_bridge=None, prolog_engine=None):
        """
        Initialize the temporal bridge.
        
        Args:
            neuro_symbolic_bridge: Existing EnhancedNeuroSymbolicBridge (optional)
            prolog_engine: Existing PrologEngine (optional)
        """
        self.logger = logging.getLogger(__name__)
        
        # Use provided components or create new ones
        self.bridge = neuro_symbolic_bridge or EnhancedNeuroSymbolicBridge()
        self.prolog = prolog_engine or self.bridge.symbolic
        
        # Track initialization failures
        self.initialization_success = True
        
        # Initialize the temporal reasoner
        try:
            self.temporal = TemporalReasoner(prolog_engine=self.prolog)
            
            # Integrate with the Prolog engine
            success = self.temporal.integrate_with_prolog_engine(self.prolog)
            if not success:
                self.logger.warning("Integration with Prolog engine was not fully successful.")
                self.initialization_success = False
        except Exception as e:
            self.logger.error(f"Failed to initialize temporal reasoner: {str(e)}")
            self.temporal = None  # Set to None so we know it failed
            self.initialization_success = False
            
        # Track temporal context
        self.temporal_context = {
            'reference_time': datetime.now(),
            'time_focus': 'present',  # past, present, future
            'temporal_scope': 'point'  # point, interval, indefinite
        }
        
        if self.initialization_success:
            self.logger.info("Temporal Bridge initialized and integrated with neuro-symbolic architecture")
        else:
            self.logger.warning("Temporal Bridge initialized with limited functionality due to initialization errors")
            
        def update_reference_time(self, time_str=None):
            """
            Update the reference time for temporal reasoning.
            
            Args:
                time_str (str): ISO format time string (optional)
                
            Returns:
                datetime: Updated reference time
            """
            if time_str:
                try:
                    self.temporal_context['reference_time'] = datetime.fromisoformat(time_str)
                except ValueError:
                    self.logger.warning(f"Invalid time format: {time_str}. Using current time.")
                    self.temporal_context['reference_time'] = datetime.now()
            else:
                self.temporal_context['reference_time'] = datetime.now()
                
            return self.temporal_context['reference_time']
        
    def extract_temporal_information(self, text):
        """
        Extract time-related information from text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Extracted temporal information
        """
        # Extract time expressions
        time_expressions = self.temporal.extract_time_expressions(text)
        
        # Extract temporal relationships
        temporal_triples = self.temporal.extract_temporal_triples(text)
        
        # Determine temporal focus based on extracted information
        self._update_temporal_focus(text, time_expressions)
        
        return {
            'time_expressions': time_expressions,
            'temporal_triples': temporal_triples,
            'temporal_context': self.temporal_context.copy()
        }
    
    def _update_temporal_focus(self, text, time_expressions):
        """
        Update the temporal focus based on the input text.
        
        Args:
            text (str): Input text
            time_expressions (list): Extracted time expressions
        """
        text_lower = text.lower()
        
        # Check for explicit past/present/future indicators
        if any(word in text_lower for word in ['previously', 'before', 'used to', 'in the past', 'earlier']):
            self.temporal_context['time_focus'] = 'past'
        elif any(word in text_lower for word in ['currently', 'now', 'at present', 'at the moment']):
            self.temporal_context['time_focus'] = 'present'
        elif any(word in text_lower for word in ['will', 'going to', 'in the future', 'later']):
            self.temporal_context['time_focus'] = 'future'
        
        # Update temporal scope based on expressions
        has_interval = False
        for expr in time_expressions:
            if expr['type'] in ['interval', 'relative_period', 'duration']:
                has_interval = True
                break
                
        self.temporal_context['temporal_scope'] = 'interval' if has_interval else 'point'
    
    def process_text_with_temporal_reasoning(self, text):
        """
        Process text with integrated temporal reasoning.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Results including temporal reasoning
        """
        # Check if initialization was successful
        if not getattr(self, 'initialization_success', False) or self.temporal is None:
            self.logger.warning("Temporal reasoning not fully initialized. Falling back to standard processing.")
            # Fall back to standard processing
            bridge_results = self.bridge.process_text_and_reason(text)
            
            # Add a note about temporal reasoning failure
            if 'response' in bridge_results:
                bridge_results['response'] += "\n\n(Note: I couldn't apply temporal reasoning to your query due to initialization issues.)"
                
            # Add empty temporal data
            bridge_results['temporal'] = {
                'success': False,
                'data': None,
                'error': "Temporal reasoning not initialized"
            }
            
            return bridge_results
            
        # Extract temporal information
        try:
            temporal_info = self.extract_temporal_information(text)
        except Exception as e:
            self.logger.error(f"Error extracting temporal information: {str(e)}")
            temporal_info = {
                'time_expressions': [],
                'temporal_triples': [],
                'temporal_context': self.temporal_context.copy()
            }
        
        # Process the text through the core neuro-symbolic bridge
        bridge_results = self.bridge.process_text_and_reason(text)
        
        # If the bridge processing succeeded
        if bridge_results.get('symbolic', {}).get('success', False):
            # Try to extract triples for temporal enrichment
            try:
                triples = bridge_results.get('neural', {}).get('data', {}).get('triples', [])
                
                # Add temporal information to the triples
                temporal_triples = self._add_temporal_context_to_triples(triples, temporal_info)
                
                # Add the temporal triples to the knowledge base
                self._assert_temporal_triples(temporal_triples)
                
                # Add temporal queries to the results if appropriate
                if self._should_perform_temporal_reasoning(text):
                    # Perform temporal reasoning
                    try:
                        temporal_results = self._perform_temporal_reasoning(text, temporal_info)
                        
                        # Add temporal results to the output
                        bridge_results['temporal'] = {
                            'success': True,
                            'data': {
                                'temporal_info': temporal_info,
                                'temporal_results': temporal_results
                            },
                            'error': None
                        }
                        
                        # Enhance the response with temporal information if available
                        if temporal_results and 'response' in temporal_results:
                            # If original response exists, combine them
                            if bridge_results.get('response'):
                                bridge_results['response'] += f"\n\nTemporal context: {temporal_results['response']}"
                            else:
                                bridge_results['response'] = temporal_results['response']
                    except Exception as e:
                        self.logger.error(f"Error in temporal reasoning: {str(e)}")
                        bridge_results['temporal'] = {
                            'success': False,
                            'data': {'temporal_info': temporal_info},
                            'error': f"Error in temporal reasoning: {str(e)}"
                        }
                else:
                    # Just add the temporal information without reasoning
                    bridge_results['temporal'] = {
                        'success': True,
                        'data': {'temporal_info': temporal_info},
                        'error': None
                    }
            except Exception as e:
                self.logger.error(f"Error processing temporal triples: {str(e)}")
                bridge_results['temporal'] = {
                    'success': False,
                    'data': {'temporal_info': temporal_info},
                    'error': f"Error processing temporal triples: {str(e)}"
                }
        else:
            # Bridge processing failed, but we can still provide temporal information
            bridge_results['temporal'] = {
                'success': False,
                'data': {'temporal_info': temporal_info},
                'error': "Main bridge processing failed, could not perform temporal reasoning."
            }
            
        return bridge_results
    
    def _add_temporal_context_to_triples(self, triples, temporal_info):
        """
        Add temporal context to the extracted triples.
        
        Args:
            triples (list): Extracted triples
            temporal_info (dict): Extracted temporal information
            
        Returns:
            list: Triples with temporal context
        """
        temporal_triples = []
        
        # Get reference time
        reference_time = self.temporal_context['reference_time']
        
        # Get extracted time expressions
        time_expressions = temporal_info.get('time_expressions', [])
        
        # If we have specific time expressions, use them
        specific_times = []
        for expr in time_expressions:
            norm_time = self.temporal.normalize_time_expression(expr['text'], reference_time)
            if norm_time:
                specific_times.append(norm_time)
        
        # If no specific times, use the temporal focus
        if not specific_times:
            # Default time is current reference time
            if self.temporal_context['time_focus'] == 'present':
                time_point = reference_time.isoformat()
                end_point = None
            elif self.temporal_context['time_focus'] == 'past':
                # For past, use a generic "one year ago to now" interval
                time_point = (reference_time - timedelta(days=365)).isoformat()
                end_point = reference_time.isoformat()
            elif self.temporal_context['time_focus'] == 'future':
                # For future, use a generic "now to one year ahead" interval
                time_point = reference_time.isoformat()
                end_point = (reference_time + timedelta(days=365)).isoformat()
        else:
            # Use the first specific time
            if specific_times[0]['type'] == 'point':
                time_point = specific_times[0]['iso_format']
                end_point = None
            else:
                time_point = specific_times[0]['iso_format_start']
                end_point = specific_times[0]['iso_format_end']
        
        # Add temporal context to each triple
        for triple in triples:
            if len(triple) >= 4:
                subj, pred, obj, conf = triple
                
                temporal_triple = {
                    'subject': subj,
                    'predicate': pred,
                    'object': obj,
                    'confidence': conf,
                    'start_time': time_point,
                    'end_time': end_point
                }
                
                temporal_triples.append(temporal_triple)
        
        return temporal_triples
    
    def _assert_temporal_triples(self, temporal_triples):
        """
        Assert temporal triples to the knowledge base.
        
        Args:
            temporal_triples (list): Triples with temporal context
        """
        for triple in temporal_triples:
            self.temporal.add_time_dependent_facts(
                triple['predicate'],
                triple['subject'],
                triple['object'],
                triple['start_time'],
                triple['end_time'],
                triple['confidence']
            )
            
            self.logger.info(f"Added temporal fact: {triple['predicate']}({triple['subject']}, {triple['object']}) from {triple['start_time']} to {triple['end_time'] or 'indefinite'}")
    
    def _should_perform_temporal_reasoning(self, text):
        """
        Determine if we should perform temporal reasoning on this text.
        
        Args:
            text (str): Input text
            
        Returns:
            bool: True if temporal reasoning should be performed
        """
        text_lower = text.lower()
        
        # Check for temporal query indicators
        temporal_indicators = [
            'when', 'before', 'after', 'during', 'until', 'since',
            'past', 'previously', 'used to',
            'already', 'yet', 'still',
            'at that time', 'at the time',
            'in the future', 'will', 'going to',
            'now', 'current', 'present',
            'always', 'never', 'ever'
        ]
        
        # Check if any temporal indicator is present
        return any(indicator in text_lower for indicator in temporal_indicators)
    
    def _perform_temporal_reasoning(self, text, temporal_info):
        """
        Perform temporal reasoning on the text.
        
        Args:
            text (str): Input text
            temporal_info (dict): Extracted temporal information
            
        Returns:
            dict: Temporal reasoning results
        """
        # Get the reference time point for queries
        reference_time = self.temporal_context['reference_time'].isoformat()
        
        # Extract any time points from the text
        time_expressions = temporal_info.get('time_expressions', [])
        time_points = []
        
        for expr in time_expressions:
            norm_time = self.temporal.normalize_time_expression(expr['text'], self.temporal_context['reference_time'])
            if norm_time:
                if norm_time['type'] == 'point':
                    time_points.append(norm_time['iso_format'])
                else:
                    time_points.append(norm_time['iso_format_start'])
                    time_points.append(norm_time['iso_format_end'])
        
        # If no specific time points found, use the reference time
        if not time_points:
            time_points = [reference_time]
        
        # Check if this is a question about when something happened
        when_match = re.search(r'when\s+did\s+(\w+)\s+(trust|like|love|know|fear|hate|avoid)\s+(\w+)', text.lower())
        
        if when_match:
            # This is a "when" question
            subj, pred, obj = when_match.groups()
            
            # Query across multiple time points to find when this was true
            temporal_results = []
            
            for time_point in time_points:
                results = self.temporal.query_at_time_point(pred, subj, obj, time_point)
                temporal_results.extend(results)
            
            # Also try to reason about changes over time
            changes = self.temporal.reason_about_changes(subj, pred, time_points[0], time_points[-1])
            
            # Format response
            if temporal_results:
                # Found specific time points when the relation was true
                response = f"According to my information, {subj} {pred} {obj} "
                
                if len(temporal_results) == 1:
                    result = temporal_results[0]
                    time_str = self._format_time_for_display(result['time_point'])
                    response += f"at {time_str}."
                else:
                    response += "during these times: "
                    time_strs = [self._format_time_for_display(r['time_point']) for r in temporal_results]
                    response += ", ".join(time_strs) + "."
                    
                if changes.get('analysis') and changes['analysis'] != "Relation is stable over time":
                    response += f" {changes['analysis']}."
            else:
                # No specific times found
                response = f"I don't have information about when {subj} {pred} {obj}."
                
            return {
                'response': response,
                'results': temporal_results,
                'changes': changes.get('changes', [])
            }
            
        # Check if this is a question about whether something was true at a specific time
        was_true_match = re.search(r'(did|was)\s+(\w+)\s+(trust|like|love|know|fear|hate|avoid)\s+(\w+)\s+(at|in|during|before|after)\s+(.*?)[\?\.]', text.lower())
        
        if was_true_match:
            # This is a question about truth at a specific time
            _, subj, pred, obj, temp_rel, temp_expr = was_true_match.groups()
            
            # Normalize the time expression
            norm_time = self.temporal.normalize_time_expression(temp_expr, self.temporal_context['reference_time'])
            
            if norm_time:
                if norm_time['type'] == 'point':
                    query_time = norm_time['iso_format']
                else:
                    # For intervals, use the start time for 'before', end time for 'after', 
                    # or middle for other cases
                    if temp_rel == 'before':
                        query_time = norm_time['iso_format_start']
                    elif temp_rel == 'after':
                        query_time = norm_time['iso_format_end']
                    else:
                        # Use middle of interval for 'during', 'at', etc.
                        start = datetime.fromisoformat(norm_time['iso_format_start'])
                        end = datetime.fromisoformat(norm_time['iso_format_end'])
                        middle = start + (end - start) / 2
                        query_time = middle.isoformat()
            else:
                # If we couldn't normalize, use the reference time
                query_time = reference_time
                
            # Query whether the relation was true at this time
            results = self.temporal.query_at_time_point(pred, subj, obj, query_time)
            
            # Format response
            if results:
                result = results[0]
                time_str = self._format_time_for_display(result['time_point'])
                conf_str = "definitely" if result['confidence'] > 0.8 else "probably"
                response = f"Yes, {subj} {conf_str} {pred} {obj} {temp_rel} {temp_expr}."
            else:
                response = f"I don't have information indicating that {subj} {pred} {obj} {temp_rel} {temp_expr}."
                
            return {
                'response': response,
                'results': results,
                'query_time': query_time
            }
            
        # For other types of text, just perform general temporal analysis
        # Try to extract any relationships mentioned and check their temporal validity
        temporal_triples = temporal_info.get('temporal_triples', [])
        
        if temporal_triples:
            # For each triple, check when it was true
            results = []
            
            for triple in temporal_triples:
                for time_point in time_points:
                    point_results = self.temporal.query_at_time_point(
                        triple['predicate'],
                        triple['subject'],
                        triple['object'],
                        time_point
                    )
                    results.extend(point_results)
            
            # Format response
            if results:
                response = "Based on temporal analysis, I found these time-dependent relationships:\n"
                for result in results:
                    time_str = self._format_time_for_display(result['time_point'])
                    response += f"- {result['subject']} {result['predicate']} {result['object']} at {time_str}\n"
            else:
                response = "I performed temporal analysis but didn't find any time-dependent relationships."
                
            return {
                'response': response,
                'results': results
            }
        
        # Default case: no specific temporal reasoning performed
        return {
            'response': "I analyzed the temporal aspects but didn't find specific time-dependent information.",
            'results': []
        }
    
    def _format_time_for_display(self, iso_time):
        """
        Format an ISO time string for human-readable display.
        
        Args:
            iso_time (str): ISO format time string
            
        Returns:
            str: Human-readable time string
        """
        try:
            dt = datetime.fromisoformat(iso_time)
            
            # If the time has non-zero hours/minutes/seconds, include them
            if dt.hour != 0 or dt.minute != 0 or dt.second != 0:
                return dt.strftime("%B %d, %Y at %I:%M %p")
            else:
                # Otherwise just show the date
                return dt.strftime("%B %d, %Y")
        except ValueError:
            # If we can't parse it, return as is
            return iso_time
    
    def ask_temporal_question(self, question, subject=None, predicate=None, object_value=None, time_point=None):
        """
        Ask a specific temporal question about a relationship.
        
        Args:
            question (str): Question type ('when', 'was_true', 'duration')
            subject (str): Subject entity (optional)
            predicate (str): Predicate relation (optional)
            object_value (str): Object entity (optional)
            time_point (str): Time point in ISO format (optional)
            
        Returns:
            dict: Question results
        """
        if not any([subject, predicate, object_value]) and not question == 'list_all':
            return {
                'success': False,
                'error': "Must provide at least one of subject, predicate, or object for the query."
            }
            
        try:
            # If no time point provided, use reference time
            if not time_point:
                time_point = self.temporal_context['reference_time'].isoformat()
                
            if question == 'when':
                # When was this relationship true?
                results = []
                
                # Use an object wildcard if not specified
                obj_val = object_value or '_'
                
                # Query for all time points when this was true
                query = f"fact_true_at('{predicate}', '{subject}', '{obj_val}', T, C), C >= 0.5"
                
                for result in self.prolog.prolog.query(query):
                    if 'T' in result and 'C' in result:
                        results.append({
                            'subject': subject,
                            'predicate': predicate,
                            'object': result.get('O', obj_val),
                            'time_point': str(result['T']),
                            'confidence': float(result['C'])
                        })
                        
                return {
                    'success': True,
                    'question': question,
                    'results': results
                }
                
            elif question == 'was_true':
                # Was this relationship true at the given time point?
                results = self.temporal.query_at_time_point(
                    predicate, subject, object_value or '_', time_point
                )
                
                return {
                    'success': True,
                    'question': question,
                    'time_point': time_point,
                    'results': results
                }
                
            elif question == 'duration':
                # How long was this relationship true?
                changes = self.temporal.reason_about_changes(subject, predicate, time_point, None)
                
                return {
                    'success': True,
                    'question': question,
                    'changes': changes
                }
                
            elif question == 'list_all':
                # List all temporal facts
                all_facts = []
                
                query = "temporal_fact(P, S, O, Start, End, C)"
                
                for result in self.prolog.prolog.query(query):
                    all_facts.append({
                        'predicate': str(result['P']),
                        'subject': str(result['S']),
                        'object': str(result['O']),
                        'start_time': str(result['Start']),
                        'end_time': str(result['End']),
                        'confidence': float(result['C'])
                    })
                    
                return {
                    'success': True,
                    'question': question,
                    'facts': all_facts
                }
                
            else:
                return {
                    'success': False,
                    'error': f"Unknown question type: {question}"
                }
                
        except Exception as e:
            self.logger.error(f"Error in temporal question: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_integration_metrics(self):
        """
        Get metrics about the temporal integration.
        
        Returns:
            dict: Integration metrics
        """
        # Track temporal facts
        total_facts = 0
        
        try:
            query = "temporal_fact(P, S, O, Start, End, C)"
            results = list(self.prolog.prolog.query(query))
            total_facts = len(results)
        except Exception:
            pass
            
        # Get reference time
        ref_time = self.temporal_context['reference_time']
        
        return {
            'temporal_facts': total_facts,
            'reference_time': ref_time.isoformat(),
            'time_focus': self.temporal_context['time_focus'],
            'temporal_scope': self.temporal_context['temporal_scope'],
            'bridge_integration': self.bridge.get_integration_metrics() if hasattr(self.bridge, 'get_integration_metrics') else {}
        }