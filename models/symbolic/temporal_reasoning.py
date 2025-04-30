# cogmenta_core/models/symbolic/temporal_reasoning.py
import re
import logging
from datetime import datetime, timedelta
try:
    from pyswip import Prolog, Variable, Functor, Query, Term
except ImportError:
    # Provide mock classes to avoid syntax errors if PySwip is not installed
    class Prolog: pass
    class Variable: pass
    class Functor: pass
    class Query: pass
    class Term: pass

class TemporalReasoner:
    """
    Temporal reasoning component for handling time-based relations and reasoning.
    Integrates with the core Prolog engine to provide temporal logic capabilities.
    """
    def __init__(self, prolog_engine=None):
        """
        Initialize the temporal reasoner.
        
        Args:
            prolog_engine: Existing PrologEngine instance (optional)
        """
        self.logger = logging.getLogger(__name__)
        
        # Use provided Prolog engine or create a new one
        if prolog_engine:
            self.prolog = prolog_engine.prolog 
        else:
            try:
                from pyswip import Prolog
                self.prolog = Prolog()
            except ImportError:
                self.logger.error("PySwip not installed. Cannot initialize temporal reasoning.")
                raise ImportError("PySwip is required for temporal reasoning")
        
        # Add datetime parsing functionality
        success = self._add_datetime_parsing_rules()
        
        if success:
            # Initialize the temporal reasoning framework
            self._initialize_temporal_framework()
        else:
            self.logger.warning("Temporal reasoning may have limited functionality due to initialization errors.")
            

    def _initialize_temporal_framework(self):
        """Initialize the temporal logic framework in Prolog."""
        try:
            # Instead of trying to assert the entire block at once,
            # assert each rule individually
            
            # Basic temporal rules
            self.prolog.assertz("after(X, Y) :- before(Y, X)")
            self.prolog.assertz("contains(X, Y) :- during(Y, X)")
            
            # Transitivity rules
            self.prolog.assertz("before(X, Z) :- before(X, Y), before(Y, Z)")
            self.prolog.assertz("during(X, Z) :- during(X, Y), during(Y, Z)")
            
            # Time point comparison
            self.prolog.assertz("time_point_before(TimePoint1, TimePoint2) :- TimePoint1 < TimePoint2")
            
            # Datetime parsing
            self.prolog.assertz("""
                parse_datetime(DateTimeStr, Timestamp) :-
                    atom_chars(DateTimeStr, Chars),
                    parse_datetime_chars(Chars, Year, Month, Day, Hour, Min, Sec),
                    Date is Year*10000 + Month*100 + Day,
                    Time is Hour*10000 + Min*100 + Sec,
                    Timestamp is Date*1000000 + Time
            """)
            
            # Simple placeholder implementation for parse_datetime_chars
            self.prolog.assertz("""
                parse_datetime_chars('2023-01-01T00:00:00', 2023, 1, 1, 0, 0, 0)
            """)
            
            # Temporal event definitions
            self.prolog.assertz("""
                event_time_point(Event, Type, TimePoint) :-
                    event(Event, Type, TimePointStr),
                    parse_datetime(TimePointStr, TimePoint)
            """)
            
            self.prolog.assertz("""
                event_interval(Event, Type, StartPoint, EndPoint) :-
                    event_start(Event, Type, StartPointStr),
                    event_end(Event, Type, EndPointStr),
                    parse_datetime(StartPointStr, StartPoint),
                    parse_datetime(EndPointStr, EndPoint)
            """)
            
            # Event relationships
            self.prolog.assertz("""
                event_before(EventA, EventB) :-
                    event_time_point(EventA, _, TimeA),
                    event_time_point(EventB, _, TimeB),
                    time_point_before(TimeA, TimeB)
            """)
            
            self.prolog.assertz("""
                event_during(EventA, IntervalB) :-
                    event_time_point(EventA, _, TimeA),
                    event_interval(IntervalB, _, StartB, EndB),
                    StartB =< TimeA,
                    TimeA =< EndB
            """)
            
            self.prolog.assertz("""
                interval_before(IntervalA, IntervalB) :-
                    event_interval(IntervalA, _, _, EndA),
                    event_interval(IntervalB, _, StartB, _),
                    time_point_before(EndA, StartB)
            """)
            
            self.prolog.assertz("""
                interval_overlaps(IntervalA, IntervalB) :-
                    event_interval(IntervalA, _, StartA, EndA),
                    event_interval(IntervalB, _, StartB, EndB),
                    StartA =< EndB,
                    StartB =< EndA
            """)
            
            # Temporal facts and fact retrieval
            self.prolog.assertz("temporal_fact(P, S, O, Start, End, C) :- true")
            
            self.prolog.assertz("""
                fact_true_at(P, S, O, TimePoint, C) :-
                    temporal_fact(P, S, O, Start, End, C),
                    parse_datetime(Start, StartPoint),
                    parse_datetime(End, EndPoint),
                    parse_datetime(TimePoint, Time),
                    StartPoint =< Time,
                    Time =< EndPoint
            """)
            
            self.prolog.assertz("""
                relation_at_time(P, S, O, TimePoint) :-
                    fact_true_at(P, S, O, TimePoint, C),
                    C >= 0.5
            """)
            
            self.prolog.assertz("""
                query_at_time(P, S, O, TimePoint, Confidence) :-
                    fact_true_at(P, S, O, TimePoint, Confidence)
            """)
            
            # Causal relations
            self.prolog.assertz("""
                causes(CauseEvent, EffectEvent, ConfidenceLevel) :-
                    event_before(CauseEvent, EffectEvent),
                    causal_link(CauseEvent, EffectEvent, ConfidenceLevel)
            """)
            
            self.prolog.assertz("causal_link(CauseEvent, EffectEvent, ConfidenceLevel) :- true")
            
            # Add a simple placeholder implementation for parse_datetime
            # This is needed for basic functioning until we have a full implementation
            current_time = datetime.now()
            year, month, day = current_time.year, current_time.month, current_time.day
            timestamp = year*10000000000 + month*100000000 + day*1000000
            
            self.prolog.assertz(f"parse_datetime('now', {timestamp})")
            self.prolog.assertz(f"parse_datetime('today', {timestamp})")
            
            # Add some common time points for testing
            for year in range(2020, 2026):
                ts = year*10000000000 + 1*100000000 + 1*1000000
                self.prolog.assertz(f"parse_datetime('{year}-01-01T00:00:00', {ts})")
            
            self.logger.info("Temporal reasoning framework initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize temporal framework: {str(e)}")
            raise
    
    def add_temporal_fact(self, fact_type, entity, time_info):
        """
        Add a temporal fact to the knowledge base.
        
        Args:
            fact_type (str): Type of temporal fact (event, state, etc.)
            entity (str): Entity the fact is about
            time_info (dict): Time information including start/end times
            
        Returns:
            bool: Success status
        """
        try:
            # Handle different types of temporal facts
            if fact_type == 'point_event':
                # Single point in time event
                time_point = time_info.get('time_point', '')
                prolog_fact = f"event('{entity}', '{fact_type}', '{time_point}')"
                self.prolog.assertz(prolog_fact)
                
            elif fact_type == 'interval_event':
                # Interval event with start and end times
                start_time = time_info.get('start_time', '')
                end_time = time_info.get('end_time', '')
                
                # Assert start and end points
                start_fact = f"event_start('{entity}', '{fact_type}', '{start_time}')"
                end_fact = f"event_end('{entity}', '{fact_type}', '{end_time}')"
                
                self.prolog.assertz(start_fact)
                self.prolog.assertz(end_fact)
                
            elif fact_type == 'state':
                # Ongoing state with possible start/end times
                state = entity
                subject = time_info.get('subject', '')
                start_time = time_info.get('start_time', '')
                
                # Assert state with start time
                state_fact = f"state('{subject}', '{state}', '{start_time}')"
                self.prolog.assertz(state_fact)
                
                # If there's an end time, add it too
                if 'end_time' in time_info:
                    end_time = time_info['end_time']
                    end_fact = f"state_end('{subject}', '{state}', '{end_time}')"
                    self.prolog.assertz(end_fact)
            
            self.logger.info(f"Added temporal fact: {fact_type} for {entity}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add temporal fact: {str(e)}")
            return False
    
    def add_temporal_relation(self, relation_type, entity1, entity2):
        """
        Add a temporal relation between two entities.
        
        Args:
            relation_type (str): Type of relation (before, during, etc.)
            entity1 (str): First entity
            entity2 (str): Second entity
            
        Returns:
            bool: Success status
        """
        try:
            # Format and add the relation
            relation = f"{relation_type}('{entity1}', '{entity2}')"
            self.prolog.assertz(relation)
            
            self.logger.info(f"Added temporal relation: {entity1} {relation_type} {entity2}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add temporal relation: {str(e)}")
            return False
    
    def query_temporal_relationship(self, entity1, entity2):
        """
        Query the temporal relationship between two entities.
        
        Args:
            entity1 (str): First entity
            entity2 (str): Second entity
            
        Returns:
            list: List of temporal relationships between the entities
        """
        try:
            # Query for all possible temporal relations
            relations = []
            
            # Check for 'before' relation
            before_query = f"event_before('{entity1}', '{entity2}')"
            if list(self.prolog.query(before_query)):
                relations.append('before')
            
            # Check for 'after' relation
            after_query = f"event_before('{entity2}', '{entity1}')"
            if list(self.prolog.query(after_query)):
                relations.append('after')
            
            # Check for 'during' relation
            during_query = f"event_during('{entity1}', '{entity2}')"
            if list(self.prolog.query(during_query)):
                relations.append('during')
            
            # Check for other relations as needed
            overlaps_query = f"interval_overlaps('{entity1}', '{entity2}')"
            if list(self.prolog.query(overlaps_query)):
                relations.append('overlaps')
            
            return relations
            
        except Exception as e:
            self.logger.error(f"Error querying temporal relationship: {str(e)}")
            return []
    
    def reason_with_temporal_rules(self, query, variables=None):
        """
        Perform reasoning with temporal rules.
        
        Args:
            query (str): Prolog query string
            variables (dict): Variables to bind in the query
            
        Returns:
            list: Query results
        """
        try:
            # Handle variable binding if provided
            if variables:
                # Create variables for binding
                var_terms = {}
                for var_name in variables:
                    var_terms[var_name] = Variable(var_name)
                
                # Substitute variables in the query
                for var_name, value in variables.items():
                    if isinstance(value, str):
                        query = query.replace(f"?{var_name}", f"'{value}'")
            
            # Execute the query
            results = list(self.prolog.query(query))
            return results
            
        except Exception as e:
            self.logger.error(f"Error in temporal reasoning: {str(e)}")
            return []
    
    def extract_time_expressions(self, text):
        """
        Extract time-related expressions from text.
        
        Args:
            text (str): Input text
            
        Returns:
            list: Extracted time expressions
        """
        # Regular expressions for common time formats
        patterns = [
            # ISO format dates: 2023-04-15
            r'\d{4}-\d{2}-\d{2}',
            # Times: 14:30:00 or 2:30 PM
            r'\d{1,2}:\d{2}(:\d{2})?\s*(?:AM|PM|am|pm)?',
            # Date with month names: April 15, 2023 or 15 April 2023
            r'(?:\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})|(?:(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})',
            # Relative time expressions
            r'(?:today|yesterday|tomorrow|last|next|this)\s+(?:week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)',
            # Duration expressions
            r'(?:for|during)\s+(?:\d+)\s+(?:second|minute|hour|day|week|month|year)s?',
        ]
        
        all_expressions = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                all_expressions.append({
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'type': self._classify_time_expression(match.group(0))
                })
        
        return all_expressions
    
    def _classify_time_expression(self, expr):
        """
        Classify the type of time expression.
        
        Args:
            expr (str): Time expression
            
        Returns:
            str: Type of time expression
        """
        expr_lower = expr.lower()
        
        # Check for point-in-time expressions
        if re.search(r'\d{4}-\d{2}-\d{2}', expr_lower):
            return 'date'
        elif re.search(r'\d{1,2}:\d{2}', expr_lower):
            return 'time'
        
        # Check for relative time expressions
        if any(word in expr_lower for word in ['today', 'yesterday', 'tomorrow']):
            return 'relative_day'
        elif any(word in expr_lower for word in ['last', 'next', 'this']):
            if any(unit in expr_lower for unit in ['week', 'month', 'year']):
                return 'relative_period'
            else:
                return 'relative_day'
        
        # Check for duration expressions
        if any(word in expr_lower for word in ['for', 'during']):
            return 'duration'
        
        # Default to generic time expression
        return 'time_expression'
    
    def normalize_time_expression(self, expr, reference_date=None):
        """
        Convert a time expression to a normalized format.
        
        Args:
            expr (str): Time expression
            reference_date (datetime): Reference date for relative expressions
            
        Returns:
            dict: Normalized time information
        """
        if reference_date is None:
            reference_date = datetime.now()
            
        expr_type = self._classify_time_expression(expr)
        expr_lower = expr.lower()
        
        try:
            # Handle ISO dates directly
            if expr_type == 'date' and re.search(r'\d{4}-\d{2}-\d{2}', expr_lower):
                match = re.search(r'(\d{4})-(\d{2})-(\d{2})', expr_lower)
                if match:
                    year, month, day = map(int, match.groups())
                    dt = datetime(year, month, day)
                    return {
                        'type': 'point',
                        'datetime': dt,
                        'iso_format': dt.isoformat()
                    }
            
            # Handle relative day expressions
            if expr_type == 'relative_day':
                if 'today' in expr_lower:
                    dt = reference_date
                elif 'yesterday' in expr_lower:
                    dt = reference_date - timedelta(days=1)
                elif 'tomorrow' in expr_lower:
                    dt = reference_date + timedelta(days=1)
                else:
                    # Parse day names (Monday, Tuesday, etc.)
                    day_match = re.search(r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)', expr_lower)
                    if day_match:
                        day_name = day_match.group(1)
                        # Convert day name to day number (0=Monday, 6=Sunday)
                        day_map = {
                            'monday': 0, 'mon': 0,
                            'tuesday': 1, 'tue': 1,
                            'wednesday': 2, 'wed': 2,
                            'thursday': 3, 'thu': 3,
                            'friday': 4, 'fri': 4,
                            'saturday': 5, 'sat': 5,
                            'sunday': 6, 'sun': 6
                        }
                        target_day = day_map.get(day_name)
                        if target_day is not None:
                            # Calculate days to add
                            current_day = reference_date.weekday()
                            days_to_add = (target_day - current_day) % 7
                            if days_to_add == 0:  # Same day next week
                                days_to_add = 7
                            dt = reference_date + timedelta(days=days_to_add)
                        else:
                            return None
                    else:
                        return None
                
                return {
                    'type': 'point',
                    'datetime': dt,
                    'iso_format': dt.isoformat()
                }
            
            # Handle relative periods (last week, next month, etc.)
            if expr_type == 'relative_period':
                if 'last week' in expr_lower:
                    start = reference_date - timedelta(days=7)
                    end = reference_date
                elif 'this week' in expr_lower:
                    # Get start of week (Monday)
                    days_since_monday = reference_date.weekday()
                    start = reference_date - timedelta(days=days_since_monday)
                    end = start + timedelta(days=6)  # Sunday
                elif 'next week' in expr_lower:
                    days_to_next_monday = (7 - reference_date.weekday()) % 7
                    start = reference_date + timedelta(days=days_to_next_monday)
                    end = start + timedelta(days=6)  # Sunday
                elif 'last month' in expr_lower:
                    # Simplified approach
                    if reference_date.month == 1:
                        start = datetime(reference_date.year - 1, 12, 1)
                    else:
                        start = datetime(reference_date.year, reference_date.month - 1, 1)
                    end = datetime(reference_date.year, reference_date.month, 1) - timedelta(days=1)
                else:
                    return None
                
                return {
                    'type': 'interval',
                    'start': start,
                    'end': end,
                    'iso_format_start': start.isoformat(),
                    'iso_format_end': end.isoformat()
                }
            
            # Default return for unhandled expressions
            return None
            
        except Exception as e:
            self.logger.error(f"Error normalizing time expression '{expr}': {str(e)}")
            return None
    
    def add_time_dependent_facts(self, predicate, subject, object_value, start_time, end_time=None, confidence=0.9):
        """
        Add time-dependent facts to the knowledge base.
        
        Args:
            predicate (str): The predicate (relation type)
            subject (str): The subject entity
            object_value (str): The object entity
            start_time (str): Start time in ISO format
            end_time (str): End time in ISO format (optional)
            confidence (float): Confidence value (0.0 to 1.0)
            
        Returns:
            bool: Success status
        """
        try:
            # If no end time provided, make it potentially "forever"
            if not end_time:
                end_time = "9999-12-31T23:59:59"
                
            # Create the temporal fact
            fact = f"temporal_fact('{predicate}', '{subject}', '{object_value}', '{start_time}', '{end_time}', {confidence})"
            self.prolog.assertz(fact)
            
            self.logger.info(f"Added temporal fact: {predicate}({subject}, {object_value}) from {start_time} to {end_time}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add time-dependent fact: {str(e)}")
            return False
    
    def query_at_time_point(self, predicate, subject, object_value, time_point, threshold=0.5):
        """
        Query if a fact is true at a specific time point.
        
        Args:
            predicate (str): The predicate to query
            subject (str): The subject entity
            object_value (str): The object entity (or '_' for any object)
            time_point (str): Time point in ISO format
            threshold (float): Confidence threshold
            
        Returns:
            list: Query results with confidence values
        """
        try:
            # Construct the appropriate query
            if object_value == '_':
                # Query for any objects related to the subject at this time
                query = f"query_at_time('{predicate}', '{subject}', O, '{time_point}', C), C >= {threshold}"
            else:
                # Query for specific subject-object relation
                query = f"query_at_time('{predicate}', '{subject}', '{object_value}', '{time_point}', C), C >= {threshold}"
                
            # Execute the query
            results = list(self.prolog.query(query))
            
            # Format the results
            formatted_results = []
            for result in results:
                formatted_result = {
                    'predicate': predicate,
                    'subject': subject,
                    'object': str(result.get('O', object_value)),
                    'time_point': time_point,
                    'confidence': float(result.get('C', 0.0))
                }
                formatted_results.append(formatted_result)
                
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error in temporal query: {str(e)}")
            return []
    
    def extract_temporal_triples(self, text):
        """
        Extract triples with temporal information from text.
        
        Args:
            text (str): Input text to process
            
        Returns:
            list: Temporal triples extracted from the text
        """
        # Example pattern: "X liked Y from 2020 to 2022"
        # This is a simplified version - a real implementation would use NLP
        pattern = re.compile(
            r"(\w+)\s+(liked|trusted|knew|feared|hated)\s+(\w+)\s+(from|between|during|in|since)\s+([^\s]+(?:\s+to\s+[^\s]+)?)",
            re.IGNORECASE
        )
        
        matches = pattern.finditer(text)
        temporal_triples = []
        
        for match in matches:
            subject, predicate, object_val, temp_rel, temp_expr = match.groups()
            
            # Extract start and end times
            if 'to' in temp_expr:
                start_str, end_str = temp_expr.split('to')
                start_time = start_str.strip()
                end_time = end_str.strip()
            else:
                start_time = temp_expr.strip()
                end_time = None
                
            # Try to normalize the time expressions
            start_norm = self.normalize_time_expression(start_time)
            end_norm = self.normalize_time_expression(end_time) if end_time else None
            
            # Create the temporal triple
            temporal_triple = {
                'subject': subject.lower(),
                'predicate': predicate.lower(),
                'object': object_val.lower(),
                'temporal_relation': temp_rel.lower(),
                'start_time': start_norm.get('iso_format') if start_norm else start_time,
                'end_time': end_norm.get('iso_format') if end_norm else end_time,
                'confidence': 0.8  # Default confidence
            }
            
            temporal_triples.append(temporal_triple)
            
        return temporal_triples

    def integrate_with_prolog_engine(self, prolog_engine):
        """
        Integrate this temporal reasoner with an existing Prolog engine.
        
        Args:
            prolog_engine: PrologEngine instance
            
        Returns:
            bool: Success status
        """
        try:
            # Use the existing Prolog instance
            self.prolog = prolog_engine.prolog
            
            # Re-initialize the temporal framework
            self._initialize_temporal_framework()
            
            # Define temporal extensions to existing predicates
            for predicate in ["trusts", "likes", "hates", "knows", "fears", "distrusts", "avoids"]:
                # Create temporal versions of these predicates
                temporal_pred = f"{predicate}_at_time"
                rule = f"{temporal_pred}(X, Y, T) :- fact_true_at({predicate}, X, Y, T, C), C >= 0.5"
                self.prolog.assertz(rule)
                
            self.logger.info("Successfully integrated with Prolog engine")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with Prolog engine: {str(e)}")
            return False
            
    def reason_about_changes(self, subject, predicate, start_time, end_time):
        """
        Reason about how a relation changes over time.
        
        Args:
            subject (str): The subject entity
            predicate (str): The relation predicate
            start_time (str): Start time in ISO format
            end_time (str): End time in ISO format
            
        Returns:
            dict: Analysis of changes over time
        """
        try:
            # Query all temporal facts about this relation
            query = f"temporal_fact('{predicate}', '{subject}', O, Start, End, C)"
            
            results = list(self.prolog.query(query))
            if not results:
                return {'changes': [], 'analysis': 'No temporal data found'}
                
            # Analyze the changes
            changes = []
            for result in results:
                obj = str(result.get('O', ''))
                start = str(result.get('Start', ''))
                end = str(result.get('End', ''))
                conf = float(result.get('C', 0.0))
                
                changes.append({
                    'object': obj,
                    'start_time': start,
                    'end_time': end,
                    'confidence': conf
                })
                
            # Sort changes by start time
            changes.sort(key=lambda x: x['start_time'])
            
            # Analyze the pattern of changes
            if len(changes) <= 1:
                analysis = "Relation is stable over time"
            else:
                # Check if there are multiple objects for the same predicate
                objects = set(change['object'] for change in changes)
                if len(objects) > 1:
                    analysis = f"Relation changed from {changes[0]['object']} to {changes[-1]['object']} over time"
                else:
                    analysis = "Relation target remained the same but confidence may have changed"
                    
            return {
                'changes': changes,
                'analysis': analysis
            }
                
        except Exception as e:
            self.logger.error(f"Error reasoning about changes: {str(e)}")
            return {'error': str(e)}

    def integrate_with_vector_engine(self, vector_engine):
        """Integrate temporal reasoning with vector symbolic engine"""
        self.vector = vector_engine
        
        # Add temporal binding patterns
        temporal_patterns = {
            'before': r'(\w+)\s+happened\s+before\s+(\w+)',
            'after': r'(\w+)\s+occurred\s+after\s+(\w+)',
            'during': r'(\w+)\s+happened\s+during\s+(\w+)',
            'while': r'(\w+)\s+while\s+(\w+)'
        }
        
        for rel_type, pattern in temporal_patterns.items():
            self.vector.add_extraction_pattern(pattern, f"temporal_{rel_type}")

    def extract_temporal_semantics(self, text):
        """Extract temporal meaning using both symbolic and vector approaches"""
        # Get symbolic temporal facts
        temporal_facts = self.extract_temporal_triples(text)
        
        # Get vector-based temporal relations if vector engine exists
        vector_relations = []
        if hasattr(self, 'vector'):
            # Extract semantic temporal relationships
            facts = self.vector._extract_facts_from_text(text)
            vector_relations = [f for f in facts 
                              if f['predicate'].startswith('temporal_')]
        
        return {
            'symbolic_facts': temporal_facts,
            'vector_relations': vector_relations
        }