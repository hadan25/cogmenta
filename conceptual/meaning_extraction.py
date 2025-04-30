"""
Meaning Extraction System for Cogmenta Core.
Extracts structured meaning from natural language text.
Converts token-level representation to conceptual representation.
"""

import re
import time
import json
import uuid
from collections import defaultdict

class MeaningExtractionSystem:
    """
    Converts natural language text into structured meaning representations.
    Bridges the gap between token-based processing and concept-based understanding.
    """
    
    def __init__(self, concept_system=None, meaning_map=None, nlp_pipeline=None):
        """
        Initialize the meaning extraction system.
        
        Args:
            concept_system: ConceptEmbeddingSystem instance (optional)
            meaning_map: StructuredMeaningMap instance (optional)
            nlp_pipeline: NLPPipeline instance (optional)
        """
        self.concept_system = concept_system
        self.meaning_map = meaning_map
        self.nlp_pipeline = nlp_pipeline
        
        # Frame definitions for common situations
        self.frames = {
            'trust_relation': {
                'elements': ['truster', 'trusted', 'trust_level'],
                'patterns': [
                    r'(\w+)\s+(trusts|does\s+not\s+trust)\s+(\w+)',
                    r'(\w+)\s+(doesn\'t\s+trust)\s+(\w+)',
                    r'(\w+)\s+(has\s+trust\s+in)\s+(\w+)',
                    r'(\w+)\s+(puts\s+trust\s+in)\s+(\w+)'
                ]
            },
            'like_relation': {
                'elements': ['liker', 'liked', 'like_level'],
                'patterns': [
                    r'(\w+)\s+(likes|does\s+not\s+like)\s+(\w+)',
                    r'(\w+)\s+(doesn\'t\s+like)\s+(\w+)',
                    r'(\w+)\s+(enjoys|appreciates)\s+(\w+)'
                ]
            },
            'identity_relation': {
                'elements': ['entity', 'type'],
                'patterns': [
                    r'(\w+)\s+is\s+(?:a|an)\s+(\w+)',
                    r'(\w+)\s+(?:is|are)\s+(\w+)'
                ]
            }
        }
        
        # Logical operators for handling complex meaning
        self.logical_operators = {
            'and': {'type': 'conjunction', 'patterns': [r'\sand\s', r'\s&\s', r'\splus\s']},
            'or': {'type': 'disjunction', 'patterns': [r'\sor\s', r'\s\|\s']},
            'not': {'type': 'negation', 'patterns': [r'\snot\s', r'\sn\'t\s', r'doesn\'t', r'don\'t', r'isn\'t']}
        }
        
        # Templates for common constructions
        self.meaning_templates = self._init_meaning_templates()
        
        # For tracking processed texts
        self.processed_texts = {}  # text -> meaning structure
    
    def _init_meaning_templates(self):
        """Initialize templates for common meaning structures"""
        return {
            # Simple subject-verb-object template
            'svo': {
                'pattern': r'(\w+)\s+(\w+(?:\s+\w+)?)\s+(\w+)',
                'structure': {
                    'type': 'proposition',
                    'subject': '{1}',
                    'predicate': '{2}',
                    'object': '{3}'
                }
            },
            # Comparison template
            'comparison': {
                'pattern': r'(\w+)\s+(?:is|seems|appears)\s+(more|less)\s+(\w+)\s+than\s+(\w+)',
                'structure': {
                    'type': 'comparison',
                    'entity1': '{1}',
                    'entity2': '{4}',
                    'property': '{3}',
                    'direction': '{2}'
                }
            },
            # Causation template
            'causation': {
                'pattern': r'(\w+)\s+(?:causes|makes|forces)\s+(\w+)\s+to\s+(\w+)',
                'structure': {
                    'type': 'causation',
                    'cause': '{1}',
                    'affected': '{2}',
                    'effect': '{3}'
                }
            }
        }
    
    def extract_meaning(self, text):
        """
        Extract structured meaning from text.
        
        Args:
            text: Input text to process
            
        Returns:
            Structured meaning representation
        """
        # Check if we've already processed this text
        if text in self.processed_texts:
            return self.processed_texts[text]
            
        # Initialize the meaning structure
        meaning = {
            'id': str(uuid.uuid4()),
            'type': 'meaning_structure',
            'text': text,
            'timestamp': time.time(),
            'concepts': [],
            'propositions': [],
            'frames': [],
            'sentiment': 0
        }
        
        # Use NLP pipeline for preliminary analysis if available
        nlp_analysis = None
        if self.nlp_pipeline:
            nlp_analysis = self.nlp_pipeline.process(text)
            
            # Extract entities and relations from NLP analysis
            if nlp_analysis:
                # Extract entities
                for entity in nlp_analysis.get('entities', []):
                    meaning['concepts'].append({
                        'id': str(uuid.uuid4()),
                        'type': 'entity',
                        'text': entity['text'],
                        'span': (entity['start'], entity['end'])
                    })
                
                # Extract relations
                for relation in nlp_analysis.get('relations', []):
                    proposition = {
                        'id': str(uuid.uuid4()),
                        'type': 'proposition',
                        'subject': relation['subject'],
                        'predicate': relation['predicate'],
                        'object': relation['object'],
                        'negated': relation.get('negated', False),
                        'confidence': relation.get('confidence', 0.9)
                    }
                    meaning['propositions'].append(proposition)
                
                # Set sentiment
                if 'sentiment' in nlp_analysis:
                    meaning['sentiment'] = nlp_analysis['sentiment']
        
        # Use concept system to extract concepts if available
        if self.concept_system:
            extracted_concepts = self.concept_system.extract_concepts_from_text(text)
            
            # Merge with existing concepts
            existing_concept_texts = set(c['text'].lower() for c in meaning['concepts'])
            
            for concept_name, confidence, text_span in extracted_concepts:
                if text_span.lower() not in existing_concept_texts:
                    meaning['concepts'].append({
                        'id': str(uuid.uuid4()),
                        'type': 'concept',
                        'text': text_span,
                        'concept': concept_name,
                        'confidence': confidence
                    })
                    existing_concept_texts.add(text_span.lower())
        
        # Match against frame patterns
        for frame_name, frame in self.frames.items():
            matches = self._match_frame_patterns(text, frame)
            
            # Match against frame patterns
            for match in matches:
                elements = {}
                
                # Extract elements based on match groups
                groups = match.groups()
                if len(groups) >= 3:  # We expect at least 3 capturing groups
                    # Standard pattern with subject, predicate, object
                    elements['subject'] = groups[0]
                    elements['predicate'] = groups[1]
                    elements['object'] = groups[2]
                    
                    # Check for negation in predicate
                    negated = any(neg in groups[1].lower() for neg in ['not', 'n\'t', 'doesn'])
                    
                    # Map to frame elements
                    if frame_name == 'trust_relation':
                        elements['truster'] = groups[0]
                        elements['trusted'] = groups[2]
                        elements['trust_level'] = 'negative' if negated else 'positive'
                    elif frame_name == 'like_relation':
                        elements['liker'] = groups[0]
                        elements['liked'] = groups[2]
                        elements['like_level'] = 'negative' if negated else 'positive'
                    elif frame_name == 'identity_relation':
                        elements['entity'] = groups[0]
                        elements['type'] = groups[2]
                
                # Create frame instance
                frame_instance = {
                    'id': str(uuid.uuid4()),
                    'type': 'frame_instance',
                    'frame': frame_name,
                    'elements': elements,
                    'span': (match.start(), match.end()),
                    'text': match.group(0)
                }
                
                meaning['frames'].append(frame_instance)
                
                # Also create corresponding proposition
                if frame_name == 'trust_relation':
                    predicate = 'distrusts' if elements.get('trust_level') == 'negative' else 'trusts'
                    proposition = {
                        'id': str(uuid.uuid4()),
                        'type': 'proposition',
                        'subject': elements.get('truster', ''),
                        'predicate': predicate,
                        'object': elements.get('trusted', ''),
                        'negated': False,  # Already captured in the predicate
                        'confidence': 0.9,
                        'source': 'frame'
                    }
                    
                    # Add if not duplicate
                    self._add_proposition_if_new(meaning, proposition)
                elif frame_name == 'like_relation':
                    predicate = 'dislikes' if elements.get('like_level') == 'negative' else 'likes'
                    proposition = {
                        'id': str(uuid.uuid4()),
                        'type': 'proposition',
                        'subject': elements.get('liker', ''),
                        'predicate': predicate,
                        'object': elements.get('liked', ''),
                        'negated': False,  # Already captured in the predicate
                        'confidence': 0.9,
                        'source': 'frame'
                    }
                    
                    # Add if not duplicate
                    self._add_proposition_if_new(meaning, proposition)
                elif frame_name == 'identity_relation':
                    proposition = {
                        'id': str(uuid.uuid4()),
                        'type': 'proposition',
                        'subject': elements.get('entity', ''),
                        'predicate': 'is_a',
                        'object': elements.get('type', ''),
                        'negated': False,
                        'confidence': 0.9,
                        'source': 'frame'
                    }
                    
                    # Add if not duplicate
                    self._add_proposition_if_new(meaning, proposition)
        
        # Apply meaning templates
        self._apply_meaning_templates(text, meaning)
        
        # Extract logical structure
        self._extract_logical_structure(text, meaning)
        
        # Create a structured meaning map if available
        if self.meaning_map:
            graph_id = self.meaning_map.create_meaning_graph(text)
            if graph_id:
                meaning['meaning_graph_id'] = graph_id
        
        # Store for future reference
        self.processed_texts[text] = meaning
        
        return meaning
    
    def _match_frame_patterns(self, text, frame):
        """
        Match text against frame patterns.
        
        Args:
            text: Text to match
            frame: Frame definition
            
        Returns:
            List of regex match objects
        """
        matches = []
        
        for pattern in frame['patterns']:
            # Find all matches
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append(match)
                
        return matches
    
    def _add_proposition_if_new(self, meaning, proposition):
        """
        Add a proposition to the meaning structure if it's not a duplicate.
        
        Args:
            meaning: Meaning structure to update
            proposition: Proposition to add
            
        Returns:
            True if added, False if duplicate
        """
        # Check for duplicates
        for existing in meaning['propositions']:
            if (existing['subject'].lower() == proposition['subject'].lower() and
                existing['predicate'].lower() == proposition['predicate'].lower() and
                existing['object'].lower() == proposition['object'].lower() and
                existing['negated'] == proposition['negated']):
                return False
                
        # Add new proposition
        meaning['propositions'].append(proposition)
        return True
    
    def _apply_meaning_templates(self, text, meaning):
        """
        Apply meaning templates to extract structured meanings.
        
        Args:
            text: Input text
            meaning: Meaning structure to update
        """
        for template_name, template in self.meaning_templates.items():
            # Find all matches
            for match in re.finditer(template['pattern'], text, re.IGNORECASE):
                # Create structure from template
                structure = template['structure'].copy()
                
                # Replace placeholders with match groups
                for key, value in list(structure.items()):
                    if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                        # Extract group index
                        try:
                            group_idx = int(value[1:-1])
                            if group_idx <= len(match.groups()):
                                structure[key] = match.group(group_idx)
                        except (ValueError, IndexError):
                            continue
                
                # Add ID
                structure['id'] = str(uuid.uuid4())
                
                # Add to appropriate section based on type
                if structure['type'] == 'proposition':
                    # Convert to proposition
                    proposition = {
                        'id': structure['id'],
                        'type': 'proposition',
                        'subject': structure['subject'],
                        'predicate': structure['predicate'],
                        'object': structure['object'],
                        'negated': False,
                        'confidence': 0.8,
                        'source': 'template'
                    }
                    
                    # Add if not duplicate
                    self._add_proposition_if_new(meaning, proposition)
                else:
                    # Add other structure types directly
                    if 'structures' not in meaning:
                        meaning['structures'] = []
                        
                    meaning['structures'].append(structure)
    
    def _extract_logical_structure(self, text, meaning):
        """
        Extract logical structure from text.
        
        Args:
            text: Input text
            meaning: Meaning structure to update
        """
        # Check for conjunctions (and)
        for op_name, operator in self.logical_operators.items():
            if op_name == 'and':
                # Look for conjunction patterns
                for pattern in operator['patterns']:
                    # Split text on conjunction
                    parts = re.split(pattern, text)
                    
                    if len(parts) > 1:
                        # We have a conjunction
                        conjunction = {
                            'id': str(uuid.uuid4()),
                            'type': 'logical_operation',
                            'operator': 'conjunction',
                            'components': []
                        }
                        
                        # Process each part
                        for i, part in enumerate(parts):
                            if part.strip():
                                # Extract meaning from this part
                                part_meaning = self.extract_meaning(part.strip())
                                
                                # Add propositions to components
                                for prop in part_meaning.get('propositions', []):
                                    conjunction['components'].append(prop['id'])
                                
                                # Also add propositions to the main meaning
                                for prop in part_meaning.get('propositions', []):
                                    self._add_proposition_if_new(meaning, prop)
                        
                        # Add conjunction if it has components
                        if conjunction['components']:
                            if 'logical_operations' not in meaning:
                                meaning['logical_operations'] = []
                                
                            meaning['logical_operations'].append(conjunction)
            
            elif op_name == 'not':
                # Look for negation patterns
                for pattern in operator['patterns']:
                    if re.search(pattern, text, re.IGNORECASE):
                        # Mark that this meaning contains negation
                        meaning['contains_negation'] = True
                        break
    
    def extract_concepts_and_relations(self, text):
        """
        Extract concepts and relations from text.
        
        Args:
            text: Input text
            
        Returns:
            Dict with concepts and relations
        """
        # Extract full meaning
        meaning = self.extract_meaning(text)
        
        # Extract concepts and relations into simpler format
        result = {
            'concepts': [],
            'relations': []
        }
        
        # Add concepts
        for concept in meaning.get('concepts', []):
            result['concepts'].append({
                'text': concept['text'],
                'type': concept.get('type', 'entity'),
                'confidence': concept.get('confidence', 1.0)
            })
            
        # Add relations from propositions
        for prop in meaning.get('propositions', []):
            relation = {
                'subject': prop['subject'],
                'predicate': prop['predicate'],
                'object': prop['object'],
                'negated': prop.get('negated', False),
                'confidence': prop.get('confidence', 1.0)
            }
            result['relations'].append(relation)
            
        return result
    
    def compare_meanings(self, text1, text2):
        """
        Compare the meanings of two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Comparison result dictionary
        """
        # Extract meanings
        meaning1 = self.extract_meaning(text1)
        meaning2 = self.extract_meaning(text2)
        
        # Compare propositions
        shared_props = []
        unique_props1 = []
        unique_props2 = []
        
        # Normalize propositions for comparison
        props1 = [(p['subject'].lower(), p['predicate'].lower(), p['object'].lower(), p['negated']) 
                 for p in meaning1.get('propositions', [])]
        props2 = [(p['subject'].lower(), p['predicate'].lower(), p['object'].lower(), p['negated']) 
                 for p in meaning2.get('propositions', [])]
        
        # Find shared and unique
        for p1 in props1:
            if p1 in props2:
                shared_props.append(p1)
            else:
                unique_props1.append(p1)
                
        for p2 in props2:
            if p2 not in props1:
                unique_props2.append(p2)
        
        # Compare concepts
        concepts1 = set(c['text'].lower() for c in meaning1.get('concepts', []))
        concepts2 = set(c['text'].lower() for c in meaning2.get('concepts', []))
        
        shared_concepts = concepts1 & concepts2
        unique_concepts1 = concepts1 - concepts2
        unique_concepts2 = concepts2 - concepts1
        
        # Calculate similarity scores
        prop_similarity = len(shared_props) / max(1, len(props1) + len(props2) - len(shared_props))
        concept_similarity = len(shared_concepts) / max(1, len(concepts1) + len(concepts2) - len(shared_concepts))
        
        # Overall similarity score (weighted average)
        overall_similarity = 0.7 * prop_similarity + 0.3 * concept_similarity
        
        return {
            'overall_similarity': overall_similarity,
            'proposition_similarity': prop_similarity,
            'concept_similarity': concept_similarity,
            'shared_propositions': shared_props,
            'text1_unique_propositions': unique_props1,
            'text2_unique_propositions': unique_props2,
            'shared_concepts': list(shared_concepts),
            'text1_unique_concepts': list(unique_concepts1),
            'text2_unique_concepts': list(unique_concepts2)
        }
    
    def extract_key_meaning(self, text):
        """
        Extract the most important/central meaning from text.
        
        Args:
            text: Input text
            
        Returns:
            Key meaning as a proposition or None
        """
        # Extract full meaning
        meaning = self.extract_meaning(text)
        
        # No propositions, no key meaning
        if not meaning.get('propositions'):
            return None
            
        # Score propositions by importance
        scored_props = []
        
        for prop in meaning['propositions']:
            score = 0
            
            # Propositions from frames are more likely to be central
            if prop.get('source') == 'frame':
                score += 0.3
                
            # Propositions with entities mentioned in other propositions are more central
            subj = prop['subject'].lower()
            obj = prop['object'].lower()
            
            subj_mentions = 0
            obj_mentions = 0
            
            for other_prop in meaning['propositions']:
                if other_prop['id'] != prop['id']:
                    if other_prop['subject'].lower() == subj or other_prop['object'].lower() == subj:
                        subj_mentions += 1
                    if other_prop['subject'].lower() == obj or other_prop['object'].lower() == obj:
                        obj_mentions += 1
            
            # Add score based on mentions
            score += 0.1 * subj_mentions
            score += 0.1 * obj_mentions
            
            # Common relations like 'trusts', 'likes' are often central
            if prop['predicate'].lower() in ['trusts', 'likes', 'knows', 'fears', 'is_a']:
                score += 0.2
                
            # Negated propositions can be more informative
            if prop.get('negated', False):
                score += 0.1
                
            scored_props.append((prop, score))
            
        # Sort by score (descending)
        scored_props.sort(key=lambda x: x[1], reverse=True)
        
        # Return top proposition if any
        if scored_props:
            return scored_props[0][0]
            
        return None
    
    def extract_meaning_summary(self, text):
        """
        Create a concise natural language summary of the meaning.
        
        Args:
            text: Input text
            
        Returns:
            Summary string
        """
        # Extract meaning
        meaning = self.extract_meaning(text)
        
        # No propositions, no summary
        if not meaning.get('propositions'):
            return "No clear meaning could be extracted."
            
        # Get key propositions (up to 3)
        key_props = []
        
        # Score propositions by importance (similar to extract_key_meaning)
        scored_props = []
        
        for prop in meaning['propositions']:
            score = 0
            
            # Propositions from frames are more likely to be central
            if prop.get('source') == 'frame':
                score += 0.3
                
            # Propositions with entities mentioned in other propositions are more central
            subj = prop['subject'].lower()
            obj = prop['object'].lower()
            
            subj_mentions = 0
            obj_mentions = 0
            
            for other_prop in meaning['propositions']:
                if other_prop['id'] != prop['id']:
                    if other_prop['subject'].lower() == subj or other_prop['object'].lower() == subj:
                        subj_mentions += 1
                    if other_prop['subject'].lower() == obj or other_prop['object'].lower() == obj:
                        obj_mentions += 1
            
            # Add score based on mentions
            score += 0.1 * subj_mentions
            score += 0.1 * obj_mentions
            
            # Common relations like 'trusts', 'likes' are often central
            if prop['predicate'].lower() in ['trusts', 'likes', 'knows', 'fears', 'is_a']:
                score += 0.2
                
            scored_props.append((prop, score))
            
        # Sort by score (descending)
        scored_props.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 3 propositions
        key_props = [p for p, _ in scored_props[:3]]
        
        # Format each proposition as natural language
        sentences = []
        
        for prop in key_props:
            subject = prop['subject'].capitalize()
            predicate = prop['predicate'].lower()
            object_val = prop['object']
            
            # Format based on predicate
            if predicate == 'trusts':
                sentence = f"{subject} trusts {object_val}."
            elif predicate == 'distrusts':
                sentence = f"{subject} does not trust {object_val}."
            elif predicate == 'likes':
                sentence = f"{subject} likes {object_val}."
            elif predicate == 'dislikes':
                sentence = f"{subject} does not like {object_val}."
            elif predicate == 'is_a':
                sentence = f"{subject} is a {object_val}."
            else:
                # Generic format
                if prop.get('negated', False):
                    sentence = f"{subject} does not {predicate} {object_val}."
                else:
                    sentence = f"{subject} {predicate} {object_val}."
                    
            sentences.append(sentence)
            
        # Add sentiment if available
        if 'sentiment' in meaning:
            sentiment = meaning['sentiment']
            if sentiment > 0.3:
                sentences.append("The overall sentiment is positive.")
            elif sentiment < -0.3:
                sentences.append("The overall sentiment is negative.")
                
        # Combine sentences
        summary = " ".join(sentences)
        
        return summary