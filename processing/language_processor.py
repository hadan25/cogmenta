# cogmenta_core/processing/language_processor.py

import re
from collections import defaultdict

class LanguageProcessor:
    """
    Comprehensive language processing module that combines NLP pipeline and language understanding.
    """
    
    def __init__(self):
        """Initialize language processor"""
        self.nlp_pipeline = NLPPipeline()
        self.language_understanding = LanguageUnderstanding(self.nlp_pipeline)
        
    def process(self, text):
        """
        Process text with both NLP pipeline and enhanced understanding.
        
        Args:
            text: Input text to process
            
        Returns:
            Comprehensive language processing results
        """
        # Get enhanced understanding
        enhanced = self.language_understanding.process(text)
        
        # Extract facts as triples for symbolic processing
        triples = self._extract_triples(enhanced)
        
        # Return combined results
        return {
            **enhanced,
            "triples": triples
        }
    
    def _extract_triples(self, processed_data):
        """
        Extract subject-predicate-object triples from processed data.
        
        Args:
            processed_data: Data already processed by language understanding
            
        Returns:
            List of (subject, predicate, object, confidence) tuples
        """
        triples = []
        
        # Extract from relations
        if "relations" in processed_data:
            for relation in processed_data["relations"]:
                # Handle negation
                predicate = relation["predicate"]
                if relation.get("negated", False):
                    predicate = f"not_{predicate}"
                    
                # Create triple with confidence
                triples.append((
                    relation["subject"],
                    predicate,
                    relation["object"],
                    relation.get("confidence", 0.7)
                ))
                
        return triples
    
    def extract_facts(self, text):
        """
        Extract facts directly from text.
        
        Args:
            text: Input text
            
        Returns:
            List of (subject, predicate, object, confidence) tuples
        """
        processed = self.process(text)
        return processed["triples"]
    
    def get_intent(self, text):
        """
        Get the primary intent from text.
        
        Args:
            text: Input text
            
        Returns:
            Intent string
        """
        processed = self.process(text)
        return processed.get("enhanced_intent", {}).get("intent", "unknown")
    
    def extract_concepts(self, text):
        """
        Extract key concepts from text.
        
        Args:
            text: Input text
            
        Returns:
            List of concepts with confidence scores
        """
        processed = self.process(text)
        concepts = []
        
        # Extract from entities
        for entity in processed.get("enhanced_entities", []):
            if entity.get("type") == "concept":
                concepts.append((entity["text"], 0.8))
                
        # Extract additional concepts from text
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        for word in words:
            if word not in [c[0].lower() for c in concepts]:
                # Simple heuristic: longer words are more likely to be concepts
                confidence = min(0.6, 0.3 + (len(word) / 20))
                concepts.append((word, confidence))
                
        return concepts


class NLPPipeline:
    def __init__(self):
        # Would use spaCy in a real implementation
        self.relation_patterns = [
            (r"(\w+)\s+(trusts|likes|loves|knows|fears|hates|avoids|helps)\s+(\w+)", 0.9),
            (r"(\w+)\s+(trusts|likes|loves|knows)\s+(no\s*one|nobody)", 0.9),
            (r"(\w+)\s+(doesn't|does\s+not)\s+(trust|like|love|know|fear)\s+(\w+)", 0.85)
        ]
        
    def process(self, text):
        """Process text and extract structured information"""
        # Extract entities
        entities = self._extract_entities(text)
        
        # Extract relations
        relations = self._extract_relations(text)
        
        # Analyze sentiment (simplified)
        sentiment = self._analyze_sentiment(text)
        
        # Detect intent
        intent = self._detect_intent(text)
        
        return {
            "entities": entities,
            "relations": relations,
            "sentiment": sentiment,
            "intent": intent,
            "text": text  # Return original text
        }
        
    def _extract_entities(self, text):
        """Extract named entities (simplified)"""
        entities = []
        
        # Simple pattern matching for names (capitalized words)
        for match in re.finditer(r'\b([A-Z][a-z]+)\b', text):
            entities.append({
                "text": match.group(1),
                "type": "PERSON",  # Assume all capitalized words are people
                "start": match.start(),
                "end": match.end()
            })
            
        return entities
        
    def _extract_relations(self, text):
        """Extract subject-predicate-object relations"""
        relations = []
        
        # Apply relation patterns
        for pattern, conf in self.relation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) == 3:  # Standard positive relation
                    subj, pred, obj = groups
                    
                    # Handle "no one" case
                    if obj.lower() in ['no one', 'nobody']:
                        relations.append({
                            "subject": subj.lower(),
                            "predicate": f"{pred.lower()}_nobody",
                            "object": "true",
                            "confidence": conf,
                            "negated": False
                        })
                    else:
                        relations.append({
                            "subject": subj.lower(),
                            "predicate": pred.lower(),
                            "object": obj.lower(),
                            "confidence": conf,
                            "negated": False
                        })
                        
                elif len(groups) == 4:  # Negated relation
                    subj, neg, pred, obj = groups
                    relations.append({
                        "subject": subj.lower(),
                        "predicate": pred.lower(),
                        "object": obj.lower(),
                        "confidence": conf,
                        "negated": True
                    })
                    
        return relations
        
    def _analyze_sentiment(self, text):
        """Simple sentiment analysis"""
        text_lower = text.lower()
        
        positive_words = {'good', 'great', 'excellent', 'happy', 'trust', 'like', 'love', 'friend'}
        negative_words = {'bad', 'terrible', 'sad', 'hate', 'fear', 'distrust', 'avoid', 'enemy'}
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # Check for negations
        if re.search(r"(don't|doesn't|not|no|never)\s+\w+\s+(trust|like|love)", text_lower):
            pos_count -= 1
            neg_count += 1
            
        # Calculate sentiment score (-1 to 1)
        total = pos_count + neg_count
        if total == 0:
            return 0  # Neutral
        return (pos_count - neg_count) / total
        
    def _detect_intent(self, text):
        """Detect the intent of the text"""
        text_lower = text.lower()
        
        # Question detection
        if '?' in text or re.search(r'\b(who|what|when|where|why|how|does|is|are|can|will|would)\b', text_lower):
            if re.search(r'\bwhy\b', text_lower):
                return "explanation_query"
            elif re.search(r'\b(who|what)\b.*\b(like|trust|fear|hate|avoid)\b', text_lower):
                return "relationship_query"
            else:
                return "information_query"
                
        # Statement detection
        if re.search(r'\b\w+\s+(trusts|likes|loves|knows|fears|hates|avoids)\s+\w+\b', text_lower):
            return "relationship_statement"
            
        if re.search(r'\b(if|would|suppose)\b', text_lower):
            return "hypothetical"
            
        # Default
        return "statement"


class LanguageUnderstanding:
    """
    Enhanced language understanding capabilities beyond basic NLP pipeline.
    """
    
    def __init__(self, nlp_pipeline):
        """
        Initialize language understanding.
        
        Args:
            nlp_pipeline: Base NLP pipeline
        """
        self.nlp = nlp_pipeline
        self.intent_patterns = self._init_intent_patterns()
        self.entity_types = self._init_entity_types()
        
    def _init_intent_patterns(self):
        """Initialize patterns for intent recognition"""
        return {
            "question_fact": [
                r"(?:is|are|does|do|can|could|would|will)\s+\w+",
                r"(?:who|what|when|where|why|how)\s+\w+"
            ],
            "command": [
                r"^(?:please\s+)?(?:show|tell|give|find|search|look|get)\s+\w+",
                r"^(?:please\s+)?(?:explain|describe|elaborate|clarify)\s+\w+"
            ],
            "statement": [
                r"^\w+\s+(?:is|are|was|were|has|have|had)\s+\w+"
            ],
            "greeting": [
                r"^(?:hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening))[\.\!\s]*$"
            ],
            "farewell": [
                r"^(?:goodbye|bye|see\s+you|farewell|exit|quit)[\.\!\s]*$"
            ],
            "thanks": [
                r"^(?:thanks|thank\s+you|appreciate|grateful)[\.\!\s]*$"
            ],
            "preference": [
                r"(?:prefer|like|love|enjoy|hate|dislike)\s+\w+"
            ],
            "opinion": [
                r"(?:think|believe|feel|consider|assume)\s+\w+"
            ]
        }
        
    def _init_entity_types(self):
        """Initialize entity type recognition"""
        return {
            "person": [r"[A-Z][a-z]+", r"Alice", r"Bob", r"Charlie", r"Dave"],
            "relation": [r"trusts?", r"likes?", r"knows?", r"hates?", r"avoids?"],
            "concept": [r"trust", r"friendship", r"relationship", r"confidence"],
            "time": [r"today", r"yesterday", r"tomorrow", r"now", r"later"]
        }
        
    def process(self, text):
        """
        Process text to extract enhanced understanding.
        
        Args:
            text: Input text to process
            
        Returns:
            Enhanced understanding dict
        """
        # Get basic NLP processing
        base_nlp = self.nlp.process(text)
        
        # Enhance with intent recognition
        intent_info = self._detect_intent(text)
        
        # Enhance entity recognition
        entities = self._enhance_entities(base_nlp.get("entities", []), text)
        
        # Extract conversation attributes
        attributes = self._extract_attributes(text, intent_info["intent"])
        
        # Combine results
        enhanced = {
            **base_nlp,
            "enhanced_intent": intent_info,
            "enhanced_entities": entities,
            "attributes": attributes
        }
        
        return enhanced
    
    def _detect_intent(self, text):
        """
        Detect user intent from text.
        
        Args:
            text: Input text
            
        Returns:
            Intent information
        """
        text_lower = text.lower()
        detected_intents = []
        
        # Try to match each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected_intents.append(intent)
                    break
        
        # Choose primary intent based on priorities
        primary_intent = "unknown"
        confidence = 0.5
        
        intent_priorities = ["question_fact", "command", "greeting", "farewell", "statement"]
        for priority_intent in intent_priorities:
            if priority_intent in detected_intents:
                primary_intent = priority_intent
                confidence = 0.8
                break
                
        return {
            "intent": primary_intent,
            "all_intents": detected_intents,
            "confidence": confidence
        }
    
    def _enhance_entities(self, base_entities, text):
        """
        Enhance entity recognition.
        
        Args:
            base_entities: Entities from base NLP
            text: Original text
            
        Returns:
            Enhanced entities
        """
        enhanced = base_entities.copy()
        text_lower = text.lower()
        
        # Apply additional entity recognition
        for entity_type, patterns in self.entity_types.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    # Check if this entity is already detected
                    match_text = match.group(0)
                    already_detected = False
                    
                    for entity in enhanced:
                        if (entity["text"].lower() == match_text.lower() or
                            (entity["start"] <= match.start() and entity["end"] >= match.end())):
                            already_detected = True
                            break
                            
                    if not already_detected:
                        new_entity = {
                            "text": match_text,
                            "type": entity_type,
                            "start": match.start(),
                            "end": match.end(),
                            "enhanced": True
                        }
                        enhanced.append(new_entity)
        
        return enhanced
    
    def _extract_attributes(self, text, intent):
        """
        Extract conversational attributes.
        
        Args:
            text: Input text
            intent: Detected intent
            
        Returns:
            Attribute dictionary
        """
        text_lower = text.lower()
        attributes = {
            "formality": "neutral",
            "sentiment": "neutral",
            "word_count": len(text.split()),
            "question_mark": "?" in text,
            "exclamation": "!" in text
        }
        
        # Detect formality
        formal_indicators = ["could you please", "would you kindly", "I would like", "thank you"]
        informal_indicators = ["hey", "yeah", "cool", "ok", "sup", "lol"]
        
        for indicator in formal_indicators:
            if indicator in text_lower:
                attributes["formality"] = "formal"
                break
                
        for indicator in informal_indicators:
            if indicator in text_lower:
                attributes["formality"] = "informal"
                break
                
        # Simple sentiment analysis
        positive = ["good", "great", "excellent", "amazing", "wonderful", "happy", "like", "love"]
        negative = ["bad", "terrible", "awful", "horrible", "sad", "hate", "dislike"]
        
        pos_count = sum(1 for word in positive if word in text_lower)
        neg_count = sum(1 for word in negative if word in text_lower)
        
        if pos_count > neg_count:
            attributes["sentiment"] = "positive"
        elif neg_count > pos_count:
            attributes["sentiment"] = "negative"
            
        return attributes 