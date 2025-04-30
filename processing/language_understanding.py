# cogmenta_core/processing/language_understanding.py

import re
from collections import defaultdict

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