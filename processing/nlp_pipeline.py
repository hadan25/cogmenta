import re

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