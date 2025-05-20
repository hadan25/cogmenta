#processing/context_manager.py
import time
from collections import deque

class ContextManager:
    def __init__(self, context_window=5):
        self.context_window = context_window
        self.conversation_history = deque(maxlen=context_window)
        self.current_topic = None
        self.current_entities = set()
        self.topic_history = []
        
    def add_utterance(self, text, processed_nlp=None):
        """Add an utterance to the conversation history and update context"""
        # Create utterance record
        utterance = {
            'text': text,
            'timestamp': time.time(),
            'nlp': processed_nlp,
            'entities': []
        }
        
        # Extract entities if NLP results provided
        if processed_nlp and 'entities' in processed_nlp:
            utterance['entities'] = processed_nlp['entities']
            
            # Update current entities
            for entity in processed_nlp['entities']:
                self.current_entities.add(entity['text'].lower())
                
        # Add to history
        self.conversation_history.append(utterance)
        
        # Update topic if needed
        if processed_nlp:
            self._update_topic(processed_nlp)
            
        return utterance
        
    def _update_topic(self, nlp_result):
        """Update the current conversation topic"""
        # Simple topic detection based on entities and intent
        if nlp_result.get('intent') == 'relationship_query' or nlp_result.get('intent') == 'relationship_statement':
            # Extract relationship entities
            entities = []
            for relation in nlp_result.get('relations', []):
                entities.append(relation['subject'])
                if relation['object'] != 'true':  # Skip for 'trusts_nobody' type
                    entities.append(relation['object'])
                    
            if entities:
                new_topic = "relationships:" + ",".join(entities)
                
                # Only update if topic changed
                if new_topic != self.current_topic:
                    if self.current_topic:
                        self.topic_history.append((self.current_topic, time.time()))
                    self.current_topic = new_topic
                    
        elif nlp_result.get('intent') == 'explanation_query':
            # Set topic to explanation about mentioned entities
            entities = [e['text'].lower() for e in nlp_result.get('entities', [])]
            if entities:
                new_topic = "explanation:" + ",".join(entities)
                
                if new_topic != self.current_topic:
                    if self.current_topic:
                        self.topic_history.append((self.current_topic, time.time()))
                    self.current_topic = new_topic
        
    def get_relevant_context(self, query):
        """Get context relevant to the current query"""
        query_lower = query.lower()
        
        # Extract query entities (very simplified)
        query_entities = set()
        for word in query_lower.split():
            if word[0].isupper():
                query_entities.add(word.lower())
                
        # Find relevant utterances
        relevant_utterances = []
        for utterance in self.conversation_history:
            # Check for entity overlap
            utterance_entities = {e['text'].lower() for e in utterance.get('entities', [])}
            overlap = utterance_entities.intersection(query_entities)
            
            if overlap:
                relevant_utterances.append(utterance)
                
        return {
            'current_topic': self.current_topic,
            'current_entities': list(self.current_entities),
            'relevant_history': relevant_utterances
        }
        
    def reset_context(self):
        """Reset the conversation context"""
        self.conversation_history.clear()
        self.current_topic = None
        self.current_entities = set()
        self.topic_history.append((self.current_topic, time.time()))  # Record topic end