# integrated_memory_system.py
from models.memory.episodic_memory import EpisodicMemory
from models.memory.semantic_memory import SemanticMemory
from models.memory.working_memory import WorkingMemory
import time
import numpy as np
from typing import Dict, List, Any, Optional, Union

class IntegratedMemorySystem:
    """Coordinates between episodic, semantic, and working memory systems"""
    
    def __init__(self, episodic_memory=None, semantic_memory=None, working_memory=None):
        """Initialize with memory subsystems"""
        self.episodic = episodic_memory or EpisodicMemory()
        self.semantic = semantic_memory or SemanticMemory()
        self.working = working_memory or WorkingMemory()
        
        # Integration metrics
        self.integration_level = 0.0
        self.last_memory_operation = None
        self.memory_operations_count = {
            'store': 0,
            'retrieve': 0,
            'consolidate': 0,
            'update': 0
        }
        
    def store_coherent_memory(self, experience):
        """Store memory across all systems coherently"""
        self.last_memory_operation = 'store'
        self.memory_operations_count['store'] += 1
        
        # Store in working memory first
        if isinstance(experience, str):
            # For text experiences, store with features
            result = self.working.update(
                experience, 
                item_type="text_experience", 
                features=self._extract_features(experience)
            )
        else:
            # For structured data
            result = self.working.update(experience)
        
        # Extract semantic facts for semantic memory
        semantic_facts = self._extract_semantic_facts(experience)
        
        for fact in semantic_facts:
            # Add each fact to semantic memory
            if 'subject' in fact and 'predicate' in fact:
                self.semantic.add_relation(
                    fact['subject'], 
                    fact['predicate'], 
                    fact.get('object', 'true'), 
                    fact.get('confidence', 1.0)
                )
        
        # Store in episodic memory
        importance = self._calculate_importance(experience, semantic_facts)
        self.episodic.store_episode(experience, importance=importance)
        
        return {
            'working_memory_index': result,
            'semantic_facts_extracted': len(semantic_facts),
            'importance': importance
        }
    
    def _extract_features(self, text_experience):
        """Extract features from text experience for working memory"""
        # Simple feature extraction
        words = text_experience.lower().split()
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only count non-trivial words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Find most frequent words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Create features dict
        features = {
            'length': len(words),
            'top_words': [word for word, _ in top_words],
            'timestamp': time.time()
        }
        
        # Check for emotional words
        emotional_words = ['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 
                          'joy', 'trust', 'anticipation']
        
        emotions_present = [word for word in emotional_words if word in text_experience.lower()]
        if emotions_present:
            features['emotions'] = emotions_present
        
        return features
    
    def _extract_semantic_facts(self, experience):
        """Extract factual knowledge for semantic memory"""
        facts = []
        
        if isinstance(experience, str):
            # Simple pattern matching for text
            import re
            
            # Basic relation patterns
            patterns = [
                # X is a Y
                (r"(\w+)\s+is\s+(?:a|an)\s+(\w+)", "is_a"),
                # X has Y
                (r"(\w+)\s+has\s+(\w+)", "has"),
                # X likes Y
                (r"(\w+)\s+(likes|loves|hates|fears|trusts)\s+(\w+)", None),
                # X is Y-property
                (r"(\w+)\s+is\s+(\w+)", "property")
            ]
            
            for pattern, relation_type in patterns:
                matches = re.finditer(pattern, experience.lower())
                
                for match in matches:
                    if relation_type is None:
                        # Use the matched verb as relation type
                        if len(match.groups()) >= 3:
                            subject = match.group(1)
                            relation = match.group(2)
                            obj = match.group(3)
                            
                            facts.append({
                                'subject': subject,
                                'predicate': relation,
                                'object': obj,
                                'confidence': 0.9
                            })
                    elif relation_type == "property" and len(match.groups()) >= 2:
                        # Property relation
                        subject = match.group(1)
                        property_value = match.group(2)
                        
                        # Skip common linking verbs
                        if property_value not in ['the', 'that', 'this', 'it', 'they', 'there']:
                            facts.append({
                                'subject': subject,
                                'predicate': 'is',
                                'object': property_value,
                                'confidence': 0.8
                            })
                    else:
                        # Fixed relation type with extracted subject/object
                        if len(match.groups()) >= 2:
                            if relation_type == "is_a":
                                subject = match.group(1)
                                obj = match.group(2)
                                
                                facts.append({
                                    'subject': subject,
                                    'predicate': relation_type,
                                    'object': obj,
                                    'confidence': 0.9
                                })
                            elif relation_type == "has":
                                subject = match.group(1)
                                obj = match.group(2)
                                
                                facts.append({
                                    'subject': subject,
                                    'predicate': relation_type,
                                    'object': obj,
                                    'confidence': 0.8
                                })
        elif isinstance(experience, dict):
            # Extract facts from structured data
            if all(k in experience for k in ['subject', 'predicate', 'object']):
                facts.append({
                    'subject': experience['subject'],
                    'predicate': experience['predicate'],
                    'object': experience['object'],
                    'confidence': experience.get('confidence', 0.9)
                })
            elif 'facts' in experience and isinstance(experience['facts'], list):
                facts.extend(experience['facts'])
        
        return facts
    
    def _calculate_importance(self, experience, semantic_facts):
        """Calculate importance based on semantic content and novelty"""
        # Base importance
        importance = 0.5
        
        # Adjust based on semantic richness
        if semantic_facts:
            importance += min(0.3, len(semantic_facts) * 0.05)
            
        # Adjust based on emotional content if text
        if isinstance(experience, str):
            emotional_words = ['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 
                              'joy', 'trust', 'anticipation']
            
            emotion_count = sum(word in experience.lower() for word in emotional_words)
            importance += min(0.2, emotion_count * 0.05)
        
        # Cap at 1.0
        return min(1.0, importance)
    
    def retrieve_integrated_knowledge(self, query, context=None):
        """Retrieve knowledge integrating across memory systems"""
        self.last_memory_operation = 'retrieve'
        self.memory_operations_count['retrieve'] += 1
        
        # First, check working memory for recency
        working_items = self._search_working_memory(query)
        
        # Then check episodic memory for experiences
        episodic_items = self._search_episodic_memory(query)
        
        # Finally check semantic memory for facts
        semantic_items = self._search_semantic_memory(query)
        
        # Combine and deduplicate results
        all_results = self._integrate_memory_results(working_items, episodic_items, semantic_items)
        
        # Update working memory with retrieved results
        for item in all_results[:3]:  # Only refresh top results
            if item['source'] != 'working':
                # Store in working memory for future access
                content = item['content']
                
                if isinstance(content, str):
                    self.working.update(content, item_type=f"{item['source']}_retrieval")
        
        # Calculate integration level based on diversity and connectivity of results
        self._update_integration_level(working_items, episodic_items, semantic_items)
        
        return {
            'query': query,
            'context': context,
            'results': all_results,
            'counts': {
                'working': len(working_items),
                'episodic': len(episodic_items),
                'semantic': len(semantic_items),
                'total': len(all_results)
            },
            'integration_level': self.integration_level
        }
    
    def _search_working_memory(self, query):
        """Search working memory for query"""
        results = []
        
        # Get all active items in working memory
        active_items = self.working.get_active_items()
        
        for item, activation in active_items:
            similarity = self._calculate_similarity(query, item)
            
            if similarity > 0.3:  # Low threshold for working memory
                results.append({
                    'content': item,
                    'relevance': similarity * activation,  # Scale by activation
                    'source': 'working',
                    'timestamp': time.time()
                })
                
        return results
    
    def _search_episodic_memory(self, query):
        """Search episodic memory for query"""
        # Use episodic memory's built-in retrieval
        episodes = self.episodic.retrieve_relevant(query)
        
        results = []
        for content, relevance in episodes:
            results.append({
                'content': content,
                'relevance': relevance,
                'source': 'episodic',
                'timestamp': time.time()
            })
            
        return results
    
    def _search_semantic_memory(self, query):
        """Search semantic memory for query"""
        results = []
        
        # First search for concepts
        concepts = self.semantic.retrieve_similar(query)
        
        for concept in concepts:
            results.append({
                'content': concept['name'],
                'relevance': 0.8,  # High relevance for concept matches
                'source': 'semantic',
                'concept': True,
                'properties': concept.get('properties', {})
            })
            
        # Then search for relations involving query terms
        if isinstance(query, str):
            query_terms = query.lower().split()
            
            for term in query_terms:
                # Search relations where term is subject
                relations = self.semantic.query_relations(subject=term)
                
                for relation in relations:
                    fact_text = f"{relation['subject']} {relation['predicate']} {relation['object']}"
                    results.append({
                        'content': fact_text,
                        'relevance': relation.get('confidence', 0.7),
                        'source': 'semantic',
                        'relation': relation
                    })
                    
                # Search relations where term is object
                relations = self.semantic.query_relations(object=term)
                
                for relation in relations:
                    fact_text = f"{relation['subject']} {relation['predicate']} {relation['object']}"
                    results.append({
                        'content': fact_text,
                        'relevance': relation.get('confidence', 0.7),
                        'source': 'semantic',
                        'relation': relation
                    })
                    
        return results
    
    def _integrate_memory_results(self, working_items, episodic_items, semantic_items):
        """Integrate and rank results from different memory systems"""
        # Combine all results
        all_results = working_items + episodic_items + semantic_items
        
        # Sort by relevance
        all_results.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Remove duplicates (prioritizing working memory, then episodic, then semantic)
        unique_results = []
        seen_content = set()
        
        # First add working memory items
        for item in working_items:
            content_key = str(item['content'])
            if content_key not in seen_content:
                unique_results.append(item)
                seen_content.add(content_key)
        
        # Then add episodic memory items
        for item in episodic_items:
            content_key = str(item['content'])
            if content_key not in seen_content:
                unique_results.append(item)
                seen_content.add(content_key)
        
        # Finally add semantic memory items
        for item in semantic_items:
            content_key = str(item['content'])
            if content_key not in seen_content:
                unique_results.append(item)
                seen_content.add(content_key)
        
        # Sort final list by relevance
        unique_results.sort(key=lambda x: x['relevance'], reverse=True)
        
        return unique_results
    
    def _update_integration_level(self, working_items, episodic_items, semantic_items):
        """Update integration level based on memory activity"""
        # Calculate integration based on distribution across memory systems
        system_counts = [
            len(working_items),
            len(episodic_items),
            len(semantic_items)
        ]
        
        active_systems = sum(1 for count in system_counts if count > 0)
        
        # Base integration level on active systems
        if active_systems == 0:
            self.integration_level = 0.0
        elif active_systems == 1:
            self.integration_level = 0.3  # Single system - low integration
        elif active_systems == 2:
            self.integration_level = 0.6  # Two systems - moderate integration
        else:
            self.integration_level = 0.9  # All systems - high integration
            
        # Adjust based on balance between systems
        total_items = sum(system_counts)
        if total_items > 0:
            balance = 1.0 - np.std([count/total_items for count in system_counts if total_items > 0])
            self.integration_level *= balance
            
        # Ensure reasonable bounds
        self.integration_level = max(0.1, min(1.0, self.integration_level))
    
    def _calculate_similarity(self, query, item):
        """Calculate similarity between query and memory item"""
        # Handle different item types
        if isinstance(query, str) and isinstance(item, str):
            return self._text_similarity(query, item)
        elif isinstance(query, dict) and isinstance(item, dict):
            return self._dict_similarity(query, item)
        elif isinstance(query, str) and isinstance(item, dict):
            # Compare string query with dict item
            if 'content' in item and isinstance(item['content'], str):
                return self._text_similarity(query, item['content'])
            else:
                return 0.3  # Default similarity for mixed types
        elif isinstance(query, dict) and isinstance(item, str):
            # Compare dict query with string item
            if 'content' in query and isinstance(query['content'], str):
                return self._text_similarity(query['content'], item)
            else:
                return 0.3  # Default similarity for mixed types
        else:
            # Default similarity for other types
            return 0.1
    
    def _text_similarity(self, text1, text2):
        """Calculate similarity between two text strings"""
        # Simple implementation - calculate word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _dict_similarity(self, dict1, dict2):
        """Calculate similarity between two dictionaries"""
        # Simple implementation - calculate key overlap and value similarity
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        
        if not keys1 or not keys2:
            return 0.0
            
        key_intersection = keys1.intersection(keys2)
        key_union = keys1.union(keys2)
        
        # Calculate key similarity
        key_similarity = len(key_intersection) / len(key_union)
        
        # Calculate value similarity for common keys
        value_similarities = []
        for key in key_intersection:
            value1 = dict1[key]
            value2 = dict2[key]
            
            if isinstance(value1, str) and isinstance(value2, str):
                value_similarities.append(self._text_similarity(value1, value2))
            elif type(value1) == type(value2):
                value_similarities.append(0.7)  # Same type, assume moderately similar
            else:
                value_similarities.append(0.3)  # Different types, assume low similarity
                
        # Combine key and value similarity
        if value_similarities:
            value_similarity = sum(value_similarities) / len(value_similarities)
            return (key_similarity + value_similarity) / 2
        else:
            return key_similarity
    
    def consolidate_memory(self):
        """Consolidate working memory to long-term storage"""
        self.last_memory_operation = 'consolidate'
        self.memory_operations_count['consolidate'] += 1
        
        # Get all items in working memory
        working_state = self.working.get_state()
        items = working_state.get('items', [])
        
        consolidated_count = 0
        
        for item in items:
            # Skip already consolidated items
            if item.get('consolidated', False):
                continue
                
            # Check if item should be consolidated based on activation and age
            activation = item.get('activation', 0)
            age = item.get('age', 0)
            importance = activation * (1 + min(1.0, age / 600))  # Age in seconds, capped at 10 minutes
            
            if importance > 0.7:
                # Consolidate to episodic and semantic memory
                content = item.get('item')
                
                if content:
                    # Store in episodic memory
                    self.episodic.store_episode(content, importance=activation)
                    
                    # Extract and store semantic facts
                    semantic_facts = self._extract_semantic_facts(content)
                    
                    for fact in semantic_facts:
                        if 'subject' in fact and 'predicate' in fact:
                            self.semantic.add_relation(
                                fact['subject'], 
                                fact['predicate'], 
                                fact.get('object', 'true'), 
                                fact.get('confidence', min(0.9, activation))
                            )
                    
                    # Mark as consolidated
                    item['consolidated'] = True
                    consolidated_count += 1
                    
        return {
            'consolidated_count': consolidated_count,
            'total_items': len(items)
        }
    
    def update_memory(self, item_identifier, new_content):
        """Update existing memory with new content"""
        self.last_memory_operation = 'update'
        self.memory_operations_count['update'] += 1
        
        # Check if item is in working memory first
        working_state = self.working.get_state()
        working_items = working_state.get('items', [])
        
        for item in working_items:
            # Check if this is the item to update
            if str(item.get('item')) == str(item_identifier):
                # Update in working memory
                self.working.update(new_content)
                
                # Also update in episodic memory
                # Note: We don't directly update episodic memory, but instead add a new episode
                # that links to the previous one
                importance = item.get('activation', 0.5)
                
                # Create update record
                update_record = {
                    'original_content': item_identifier,
                    'updated_content': new_content,
                    'update_type': 'modification',
                    'timestamp': time.time()
                }
                
                # Store the update record in episodic memory
                self.episodic.store_episode(update_record, importance=importance)
                
                # Update semantic facts if needed
                if isinstance(new_content, dict) and isinstance(item_identifier, dict):
                    # Check for fact updates
                    old_facts = self._extract_semantic_facts(item_identifier)
                    new_facts = self._extract_semantic_facts(new_content)
                    
                    # For facts that changed, update in semantic memory
                    self._update_semantic_facts(old_facts, new_facts)
                    
                return {
                    'success': True,
                    'updated_in': ['working', 'episodic', 'semantic'],
                    'item': new_content
                }
                
        # If not found in working memory, try to update in episodic memory
        # (Note: Episodic memory doesn't directly support updates, but we can add a new episode)
        episodes = self.episodic.retrieve_relevant(str(item_identifier))
        if episodes:
            # Create update record
            update_record = {
                'original_content': item_identifier,
                'updated_content': new_content,
                'update_type': 'modification',
                'timestamp': time.time()
            }
            
            # Store the update record in episodic memory
            self.episodic.store_episode(update_record, importance=0.7)
            
            # Update semantic facts if applicable
            if isinstance(new_content, dict):
                new_facts = self._extract_semantic_facts(new_content)
                
                for fact in new_facts:
                    if 'subject' in fact and 'predicate' in fact:
                        self.semantic.add_relation(
                            fact['subject'], 
                            fact['predicate'], 
                            fact.get('object', 'true'), 
                            fact.get('confidence', 0.8)
                        )
                
            return {
                'success': True,
                'updated_in': ['episodic', 'semantic'],
                'item': new_content
            }
        
        # Wasn't found in any memory system
        return {
            'success': False,
            'error': 'Item not found in memory systems'
        }
    
    def _update_semantic_facts(self, old_facts, new_facts):
        """Update semantic facts when memory content changes"""
        # Identify facts that have changed
        old_fact_keys = {(f['subject'], f['predicate'], f.get('object', 'true')) for f in old_facts}
        new_fact_keys = {(f['subject'], f['predicate'], f.get('object', 'true')) for f in new_facts}
        
        # Facts to remove (in old but not in new)
        to_remove = old_fact_keys - new_fact_keys
        
        # Facts to add (in new but not in old)
        to_add = new_fact_keys - old_fact_keys
        
        # Remove old facts (not directly supported in semantic memory, but we can add negations)
        for subj, pred, obj in to_remove:
            # Add negation with high confidence
            self.semantic.add_relation(
                subj, 
                f"not_{pred}", 
                obj, 
                confidence=0.9
            )
        
        # Add new facts
        for subj, pred, obj in to_add:
            matching_facts = [f for f in new_facts if 
                             f['subject'] == subj and 
                             f['predicate'] == pred and 
                             f.get('object', 'true') == obj]
            
            if matching_facts:
                confidence = matching_facts[0].get('confidence', 0.8)
                
                self.semantic.add_relation(
                    subj, 
                    pred, 
                    obj, 
                    confidence=confidence
                )
    
    def get_memory_stats(self):
        """Get statistics about memory system"""
        # Get working memory state
        working_state = self.working.get_state()
        
        return {
            'working_memory': {
                'capacity': working_state.get('capacity', 0),
                'used_slots': working_state.get('used_slots', 0),
                'active_items': len(self.working.get_active_items())
            },
            'episodic_memory': {
                'buffer_size': len(self.episodic.memory_buffer) if hasattr(self.episodic, 'memory_buffer') else 0,
                'recent_activations': len(self.episodic.recent_activations) if hasattr(self.episodic, 'recent_activations') else 0
            },
            'semantic_memory': {
                'concept_count': len(self.semantic.concepts) if hasattr(self.semantic, 'concepts') else 0,
                'relation_count': sum(len(relations) for relations in self.semantic.relations.values()) if hasattr(self.semantic, 'relations') else 0
            },
            'operations': self.memory_operations_count,
            'integration_level': self.integration_level,
            'last_operation': self.last_memory_operation
        }