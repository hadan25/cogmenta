import re
from collections import defaultdict

class SemanticMemory:
    def __init__(self):
        self.concepts = {}  # Store concepts and their properties
        self.relations = defaultdict(list)  # Store relations between concepts
        self.category_members = defaultdict(list)  # Store category memberships
        
    def add_concept(self, concept_name, properties=None):
        """Add a concept to semantic memory"""
        concept_id = self._normalize_concept_name(concept_name)
        if concept_id not in self.concepts:
            self.concepts[concept_id] = {
                'name': concept_name,
                'properties': properties or {},
                'confidence': 1.0,
                'activation': 0.0
            }
        return concept_id
        
    def add_relation(self, subject, predicate, object, confidence=1.0):
        """Add a relation between concepts"""
        subj_id = self._normalize_concept_name(subject)
        obj_id = self._normalize_concept_name(object)
        
        # Ensure both concepts exist
        if subj_id not in self.concepts:
            self.add_concept(subject)
        if obj_id not in self.concepts:
            self.add_concept(object)
            
        # Add relation
        relation = {
            'subject': subj_id,
            'predicate': predicate,
            'object': obj_id,
            'confidence': confidence
        }
        
        # Check if relation already exists
        for existing in self.relations[subj_id]:
            if (existing['predicate'] == predicate and 
                existing['object'] == obj_id):
                # Update confidence if higher
                if confidence > existing['confidence']:
                    existing['confidence'] = confidence
                return
                
        # Add new relation
        self.relations[subj_id].append(relation)
        
        # Check for category membership relations
        if predicate == 'is_a' or predicate == 'instance_of':
            self.category_members[obj_id].append(subj_id)
    
    def retrieve_concept(self, concept_name):
        """Retrieve a concept by name"""
        concept_id = self._normalize_concept_name(concept_name)
        return self.concepts.get(concept_id)
    
    def query_relations(self, subject=None, predicate=None, object=None, min_confidence=0.0):
        """Query relations matching the specified pattern"""
        results = []
        
        # If subject is specified, only check its relations
        if subject:
            subj_id = self._normalize_concept_name(subject)
            if subj_id in self.relations:
                for relation in self.relations[subj_id]:
                    if self._relation_matches(relation, subject, predicate, object, min_confidence):
                        results.append(relation)
        else:
            # Check all relations
            for subj_id, relations in self.relations.items():
                for relation in relations:
                    if self._relation_matches(relation, subject, predicate, object, min_confidence):
                        results.append(relation)
                        
        return results
    
    def retrieve_similar(self, query_terms, limit=5):
        """Retrieve concepts similar to the query terms"""
        query_terms_set = set(self._preprocess_text(query_terms).split())
        
        scored_concepts = []
        for concept_id, concept in self.concepts.items():
            # Create a text representation of the concept
            concept_text = concept['name']
            
            # Add properties to the text
            for prop, value in concept.get('properties', {}).items():
                concept_text += f" {prop} {value}"
                
            # Add relations
            for relation in self.relations.get(concept_id, []):
                obj_name = self.concepts.get(relation['object'], {}).get('name', '')
                concept_text += f" {relation['predicate']} {obj_name}"
                
            # Calculate similarity score
            concept_terms = set(self._preprocess_text(concept_text).split())
            
            # Jaccard similarity
            intersection = len(query_terms_set.intersection(concept_terms))
            union = len(query_terms_set.union(concept_terms))
            
            similarity = intersection / union if union > 0 else 0
            if similarity > 0:
                scored_concepts.append((concept, similarity))
                
        # Sort by similarity and return top results
        scored_concepts.sort(key=lambda x: x[1], reverse=True)
        return [concept for concept, _ in scored_concepts[:limit]]
    
    def _normalize_concept_name(self, name):
        """Normalize concept name for storage/retrieval"""
        if not name:
            return "_unknown_"
        return re.sub(r'[^a-z0-9_]', '_', name.lower())
    
    def _preprocess_text(self, text):
        """Preprocess text for matching"""
        if not text:
            return ""
        # Convert to lowercase and remove punctuation
        return re.sub(r'[^\w\s]', '', text.lower())
    
    def _relation_matches(self, relation, subject, predicate, object, min_confidence):
        """Check if relation matches the specified pattern"""
        if relation['confidence'] < min_confidence:
            return False
            
        if subject and relation['subject'] != self._normalize_concept_name(subject):
            return False
            
        if predicate and relation['predicate'] != predicate:
            return False
            
        if object and relation['object'] != self._normalize_concept_name(object):
            return False
            
        return True