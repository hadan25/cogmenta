"""
Structured Meaning Map system for the Cogmenta Core.
Provides graph/tree-based representation of meaning beyond vector embeddings.
Includes role relationships, temporal structure, and logical constraints.
"""

import json
import re
import time
import numpy as np
from collections import defaultdict
import uuid

class StructuredMeaningMap:
    """
    Represents meanings as structured graph/tree objects rather than as flat vectors.
    Captures rich semantic structure including roles, temporal relations,
    causal connections, and logical constraints.
    """
    
    def __init__(self, concept_system=None):
        """
        Initialize the structured meaning map system.
        
        Args:
            concept_system: ConceptEmbeddingSystem instance (optional)
        """
        self.concept_system = concept_system
        
        # Core meaning structures
        self.meaning_graphs = {}  # id -> meaning graph
        self.propositions = {}  # id -> proposition
        self.relations = {}  # id -> relation
        
        # Frames represent typical situations with roles
        self.frames = {}  # frame name -> frame definition
        
        # Schema represent event sequences
        self.schemas = {}  # schema name -> schema definition
        
        # Index for efficient retrieval
        self.text_to_meaning = {}  # text -> meaning graph id
        self.entity_index = defaultdict(list)  # entity -> list of graph ids
        self.relation_index = defaultdict(list)  # relation type -> list of relation ids
        
        # Temporal and causal relations between meaning graphs
        self.temporal_relations = []  # list of (graph1, relation, graph2) tuples
        self.causal_relations = []  # list of (cause_graph, effect_graph, confidence) tuples
        
        # For tracking meaning graph history
        self.meaning_history = []  # list of meaning graph ids in creation order
    
    def create_meaning_graph(self, text, name=None):
        """
        Create a meaning graph from text.
        
        Args:
            text: Source text
            name: Optional name for the meaning graph
            
        Returns:
            Meaning graph id
        """
        # Generate unique ID
        graph_id = str(uuid.uuid4())
        
        # Use concept system to extract concepts if available
        concepts = []
        if self.concept_system:
            extracted = self.concept_system.extract_concepts_from_text(text)
            concepts = [(c, conf) for c, conf, _ in extracted]
        
        # Create basic meaning graph structure
        graph = {
            'id': graph_id,
            'name': name or f"graph_{graph_id[:8]}",
            'source_text': text,
            'creation_time': time.time(),
            'nodes': [],
            'edges': [],
            'propositions': [],
            'concepts': concepts
        }
        
        # Parse text into structured meaning components
        self._parse_text_to_meaning(text, graph)
        
        # Store the graph
        self.meaning_graphs[graph_id] = graph
        
        # Update indexes
        self.text_to_meaning[text] = graph_id
        for node in graph['nodes']:
            if node['type'] == 'entity':
                self.entity_index[node['name']].append(graph_id)
        
        # Add to history
        self.meaning_history.append(graph_id)
        
        return graph_id
    
    def _parse_text_to_meaning(self, text, graph):
        """
        Parse text into structured meaning components.
        
        Args:
            text: Text to parse
            graph: Meaning graph to update
        """
        # Extract entities (simplified approach)
        entities = self._extract_entities(text)
        
        # Extract relations
        relations = self._extract_relations(text)
        
        # Add entities as nodes
        for entity in entities:
            node_id = str(uuid.uuid4())
            node = {
                'id': node_id,
                'type': 'entity',
                'name': entity['text'].lower(),
                'original_text': entity['text'],
                'span': entity['span']
            }
            graph['nodes'].append(node)
            
        # Add relations as edges
        for relation in relations:
            # Find source and target nodes
            source_node = None
            target_node = None
            
            for node in graph['nodes']:
                if node['name'] == relation['subject'].lower():
                    source_node = node
                elif node['name'] == relation['object'].lower():
                    target_node = node
            
            # Create nodes if they don't exist yet
            if not source_node:
                node_id = str(uuid.uuid4())
                source_node = {
                    'id': node_id,
                    'type': 'entity',
                    'name': relation['subject'].lower(),
                    'original_text': relation['subject'],
                    'span': None
                }
                graph['nodes'].append(source_node)
                
            if not target_node:
                node_id = str(uuid.uuid4())
                target_node = {
                    'id': node_id,
                    'type': 'entity',
                    'name': relation['object'].lower(),
                    'original_text': relation['object'],
                    'span': None
                }
                graph['nodes'].append(target_node)
            
            # Create edge
            edge_id = str(uuid.uuid4())
            edge = {
                'id': edge_id,
                'type': 'relation',
                'relation_type': relation['predicate'],
                'source': source_node['id'],
                'target': target_node['id'],
                'negated': relation.get('negated', False),
                'confidence': relation.get('confidence', 0.9)
            }
            graph['edges'].append(edge)
            
            # Also create a proposition
            prop_id = str(uuid.uuid4())
            proposition = {
                'id': prop_id,
                'type': 'relation_proposition',
                'subject': source_node['id'],
                'predicate': relation['predicate'],
                'object': target_node['id'],
                'negated': relation.get('negated', False),
                'confidence': relation.get('confidence', 0.9)
            }
            graph['propositions'].append(proposition)
            self.propositions[prop_id] = proposition
            
            # Update relation index
            relation_id = edge_id
            self.relations[relation_id] = {
                'id': relation_id,
                'type': relation['predicate'],
                'source_graph': graph['id'],
                'source': source_node['id'],
                'target': target_node['id'],
                'negated': relation.get('negated', False),
                'confidence': relation.get('confidence', 0.9)
            }
            self.relation_index[relation['predicate']].append(relation_id)
        
        # Check for frame matches
        self._identify_frames(graph)
    
    def _extract_entities(self, text):
        """
        Extract entities from text.
        
        Args:
            text: Text to parse
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        
        # Simple entity extraction using capitalization
        for match in re.finditer(r'\b([A-Z][a-z]+)\b', text):
            entity_text = match.group(1)
            start, end = match.span(1)
            
            entities.append({
                'text': entity_text,
                'type': 'entity',
                'span': (start, end)
            })
        
        # Also look for specific entity indicators
        indicators = ['person', 'people', 'man', 'woman', 'boy', 'girl', 'child', 'company', 'organization']
        for indicator in indicators:
            pattern = rf'(?:the|a|an)\s+([a-z]+(?:\s+[a-z]+)?)\s+{indicator}'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity_text = match.group(1)
                start, end = match.span(1)
                
                entities.append({
                    'text': entity_text,
                    'type': 'entity',
                    'span': (start, end)
                })
        
        return entities
    
    def _extract_relations(self, text):
        """
        Extract relations from text.
        
        Args:
            text: Text to parse
            
        Returns:
            List of relation dictionaries
        """
        relations = []
        
        # Basic relation patterns
        # Format: subject verb object
        basic_patterns = [
            # X trusts/likes/knows Y
            r'(\w+)\s+(trusts|likes|loves|knows|fears|hates|avoids)\s+(\w+)',
            # X is Y
            r'(\w+)\s+is\s+(?:a|an|the)?\s*(\w+)',
            # X has Y
            r'(\w+)\s+has\s+(?:a|an|the)?\s*(\w+)'
        ]
        
        for pattern in basic_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) == 3:  # subject, predicate, object
                    subject, predicate, object_val = groups
                    relations.append({
                        'subject': subject,
                        'predicate': predicate.lower(),
                        'object': object_val,
                        'negated': False,
                        'confidence': 0.9
                    })
                elif len(groups) == 2:  # special case for "is" pattern
                    subject, object_val = groups
                    relations.append({
                        'subject': subject,
                        'predicate': 'is_a',
                        'object': object_val,
                        'negated': False,
                        'confidence': 0.9
                    })
        
        # Negation patterns
        neg_patterns = [
            # X does not trust/like/know Y
            r'(\w+)\s+(?:does|do|did)\s+not\s+(trust|like|love|know|fear|hate|avoid)\s+(\w+)',
            # X doesn't trust/like/know Y
            r'(\w+)\s+(?:doesn\'t|don\'t|didn\'t)\s+(trust|like|love|know|fear|hate|avoid)\s+(\w+)'
        ]
        
        for pattern in neg_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject, predicate, object_val = match.groups()
                relations.append({
                    'subject': subject,
                    'predicate': predicate.lower(),
                    'object': object_val,
                    'negated': True,
                    'confidence': 0.8
                })
        
        return relations
    
    def _identify_frames(self, graph):
        """
        Identify and apply semantic frames to a meaning graph.
        
        Args:
            graph: Meaning graph to analyze
            
        Returns:
            List of matched frame instances
        """
        matched_frames = []
        
        # Iterate through known frames
        for frame_name, frame in self.frames.items():
            # Try to match frame pattern to graph
            matches = self._match_frame(graph, frame)
            
            for match in matches:
                # Create frame instance
                frame_instance = {
                    'id': str(uuid.uuid4()),
                    'frame': frame_name,
                    'roles': match,
                    'confidence': 0.8
                }
                
                # Add to graph
                if 'frames' not in graph:
                    graph['frames'] = []
                    
                graph['frames'].append(frame_instance)
                matched_frames.append(frame_instance)
        
        return matched_frames
    
    def _match_frame(self, graph, frame):
        """
        Match a frame pattern to a meaning graph.
        
        Args:
            graph: Meaning graph to match against
            frame: Frame definition
            
        Returns:
            List of role mappings (dict of role -> node_id)
        """
        matches = []
        
        # Get frame requirements
        required_relations = frame.get('relations', [])
        
        if not required_relations:
            return []
            
        # Check each proposition in the graph
        for prop in graph['propositions']:
            if prop['type'] != 'relation_proposition':
                continue
                
            # For each required relation in the frame
            for req_rel in required_relations:
                if prop['predicate'] == req_rel['predicate']:
                    # Potential match, map roles
                    roles = {}
                    
                    # Map subject to role
                    subj_role = req_rel.get('subject_role')
                    if subj_role:
                        roles[subj_role] = prop['subject']
                        
                    # Map object to role
                    obj_role = req_rel.get('object_role')
                    if obj_role:
                        roles[obj_role] = prop['object']
                        
                    # If we mapped all required roles, it's a match
                    if len(roles) >= len(set(r.get('subject_role', '') for r in required_relations) | 
                                      set(r.get('object_role', '') for r in required_relations)):
                        matches.append(roles)
        
        return matches
    
    def create_frame(self, name, description, roles, relations):
        """
        Create a semantic frame (situation with roles).
        
        Args:
            name: Frame name
            description: Frame description
            roles: List of role dictionaries (name, description)
            relations: List of relation requirements
            
        Returns:
            Frame definition
        """
        frame = {
            'name': name,
            'description': description,
            'roles': roles,
            'relations': relations
        }
        
        self.frames[name] = frame
        return frame
    
    def create_schema(self, name, description, events):
        """
        Create an event schema (sequence of events).
        
        Args:
            name: Schema name
            description: Schema description
            events: List of event dictionaries (name, description, preconditions, results)
            
        Returns:
            Schema definition
        """
        schema = {
            'name': name,
            'description': description,
            'events': events
        }
        
        self.schemas[name] = schema
        return schema
    
    def add_temporal_relation(self, graph1_id, relation, graph2_id, confidence=0.8):
        """
        Add a temporal relation between meaning graphs.
        
        Args:
            graph1_id: First graph ID
            relation: Temporal relation (before, after, during, etc.)
            graph2_id: Second graph ID
            confidence: Confidence value (0-1)
            
        Returns:
            Relation ID
        """
        # Check if graphs exist
        if graph1_id not in self.meaning_graphs or graph2_id not in self.meaning_graphs:
            raise ValueError("One or both graph IDs do not exist")
            
        # Create relation ID
        relation_id = str(uuid.uuid4())
        
        # Create relation record
        temporal_rel = {
            'id': relation_id,
            'type': 'temporal',
            'relation': relation,
            'graph1': graph1_id,
            'graph2': graph2_id,
            'confidence': confidence
        }
        
        # Add to relations
        self.temporal_relations.append(temporal_rel)
        
        return relation_id
    
    def add_causal_relation(self, cause_id, effect_id, confidence=0.8):
        """
        Add a causal relation between meaning graphs.
        
        Args:
            cause_id: Cause graph ID
            effect_id: Effect graph ID
            confidence: Confidence value (0-1)
            
        Returns:
            Relation ID
        """
        # Check if graphs exist
        if cause_id not in self.meaning_graphs or effect_id not in self.meaning_graphs:
            raise ValueError("One or both graph IDs do not exist")
            
        # Create relation ID
        relation_id = str(uuid.uuid4())
        
        # Create relation record
        causal_rel = {
            'id': relation_id,
            'type': 'causal',
            'cause': cause_id,
            'effect': effect_id,
            'confidence': confidence
        }
        
        # Add to relations
        self.causal_relations.append(causal_rel)
        
        return relation_id
    
    def query_by_entity(self, entity_name):
        """
        Find meaning graphs containing a specific entity.
        
        Args:
            entity_name: Name of the entity to search for
            
        Returns:
            List of meaning graph dictionaries
        """
        entity_name = entity_name.lower()
        
        # Get graph IDs from index
        graph_ids = self.entity_index.get(entity_name, [])
        
        # Retrieve full graphs
        return [self.meaning_graphs[gid] for gid in graph_ids if gid in self.meaning_graphs]
    
    def query_by_relation(self, relation_type, subject=None, object=None):
        """
        Find relations of a specific type.
        
        Args:
            relation_type: Type of relation to search for
            subject: Optional subject entity (filter)
            object: Optional object entity (filter)
            
        Returns:
            List of relation dictionaries
        """
        # Get relation IDs from index
        relation_ids = self.relation_index.get(relation_type, [])
        
        # Retrieve full relations
        relations = [self.relations[rid] for rid in relation_ids if rid in self.relations]
        
        # Apply filters if provided
        if subject:
            subject_lower = subject.lower()
            # Filter by subject
            relations = [r for r in relations if self._get_node_name(r['source']) == subject_lower]
            
        if object:
            object_lower = object.lower()
            # Filter by object
            relations = [r for r in relations if self._get_node_name(r['target']) == object_lower]
            
        return relations
    
    def _get_node_name(self, node_id):
        """
        Get entity name for a node ID.
        
        Args:
            node_id: Node ID to look up
            
        Returns:
            Entity name or None if not found
        """
        # Find the graph containing this node
        for graph_id, graph in self.meaning_graphs.items():
            for node in graph['nodes']:
                if node['id'] == node_id:
                    return node.get('name')
                    
        return None
    
    def find_similar_meanings(self, graph_id, threshold=0.7):
        """
        Find meaning graphs similar to the given one.
        
        Args:
            graph_id: ID of the meaning graph to compare
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of (graph_id, similarity_score) tuples
        """
        if graph_id not in self.meaning_graphs:
            return []
            
        source_graph = self.meaning_graphs[graph_id]
        results = []
        
        # Get entities in source graph
        source_entities = set()
        for node in source_graph['nodes']:
            if node['type'] == 'entity':
                source_entities.add(node['name'])
                
        # Get relations in source graph
        source_relations = []
        for edge in source_graph['edges']:
            if edge['type'] == 'relation':
                src_node = self._get_node_name(edge['source'])
                tgt_node = self._get_node_name(edge['target'])
                if src_node and tgt_node:
                    source_relations.append((src_node, edge['relation_type'], tgt_node))
        
        # Compare with all other graphs
        for other_id, other_graph in self.meaning_graphs.items():
            if other_id == graph_id:
                continue
                
            # Get entities in other graph
            other_entities = set()
            for node in other_graph['nodes']:
                if node['type'] == 'entity':
                    other_entities.add(node['name'])
                    
            # Entity overlap score
            entity_overlap = len(source_entities.intersection(other_entities))
            entity_union = len(source_entities.union(other_entities))
            entity_score = entity_overlap / entity_union if entity_union > 0 else 0
            
            # Get relations in other graph
            other_relations = []
            for edge in other_graph['edges']:
                if edge['type'] == 'relation':
                    src_node = self._get_node_name(edge['source'])
                    tgt_node = self._get_node_name(edge['target'])
                    if src_node and tgt_node:
                        other_relations.append((src_node, edge['relation_type'], tgt_node))
            
            # Relation overlap score
            relation_overlap = len(set(source_relations).intersection(set(other_relations)))
            relation_union = len(set(source_relations).union(set(other_relations)))
            relation_score = relation_overlap / relation_union if relation_union > 0 else 0
            
            # Combined similarity score
            similarity = 0.6 * entity_score + 0.4 * relation_score
            
            # Add to results if above threshold
            if similarity >= threshold:
                results.append((other_id, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def merge_meaning_graphs(self, graph_ids, name=None):
        """
        Merge multiple meaning graphs into a single unified graph.
        
        Args:
            graph_ids: List of graph IDs to merge
            name: Optional name for the merged graph
            
        Returns:
            ID of the merged graph
        """
        if not graph_ids:
            return None
            
        # Check if all graphs exist
        for gid in graph_ids:
            if gid not in self.meaning_graphs:
                raise ValueError(f"Graph ID {gid} does not exist")
                
        # Create a new graph
        merged_id = str(uuid.uuid4())
        merged_name = name or f"merged_{merged_id[:8]}"
        
        # Start with an empty graph
        merged_graph = {
            'id': merged_id,
            'name': merged_name,
            'source_text': "",
            'creation_time': time.time(),
            'nodes': [],
            'edges': [],
            'propositions': [],
            'concepts': [],
            'merged_from': graph_ids
        }
        
        # Track node ID mappings
        node_id_map = {}  # old_id -> new_id
        
        # First pass: collect source texts and merge nodes
        for gid in graph_ids:
            graph = self.meaning_graphs[gid]
            
            # Concatenate source texts
            if merged_graph['source_text']:
                merged_graph['source_text'] += " " + graph['source_text']
            else:
                merged_graph['source_text'] = graph['source_text']
            
            # Add concepts
            merged_graph['concepts'].extend(graph.get('concepts', []))
            
            # Process nodes
            for node in graph['nodes']:
                # Check if this entity already exists in the merged graph
                if node['type'] == 'entity':
                    existing_node = None
                    for n in merged_graph['nodes']:
                        if n['type'] == 'entity' and n['name'] == node['name']:
                            existing_node = n
                            break
                    
                    if existing_node:
                        # Map old ID to existing ID
                        node_id_map[node['id']] = existing_node['id']
                    else:
                        # Create new node with new ID
                        new_id = str(uuid.uuid4())
                        node_id_map[node['id']] = new_id
                        
                        new_node = node.copy()
                        new_node['id'] = new_id
                        merged_graph['nodes'].append(new_node)
                else:
                    # For non-entity nodes, always create a new one
                    new_id = str(uuid.uuid4())
                    node_id_map[node['id']] = new_id
                    
                    new_node = node.copy()
                    new_node['id'] = new_id
                    merged_graph['nodes'].append(new_node)
        
        # Second pass: merge edges and propositions
        for gid in graph_ids:
            graph = self.meaning_graphs[gid]
            
            # Process edges
            for edge in graph['edges']:
                # Create new edge with mapped IDs
                new_edge = edge.copy()
                new_edge['id'] = str(uuid.uuid4())
                
                # Map source and target IDs
                if edge['source'] in node_id_map:
                    new_edge['source'] = node_id_map[edge['source']]
                if edge['target'] in node_id_map:
                    new_edge['target'] = node_id_map[edge['target']]
                    
                merged_graph['edges'].append(new_edge)
            
            # Process propositions
            for prop in graph.get('propositions', []):
                # Create new proposition with mapped IDs
                new_prop = prop.copy()
                new_prop['id'] = str(uuid.uuid4())
                
                # Map subject and object IDs
                if 'subject' in prop and prop['subject'] in node_id_map:
                    new_prop['subject'] = node_id_map[prop['subject']]
                if 'object' in prop and prop['object'] in node_id_map:
                    new_prop['object'] = node_id_map[prop['object']]
                    
                merged_graph['propositions'].append(new_prop)
                self.propositions[new_prop['id']] = new_prop
        
        # Store the merged graph
        self.meaning_graphs[merged_id] = merged_graph
        
        # Update indexes
        for node in merged_graph['nodes']:
            if node['type'] == 'entity':
                self.entity_index[node['name']].append(merged_id)
        
        # Add to history
        self.meaning_history.append(merged_id)
        
        return merged_id
    
    def extract_conceptual_schema(self, graph_id):
        """
        Extract a higher-level conceptual schema from a meaning graph.
        
        Args:
            graph_id: ID of the meaning graph
            
        Returns:
            Conceptual schema dictionary
        """
        if graph_id not in self.meaning_graphs:
            return None
            
        graph = self.meaning_graphs[graph_id]
        
        # Create a higher-level schematic representation
        schema = {
            'id': str(uuid.uuid4()),
            'source_graph': graph_id,
            'entities': [],
            'relations': [],
            'patterns': []
        }
        
        # Extract entity types
        entity_types = {}
        for node in graph['nodes']:
            if node['type'] == 'entity':
                entity_type = self._infer_entity_type(node['name'])
                entity_types[node['id']] = entity_type
                
                schema['entities'].append({
                    'name': node['name'],
                    'type': entity_type
                })
        
        # Extract relation patterns
        for edge in graph['edges']:
            if edge['type'] == 'relation':
                source_type = entity_types.get(edge['source'], 'entity')
                target_type = entity_types.get(edge['target'], 'entity')
                
                schema['relations'].append({
                    'relation': edge['relation_type'],
                    'source_type': source_type,
                    'target_type': target_type,
                    'negated': edge.get('negated', False)
                })
        
        # Look for recurring patterns
        self._extract_patterns(graph, schema)
        
        return schema
    
    def _infer_entity_type(self, entity_name):
        """
        Infer the type of an entity based on name and context.
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            Inferred entity type
        """
        # Use concept system if available
        if self.concept_system:
            concept, _ = self.concept_system.get_concept_from_text(entity_name)
            if concept:
                # Check concept hierarchy
                paths = self.concept_system.get_concept_hierarchy_paths(concept)
                
                # Look for recognizable types in paths
                for path in paths:
                    for ancestor in path:
                        if ancestor in ['person', 'human', 'organization', 'company', 'location', 'place', 'object', 'event', 'time']:
                            return ancestor
        
        # Simple pattern-based type inference
        entity_lower = entity_name.lower()
        
        # Check for person names (capitalized)
        if entity_name[0].isupper() and not any(char.isdigit() for char in entity_name):
            return 'person'
            
        # Check for organizations
        if any(term in entity_lower for term in ['inc', 'corp', 'company', 'organization', 'foundation']):
            return 'organization'
            
        # Check for locations
        if any(term in entity_lower for term in ['street', 'road', 'avenue', 'city', 'town', 'country', 'state', 'place']):
            return 'location'
            
        # Default type
        return 'entity'
    
    def _extract_patterns(self, graph, schema):
        """
        Extract recurring patterns from a meaning graph.
        
        Args:
            graph: Source meaning graph
            schema: Schema to update with patterns
        """
        # Look for common relation patterns
        
        # Trust pattern: A trusts B, B trusts A
        trust_relations = []
        for edge in graph['edges']:
            if edge['type'] == 'relation' and edge['relation_type'] == 'trusts':
                trust_relations.append((edge['source'], edge['target']))
        
        # Mutual trust pattern
        mutual_trust_pairs = []
        for src, tgt in trust_relations:
            if (tgt, src) in trust_relations:
                pair = tuple(sorted([src, tgt]))
                if pair not in mutual_trust_pairs:
                    mutual_trust_pairs.append(pair)
        
        if mutual_trust_pairs:
            schema['patterns'].append({
                'name': 'mutual_trust',
                'description': 'Entities that trust each other',
                'instances': mutual_trust_pairs
            })
        
        # Trust chain pattern: A trusts B, B trusts C
        trust_chains = []
        for src1, tgt1 in trust_relations:
            for src2, tgt2 in trust_relations:
                if tgt1 == src2:
                    trust_chains.append((src1, tgt1, tgt2))
        
        if trust_chains:
            schema['patterns'].append({
                'name': 'trust_chain',
                'description': 'Chain of trust relationships',
                'instances': trust_chains
            })
    
    def derive_meaning_implications(self, graph_id):
        """
        Derive logical implications from a meaning graph.
        
        Args:
            graph_id: ID of the meaning graph
            
        Returns:
            List of derived propositions
        """
        if graph_id not in self.meaning_graphs:
            return []
            
        graph = self.meaning_graphs[graph_id]
        derived = []
        
        # IMPLICATION: If A trusts B and B trusts C, then A might trust C (with lower confidence)
        trust_relations = {}
        
        # Collect trust relations
        for prop in graph['propositions']:
            if prop['type'] == 'relation_proposition' and prop['predicate'] == 'trusts' and not prop['negated']:
                subject = prop['subject']
                object_val = prop['object']
                confidence = prop['confidence']
                
                if subject not in trust_relations:
                    trust_relations[subject] = []
                    
                trust_relations[subject].append((object_val, confidence))
        
        # Find transitive trust relations
        for subject, trusts in trust_relations.items():
            for trusted, conf1 in trusts:
                if trusted in trust_relations:
                    for trusted_trusts, conf2 in trust_relations[trusted]:
                        # Skip self-trust
                        if trusted_trusts == subject:
                            continue
                            
                        # Check if this connection already exists directly
                        direct_exists = False
                        for direct_trusted, _ in trusts:
                            if direct_trusted == trusted_trusts:
                                direct_exists = True
                                break
                                
                        if not direct_exists:
                            # Derive new transitive trust with lower confidence
                            transitive_conf = conf1 * conf2 * 0.8
                            
                            if transitive_conf >= 0.5:  # Only add if reasonable confidence
                                # Create new proposition
                                prop_id = str(uuid.uuid4())
                                new_prop = {
                                    'id': prop_id,
                                    'type': 'derived_proposition',
                                    'subject': subject,
                                    'predicate': 'trusts',
                                    'object': trusted_trusts,
                                    'negated': False,
                                    'confidence': transitive_conf,
                                    'derived_from': ['transitive_trust'],
                                    'source_graph': graph_id
                                }
                                
                                derived.append(new_prop)
                                self.propositions[prop_id] = new_prop
        
        # IMPLICATION: If A likes B and B likes C, then A might like C (with lower confidence)
        # Similar pattern as above
        like_relations = {}
        
        # Collect like relations
        for prop in graph['propositions']:
            if prop['type'] == 'relation_proposition' and prop['predicate'] == 'likes' and not prop['negated']:
                subject = prop['subject']
                object_val = prop['object']
                confidence = prop['confidence']
                
                if subject not in like_relations:
                    like_relations[subject] = []
                    
                like_relations[subject].append((object_val, confidence))
        
        # Find transitive like relations
        for subject, likes in like_relations.items():
            for liked, conf1 in likes:
                if liked in like_relations:
                    for liked_likes, conf2 in like_relations[liked]:
                        # Skip self-like
                        if liked_likes == subject:
                            continue
                            
                        # Check if this connection already exists directly
                        direct_exists = False
                        for direct_liked, _ in likes:
                            if direct_liked == liked_likes:
                                direct_exists = True
                                break
                                
                        if not direct_exists:
                            # Derive new transitive like with lower confidence
                            transitive_conf = conf1 * conf2 * 0.6  # Even lower confidence than trust
                            
                            if transitive_conf >= 0.5:  # Only add if reasonable confidence
                                # Create new proposition
                                prop_id = str(uuid.uuid4())
                                new_prop = {
                                    'id': prop_id,
                                    'type': 'derived_proposition',
                                    'subject': subject,
                                    'predicate': 'likes',
                                    'object': liked_likes,
                                    'negated': False,
                                    'confidence': transitive_conf,
                                    'derived_from': ['transitive_like'],
                                    'source_graph': graph_id
                                }
                                
                                derived.append(new_prop)
                                self.propositions[prop_id] = new_prop
        
        return derived
        
    def save_to_file(self, filename):
        """
        Save the structured meaning map system to a file.
        
        Args:
            filename: Target filename
            
        Returns:
            Success flag
        """
        try:
            # Prepare data dictionary
            data = {
                "meaning_graphs": self.meaning_graphs,
                "frames": self.frames,
                "schemas": self.schemas,
                "temporal_relations": self.temporal_relations,
                "causal_relations": self.causal_relations
            }
            
            # Save as JSON
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            return True
            
        except Exception as e:
            print(f"Error saving meaning system: {e}")
            return False
    
    def load_from_file(self, filename):
        """
        Load the structured meaning map system from a file.
        
        Args:
            filename: Source filename
            
        Returns:
            Success flag
        """
        try:
            # Load JSON data
            with open(filename, 'r') as f:
                data = json.load(f)
                
            # Reset current state
            self.meaning_graphs = {}
            self.propositions = {}
            self.relations = {}
            self.frames = {}
            self.schemas = {}
            self.text_to_meaning = {}
            self.entity_index = defaultdict(list)
            self.relation_index = defaultdict(list)
            self.temporal_relations = []
            self.causal_relations = []
            
            # Load data
            self.meaning_graphs = data.get("meaning_graphs", {})
            self.frames = data.get("frames", {})
            self.schemas = data.get("schemas", {})
            self.temporal_relations = data.get("temporal_relations", [])
            self.causal_relations = data.get("causal_relations", [])
            
            # Rebuild indexes
            for graph_id, graph in self.meaning_graphs.items():
                # Add source text to index
                if 'source_text' in graph:
                    self.text_to_meaning[graph['source_text']] = graph_id
                    
                # Add entities to index
                for node in graph.get('nodes', []):
                    if node.get('type') == 'entity' and 'name' in node:
                        self.entity_index[node['name']].append(graph_id)
                        
                # Add propositions to propositions dict
                for prop in graph.get('propositions', []):
                    if 'id' in prop:
                        self.propositions[prop['id']] = prop
                        
                # Add relations to index
                for edge in graph.get('edges', []):
                    if edge.get('type') == 'relation' and 'relation_type' in edge:
                        relation_id = edge['id']
                        relation_type = edge['relation_type']
                        
                        self.relations[relation_id] = {
                            'id': relation_id,
                            'type': relation_type,
                            'source_graph': graph_id,
                            'source': edge['source'],
                            'target': edge['target'],
                            'negated': edge.get('negated', False),
                            'confidence': edge.get('confidence', 0.9)
                        }
                        
                        self.relation_index[relation_type].append(relation_id)
            
            return True
            
        except Exception as e:
            print(f"Error loading meaning system: {e}")
            return False