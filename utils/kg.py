"""
Knowledge Graph utilities for the cognitive architecture.
Provides functions for manipulating, querying, and visualizing knowledge graphs.
"""

import re
import json
import networkx as nx
from collections import defaultdict

class KnowledgeGraphUtils:
    def __init__(self, prolog_engine=None):
        """
        Initialize the KG utilities
        
        Args:
            prolog_engine: PrologEngine instance (optional)
        """
        self.prolog = prolog_engine
        
    def extract_kg_triples(self, confidence_threshold=0.0):
        """
        Extract triples from Prolog knowledge base
        
        Args:
            confidence_threshold: Minimum confidence for included triples
            
        Returns:
            List of dicts with subject, predicate, object, confidence
        """
        if not self.prolog:
            return []
            
        triples = []
        
        try:
            # Query all facts from Prolog KB
            query = "confident_fact(P, S, O, C)"
            
            for result in self.prolog.prolog.query(query):
                # Extract components
                pred = str(result["P"])
                subj = str(result["S"])
                obj = str(result["O"])
                conf = float(result["C"])
                
                # Skip low-confidence triples
                if conf < confidence_threshold:
                    continue
                    
                # Add to results
                triple = {
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "confidence": conf
                }
                triples.append(triple)
                
        except Exception as e:
            print(f"[KGUtils] Error extracting KG triples: {e}")
            
        return triples
    
    def build_networkx_graph(self, triples=None, confidence_threshold=0.0):
        """
        Build a NetworkX graph from knowledge graph triples
        
        Args:
            triples: List of triple dicts (if None, extract from Prolog)
            confidence_threshold: Minimum confidence for included triples
            
        Returns:
            NetworkX DiGraph
        """
        G = nx.DiGraph()
        
        # If triples not provided, extract from Prolog
        if triples is None and self.prolog:
            triples = self.extract_kg_triples(confidence_threshold)
            
        if not triples:
            return G
            
        # Add nodes and edges to graph
        for triple in triples:
            subj = triple["subject"]
            pred = triple["predicate"]
            obj = triple["object"]
            conf = triple.get("confidence", 1.0)
            
            # Skip low-confidence triples
            if conf < confidence_threshold:
                continue
                
            # Add nodes if they don't exist
            if not G.has_node(subj):
                G.add_node(subj, type="entity")
                
            if not G.has_node(obj):
                G.add_node(obj, type="entity")
                
            # Add edge with predicate as relationship type
            G.add_edge(subj, obj, predicate=pred, weight=conf)
            
        return G
    
    def find_paths(self, source, target, max_length=3):
        """
        Find paths between two entities in the knowledge graph
        
        Args:
            source: Source entity
            target: Target entity
            max_length: Maximum path length
            
        Returns:
            List of paths (each path is a list of triples)
        """
        # Extract triples and build graph
        triples = self.extract_kg_triples()
        G = self.build_networkx_graph(triples)
        
        # If source or target not in graph, return empty list
        if source not in G or target not in G:
            return []
            
        paths = []
        
        # Use NetworkX to find simple paths
        for path in nx.all_simple_paths(G, source=source, target=target, cutoff=max_length):
            path_triples = []
            
            # Convert path to triples
            for i in range(len(path) - 1):
                s = path[i]
                o = path[i + 1]
                
                # Get edge data (might be multiple edges)
                edge_data = G.get_edge_data(s, o)
                
                if edge_data:
                # NetworkX may return edge data in different formats depending on multigraph settings
                # Handle both dictionary and dictionary-within-dictionary formats
                    if isinstance(edge_data, dict):
                        # Check if this is a multi-edge structure (dict of dicts)
                        if any(isinstance(v, dict) for v in edge_data.values()):
                            # Multi-edge format: process each edge
                            for edge_key, attrs in edge_data.items():
                                if isinstance(attrs, dict):
                                    pred = attrs.get("predicate", "unknown")
                                    conf = attrs.get("weight", 1.0)
                                    
                                    path_triples.append({
                                        "subject": s,
                                        "predicate": pred,
                                        "object": o,
                                        "confidence": conf
                                    })
                        else:
                            # Single edge format: edge_data is the attributes directly
                            pred = edge_data.get("predicate", "unknown")
                            conf = edge_data.get("weight", 1.0)
                            
                            path_triples.append({
                                "subject": s,
                                "predicate": pred,
                                "object": o,
                                "confidence": conf
                            })
                
            if path_triples:
                paths.append(path_triples)
        
        return paths
    
    def find_common_connections(self, entity1, entity2):
        """
        Find entities connected to both entity1 and entity2
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            Dict of common connections with their relationship types
        """
        # Extract triples and build graph
        triples = self.extract_kg_triples()
        G = self.build_networkx_graph(triples)
        
        # If either entity not in graph, return empty dict
        if entity1 not in G or entity2 not in G:
            return {}
            
        # Get neighbors of each entity
        neighbors1 = set(G.predecessors(entity1)).union(set(G.successors(entity1)))
        neighbors2 = set(G.predecessors(entity2)).union(set(G.successors(entity2)))
        
        # Find common neighbors
        common = neighbors1.intersection(neighbors2)
        
        # Build result with relationship information
        result = {}
        
        for common_entity in common:
            result[common_entity] = {
                "relationships_with_entity1": [],
                "relationships_with_entity2": []
            }
            
            # Get relationships with entity1
            if G.has_edge(entity1, common_entity):
                edge_data = G.get_edge_data(entity1, common_entity)
                for edge_key, edge_attrs in edge_data.items():
                    result[common_entity]["relationships_with_entity1"].append({
                        "direction": "outgoing",
                        "predicate": edge_attrs.get("predicate", "unknown"),
                        "confidence": edge_attrs.get("weight", 1.0)
                    })
                    
            if G.has_edge(common_entity, entity1):
                edge_data = G.get_edge_data(common_entity, entity1)
                for edge_key, edge_attrs in edge_data.items():
                    result[common_entity]["relationships_with_entity1"].append({
                        "direction": "incoming",
                        "predicate": edge_attrs.get("predicate", "unknown"),
                        "confidence": edge_attrs.get("weight", 1.0)
                    })
                    
            # Get relationships with entity2
            if G.has_edge(entity2, common_entity):
                edge_data = G.get_edge_data(entity2, common_entity)
                for edge_key, edge_attrs in edge_data.items():
                    result[common_entity]["relationships_with_entity2"].append({
                        "direction": "outgoing",
                        "predicate": edge_attrs.get("predicate", "unknown"),
                        "confidence": edge_attrs.get("weight", 1.0)
                    })
                    
            if G.has_edge(common_entity, entity2):
                edge_data = G.get_edge_data(common_entity, entity2)
                for edge_key, edge_attrs in edge_data.items():
                    result[common_entity]["relationships_with_entity2"].append({
                        "direction": "incoming",
                        "predicate": edge_attrs.get("predicate", "unknown"),
                        "confidence": edge_attrs.get("weight", 1.0)
                    })
                    
        return result
    
    def query_subgraph(self, query_pattern, max_hops=2):
        """
        Query for a subgraph matching a pattern
        
        Args:
            query_pattern: Dict with optional subject, predicate, object
            max_hops: Maximum distance from matched entities to include
            
        Returns:
            List of triples forming the subgraph
        """
        # Extract all triples
        all_triples = self.extract_kg_triples()
        
        # Find triples that match the query pattern
        matched_triples = []
        matched_entities = set()
        
        for triple in all_triples:
            match = True
            
            # Check each component of the pattern
            if "subject" in query_pattern and query_pattern["subject"] is not None:
                if triple["subject"] != query_pattern["subject"]:
                    match = False
                    
            if "predicate" in query_pattern and query_pattern["predicate"] is not None:
                if triple["predicate"] != query_pattern["predicate"]:
                    match = False
                    
            if "object" in query_pattern and query_pattern["object"] is not None:
                if triple["object"] != query_pattern["object"]:
                    match = False
                    
            if match:
                matched_triples.append(triple)
                matched_entities.add(triple["subject"])
                matched_entities.add(triple["object"])
                
        # If no matches, return empty list
        if not matched_triples:
            return []
            
        # If max_hops > 0, expand the subgraph
        if max_hops > 0:
            # Build graph from all triples
            G = self.build_networkx_graph(all_triples)
            
            # For each matched entity, find neighbors within max_hops
            expanded_entities = set(matched_entities)
            
            for entity in matched_entities:
                if entity in G:
                    # Find all nodes within max_hops
                    for target in G.nodes():
                        if entity != target:
                            # Check if path exists within max_hops
                            try:
                                path = nx.shortest_path(G, source=entity, target=target)
                                if len(path) - 1 <= max_hops:
                                    expanded_entities.add(target)
                            except nx.NetworkXNoPath:
                                # No path exists
                                pass
                                
            # Add all triples involving expanded entities
            expanded_triples = matched_triples.copy()
            
            for triple in all_triples:
                if triple not in expanded_triples:
                    if triple["subject"] in expanded_entities or triple["object"] in expanded_entities:
                        expanded_triples.append(triple)
                        
            return expanded_triples
            
        return matched_triples
    
    def find_entity_neighborhood(self, entity, max_depth=2):
        """
        Find all entities and relationships within N hops of an entity
        
        Args:
            entity: The central entity
            max_depth: Maximum number of hops
            
        Returns:
            Dict with nodes and edges of the neighborhood
        """
        # Extract triples and build graph
        triples = self.extract_kg_triples()
        G = self.build_networkx_graph(triples)
        
        if entity not in G:
            return {"nodes": [], "edges": []}
            
        # Find all nodes within max_depth
        neighborhood_nodes = set([entity])
        frontier = set([entity])
        
        for depth in range(max_depth):
            new_frontier = set()
            
            for node in frontier:
                # Add all neighbors
                neighbors = set(G.predecessors(node)).union(set(G.successors(node)))
                new_frontier.update(neighbors)
                
            # Update neighborhood and frontier
            neighborhood_nodes.update(new_frontier)
            frontier = new_frontier
            
        # Create subgraph
        subgraph = G.subgraph(neighborhood_nodes)
        
        # Convert to serializable format
        result = {
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        for node in subgraph.nodes():
            node_data = {
                "id": node,
                "label": node.replace("_", " ").title(),
                "type": subgraph.nodes[node].get("type", "entity")
            }
            result["nodes"].append(node_data)
            
        # Add edges
        for u, v, data in subgraph.edges(data=True):
            edge_data = {
                "source": u,
                "target": v,
                "label": data.get("predicate", ""),
                "confidence": data.get("weight", 1.0)
            }
            result["edges"].append(edge_data)
            
        return result
    
    def export_to_json(self, filename, triples=None):
        """
        Export knowledge graph to JSON format
        
        Args:
            filename: Output filename
            triples: Triple list (if None, extract from Prolog)
            
        Returns:
            Success flag
        """
        # If triples not provided, extract from Prolog
        if triples is None and self.prolog:
            triples = self.extract_kg_triples()
            
        if not triples:
            return False
            
        # Create graph structure for JSON
        graph = {
            "nodes": [],
            "edges": []
        }
        
        # Track nodes to avoid duplicates
        nodes = {}
        
        # Process triples
        for i, triple in enumerate(triples):
            subj = triple["subject"]
            pred = triple["predicate"]
            obj = triple["object"]
            conf = triple.get("confidence", 1.0)
            
            # Add subject node if not already added
            if subj not in nodes:
                nodes[subj] = {
                    "id": subj,
                    "label": subj.replace("_", " ").title(),
                    "type": "entity"
                }
                graph["nodes"].append(nodes[subj])
                
            # Add object node if not already added
            if obj not in nodes:
                nodes[obj] = {
                    "id": obj,
                    "label": obj.replace("_", " ").title(),
                    "type": "entity"
                }
                graph["nodes"].append(nodes[obj])
                
            # Add edge
            edge = {
                "id": f"e{i}",
                "source": subj,
                "target": obj,
                "label": pred,
                "confidence": conf
            }
            graph["edges"].append(edge)
            
        # Write to file
        try:
            with open(filename, 'w') as f:
                json.dump(graph, f, indent=2)
            return True
        except Exception as e:
            print(f"[KGUtils] Error exporting to JSON: {e}")
            return False
    
    def import_from_json(self, filename):
        """
        Import knowledge graph from JSON file
        
        Args:
            filename: Input JSON filename
            
        Returns:
            List of imported triples
        """
        triples = []
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            # Check if file has expected structure
            if "edges" not in data:
                print("[KGUtils] JSON file does not contain edges")
                return []
                
            # Create id-to-label mapping for nodes
            node_labels = {}
            if "nodes" in data:
                for node in data["nodes"]:
                    if "id" in node:
                        node_labels[node["id"]] = node.get("label", node["id"])
                        
            # Convert edges to triples
            for edge in data["edges"]:
                if "source" in edge and "target" in edge:
                    triple = {
                        "subject": edge["source"],
                        "predicate": edge.get("label", "relates_to"),
                        "object": edge["target"],
                        "confidence": edge.get("confidence", 1.0)
                    }
                    triples.append(triple)
                    
            # If Prolog engine available, assert triples
            if self.prolog and triples:
                for triple in triples:
                    try:
                        fact = f"confident_fact({triple['predicate']}, {triple['subject']}, {triple['object']}, {triple['confidence']})"
                        self.prolog.prolog.assertz(fact)
                    except Exception as e:
                        print(f"[KGUtils] Error asserting triple: {e}")
                        
        except Exception as e:
            print(f"[KGUtils] Error importing from JSON: {e}")
            
        return triples
    
    def calculate_centrality(self, centrality_type="degree"):
        """
        Calculate centrality measures for entities in the knowledge graph
        
        Args:
            centrality_type: Type of centrality to calculate
                             (degree, betweenness, closeness, or eigenvector)
            
        Returns:
            Dict mapping entities to centrality scores
        """
        # Extract triples and build graph
        triples = self.extract_kg_triples()
        G = self.build_networkx_graph(triples)
        
        if not G.nodes():
            return {}
            
        # Calculate specified centrality
        if centrality_type == "degree":
            centrality = nx.degree_centrality(G)
        elif centrality_type == "betweenness":
            centrality = nx.betweenness_centrality(G)
        elif centrality_type == "closeness":
            centrality = nx.closeness_centrality(G)
        elif centrality_type == "eigenvector":
            centrality = nx.eigenvector_centrality_numpy(G)
        else:
            # Default to degree centrality
            centrality = nx.degree_centrality(G)
            
        return centrality
    
    def extract_communities(self, algorithm="louvain"):
        """
        Identify communities of entities in the knowledge graph
        
        Args:
            algorithm: Community detection algorithm
                      (louvain, label_propagation, greedy_modularity)
            
        Returns:
            Dict mapping community IDs to lists of entities
        """
        try:
            # Import community detection algorithms
            import community as community_louvain
            from networkx.algorithms import community as nx_community
            
            # Extract triples and build graph
            triples = self.extract_kg_triples()
            
            # For community detection, convert to undirected graph
            G = nx.Graph()
            
            # Add nodes and edges
            for triple in triples:
                subj = triple["subject"]
                obj = triple["object"]
                weight = triple.get("confidence", 1.0)
                
                # Add edge with weight
                if G.has_edge(subj, obj):
                    # Update weight if existing edge
                    G[subj][obj]['weight'] = max(G[subj][obj]['weight'], weight)
                else:
                    G.add_edge(subj, obj, weight=weight)
                    
            # No edges in graph
            if not G.edges():
                return {}
                
            communities = {}
            
            # Apply specified algorithm
            if algorithm == "louvain":
                # Run Louvain community detection
                partition = community_louvain.best_partition(G)
                
                # Group by community
                for node, community_id in partition.items():
                    if community_id not in communities:
                        communities[community_id] = []
                    communities[community_id].append(node)
                    
            elif algorithm == "label_propagation":
                # Run label propagation
                community_iter = nx_community.label_propagation_communities(G)
                
                # Convert to dict format
                for i, community in enumerate(community_iter):
                    communities[i] = list(community)
                    
            elif algorithm == "greedy_modularity":
                # Run greedy modularity algorithm
                community_iter = nx_community.greedy_modularity_communities(G)
                
                # Convert to dict format
                for i, community in enumerate(community_iter):
                    communities[i] = list(community)
                    
            else:
                # Default to Louvain
                partition = community_louvain.best_partition(G)
                
                # Group by community
                for node, community_id in partition.items():
                    if community_id not in communities:
                        communities[community_id] = []
                    communities[community_id].append(node)
                    
            return communities
            
        except ImportError:
            print("[KGUtils] Community detection algorithms not available")
            print("[KGUtils] Install python-louvain package for community detection")
            return {}
            
        except Exception as e:
            print(f"[KGUtils] Error detecting communities: {e}")
            return {}