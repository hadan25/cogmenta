# cogmenta_core/cognitive/memory_manager.py

class MemoryManager:
    """
    Integrated memory management system.
    Coordinates between episodic, semantic, and working memory.
    """
    
    def __init__(self, episodic_memory, semantic_memory, working_memory=None):
        """
        Initialize the memory manager.
        
        Args:
            episodic_memory: EpisodicMemory instance
            semantic_memory: SemanticMemory instance
            working_memory: Optional WorkingMemory instance
        """
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.working = working_memory
        
        # For tracking memory operations
        self.store_count = 0
        self.retrieval_count = 0
        self.consolidation_count = 0
        
    def store(self, content, content_type="general"):
        """
        Store content in appropriate memory systems.
        
        Args:
            content: The content to store
            content_type: Type of content (input, result, fact, etc.)
            
        Returns:
            Storage success status
        """
        self.store_count += 1
        
        # Convert content to string if it's not already
        if not isinstance(content, str):
            content_str = str(content)
        else:
            content_str = content
            
        # Store in episodic memory
        importance = 0.5  # Default importance
        if content_type in ["input", "result"]:
            importance = 0.7
        elif content_type == "fact":
            importance = 0.8
        
        self.episodic.store_episode(content_str, importance=importance)
        
        # Store in semantic memory if it's a fact
        if content_type == "fact":
            # This would extract subject-predicate-object from the content
            # Simplified implementation
            if isinstance(content, dict) and "relations" in content:
                for relation in content["relations"]:
                    self.semantic.add_relation(
                        relation.get("subject", "unknown"),
                        relation.get("predicate", "related_to"),
                        relation.get("object", "unknown"),
                        relation.get("confidence", 0.8)
                    )
                    
        # Update working memory if available
        if self.working:
            # Features would be extracted from content
            features = {}
            self.working.update(content_str, content_type, features=features)
            
        return True
    
    def retrieve(self, query, memory_type="all"):
        """
        Retrieve information from memory systems.
        
        Args:
            query: The query to search for
            memory_type: Which memory to search (episodic, semantic, working, all)
            
        Returns:
            Dictionary of retrieved results
        """
        self.retrieval_count += 1
        results = {}
        
        if memory_type in ["episodic", "all"]:
            results["episodic"] = self.episodic.retrieve_relevant(query)
            
        if memory_type in ["semantic", "all"]:
            if isinstance(query, str):
                # Simple semantic query
                results["semantic"] = self.semantic.query_relations(subject=query)
            elif isinstance(query, dict):
                # Structured query
                results["semantic"] = self.semantic.query_relations(
                    subject=query.get("subject"),
                    predicate=query.get("predicate"),
                    object=query.get("object")
                )
                
        if memory_type in ["working", "all"] and self.working:
            results["working"] = self.working.get_active_items()
            
        return results
    
    def consolidate_memories(self):
        """
        Periodically consolidate memories, transferring from episodic to semantic.
        
        Returns:
            Number of facts extracted
        """
        self.consolidation_count += 1
        fact_count = 0
        
        # Get recent episodic memories
        episodes = self.episodic.retrieve_relevant("", limit=50)  # Get recent memories
        
        # Extract facts from episodes
        for episode_text, _ in episodes:
            # This would use a more sophisticated fact extraction method
            # For now, just a simple demonstration
            if "is a" in episode_text.lower():
                parts = episode_text.split("is a")
                if len(parts) == 2:
                    subject = parts[0].strip()
                    object = parts[1].strip()
                    self.semantic.add_relation(subject, "is_a", object, confidence=0.7)
                    fact_count += 1
            
        return fact_count