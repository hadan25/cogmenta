"""
Working Memory module for the cognitive architecture.
Implements a short-term, limited-capacity memory buffer with decay and activation dynamics.
"""

import time
import numpy as np
from collections import deque

class WorkingMemory:
    def __init__(self, capacity=7):
        """
        Initialize working memory with Miller's "magical number 7±2" capacity.
        
        Args:
            capacity: Maximum number of items that can be held in working memory
        """
        self.capacity = capacity
        self.buffer = []
        self.activation_levels = []
        self.creation_times = []
        
        self.decay_rate = 0.95  # Activation decay per time step
        self.minimum_activation = 0.2  # Threshold for removing items
        self.interference_factor = 0.1  # How much similar items interfere
        
        # Track focus of attention
        self.focus_index = None  # Index of item in focus
        
        # Item properties
        self.item_types = []  # Type of each item (e.g., "concept", "fact", "goal")
        self.item_features = []  # Features/properties of each item
        
    def update(self, item, item_type="generic", features=None, activation=1.0):
        """
        Add or update an item in working memory
        
        Args:
            item: The item to store
            item_type: Type of the item
            features: Dict of item features/properties
            activation: Initial activation level (0-1)
            
        Returns:
            Index of the item in working memory
        """
        features = features or {}
        current_time = time.time()
        
        # Check if item already exists
        if item in self.buffer:
            idx = self.buffer.index(item)
            self.activation_levels[idx] = activation
            self.focus_index = idx  # Set as focus of attention
            return idx
            
        # If we're at capacity, remove the least activated item
        if len(self.buffer) >= self.capacity:
            self._remove_least_activated()
            
        # Add the new item
        self.buffer.append(item)
        self.activation_levels.append(activation)
        self.creation_times.append(current_time)
        self.item_types.append(item_type)
        self.item_features.append(features)
        
        # Set as focus of attention
        self.focus_index = len(self.buffer) - 1
        
        return self.focus_index
    
    def _remove_least_activated(self):
        """Remove the item with the lowest activation level"""
        if not self.buffer:
            return
            
        # Find the least activated item
        min_idx = self.activation_levels.index(min(self.activation_levels))
        
        # Remove the item
        self.buffer.pop(min_idx)
        self.activation_levels.pop(min_idx)
        self.creation_times.pop(min_idx)
        self.item_types.pop(min_idx)
        self.item_features.pop(min_idx)
        
        # Update focus if necessary
        if self.focus_index is not None:
            if min_idx == self.focus_index:
                self.focus_index = None  # Focus was removed
            elif min_idx < self.focus_index:
                self.focus_index -= 1  # Adjust focus index
    
    def get_active_items(self, threshold=0.2):
        """
        Get items with activation above threshold
        
        Args:
            threshold: Minimum activation level
            
        Returns:
            List of (item, activation) tuples
        """
        active_items = []
        for i, item in enumerate(self.buffer):
            if self.activation_levels[i] >= threshold:
                active_items.append((item, self.activation_levels[i]))
                
        # Sort by activation (highest first)
        active_items.sort(key=lambda x: x[1], reverse=True)
        return active_items
    
    def decay(self, time_step=1.0):
        """
        Apply activation decay to all items
        
        Args:
            time_step: Time step for decay calculation
            
        Returns:
            Number of items removed due to low activation
        """
        # Calculate decay factor
        decay_factor = self.decay_rate ** time_step
        
        # Apply decay to all items
        for i in range(len(self.activation_levels)):
            # Focus of attention decays more slowly
            if i == self.focus_index:
                self.activation_levels[i] *= (decay_factor + 0.1)
            else:
                self.activation_levels[i] *= decay_factor
                
            # Apply interference effects
            for j in range(len(self.activation_levels)):
                if i != j:
                    similarity = self._calculate_similarity(i, j)
                    if similarity > 0.7:  # Only consider highly similar items
                        # High similarity causes interference (lower activation)
                        self.activation_levels[i] -= similarity * self.interference_factor
            
            # Ensure activation is within bounds
            self.activation_levels[i] = max(0.0, min(1.0, self.activation_levels[i]))
        
        # Remove items below minimum activation
        items_to_remove = []
        for i in range(len(self.activation_levels) - 1, -1, -1):
            if self.activation_levels[i] < self.minimum_activation:
                items_to_remove.append(i)
                
        # Remove items (in reverse order to avoid index shifting issues)
        for i in sorted(items_to_remove, reverse=True):
            self.buffer.pop(i)
            self.activation_levels.pop(i)
            self.creation_times.pop(i)
            self.item_types.pop(i)
            self.item_features.pop(i)
            
            # Update focus if necessary
            if self.focus_index is not None:
                if i == self.focus_index:
                    self.focus_index = None  # Focus was removed
                elif i < self.focus_index:
                    self.focus_index -= 1  # Adjust focus index
        
        return len(items_to_remove)
    
    def _calculate_similarity(self, idx1, idx2):
        """
        Calculate similarity between two items in working memory
        
        Args:
            idx1: Index of first item
            idx2: Index of second item
            
        Returns:
            Similarity score (0-1)
        """
        # If types are different, lower base similarity
        if self.item_types[idx1] != self.item_types[idx2]:
            base_similarity = 0.3
        else:
            base_similarity = 0.7
            
        # Compare features
        features1 = self.item_features[idx1]
        features2 = self.item_features[idx2]
        
        # If both have features, calculate Jaccard similarity
        if features1 and features2:
            keys1 = set(features1.keys())
            keys2 = set(features2.keys())
            
            # Calculate feature key overlap
            common_keys = keys1.intersection(keys2)
            all_keys = keys1.union(keys2)
            
            if all_keys:
                key_similarity = len(common_keys) / len(all_keys)
            else:
                key_similarity = 0
                
            # For common keys, calculate value similarity
            value_similarities = []
            for key in common_keys:
                val1 = features1[key]
                val2 = features2[key]
                
                # For numerical values
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        value_similarity = 1.0 - min(1.0, abs(val1 - val2) / max_val)
                    else:
                        value_similarity = 1.0
                # For string values
                elif isinstance(val1, str) and isinstance(val2, str):
                    if val1 == val2:
                        value_similarity = 1.0
                    else:
                        value_similarity = 0.3  # Default for different strings
                # For other types
                else:
                    value_similarity = 0.5 if val1 == val2 else 0.0
                    
                value_similarities.append(value_similarity)
                
            # Average value similarity
            if value_similarities:
                avg_value_similarity = sum(value_similarities) / len(value_similarities)
            else:
                avg_value_similarity = 0
                
            # Combined similarity
            feature_similarity = 0.5 * key_similarity + 0.5 * avg_value_similarity
        else:
            feature_similarity = 0.5  # Default if no features to compare
            
        # Combine base and feature similarity
        return 0.4 * base_similarity + 0.6 * feature_similarity
    
    def refresh(self, idx):
        """
        Refresh an item in working memory (boost activation and set as focus)
        
        Args:
            idx: Index of item to refresh
            
        Returns:
            Boolean indicating success
        """
        if 0 <= idx < len(self.buffer):
            self.activation_levels[idx] = 1.0  # Full activation
            self.focus_index = idx  # Set as focus
            return True
        return False
    
    def get_focused_item(self):
        """
        Get the item currently in focus
        
        Returns:
            (item, activation) tuple or None if no focus
        """
        if self.focus_index is not None and 0 <= self.focus_index < len(self.buffer):
            return (self.buffer[self.focus_index], self.activation_levels[self.focus_index])
        return None
    
    def shift_focus(self, direction=1):
        """
        Shift focus to the next or previous item
        
        Args:
            direction: 1 for next, -1 for previous
            
        Returns:
            New focused item or None
        """
        if not self.buffer:
            return None
            
        if self.focus_index is None:
            # If no current focus, set to first or last item depending on direction
            self.focus_index = 0 if direction > 0 else len(self.buffer) - 1
        else:
            # Shift focus in the specified direction
            self.focus_index = (self.focus_index + direction) % len(self.buffer)
            
        # Boost activation of newly focused item
        self.activation_levels[self.focus_index] = min(1.0, self.activation_levels[self.focus_index] + 0.2)
        
        return self.get_focused_item()
    
    def clear(self):
        """Clear working memory"""
        self.buffer = []
        self.activation_levels = []
        self.creation_times = []
        self.focus_index = None
        self.item_types = []
        self.item_features = []
    
    def get_state(self):
        """
        Get the full state of working memory
        
        Returns:
            Dict representing the current state
        """
        state = []
        for i, item in enumerate(self.buffer):
            state.append({
                "item": item,
                "type": self.item_types[i],
                "features": self.item_features[i],
                "activation": self.activation_levels[i],
                "age": time.time() - self.creation_times[i],
                "in_focus": (i == self.focus_index)
            })
            
        return {
            "capacity": self.capacity,
            "used_slots": len(self.buffer),
            "focus_index": self.focus_index,
            "items": state
        }