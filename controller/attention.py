"""
Attention mechanisms for the cognitive architecture.
Manages focus of attention, salience detection, and attentional control.
"""

import time
import numpy as np
from collections import deque

class AttentionMechanism:
    def __init__(self):
        """Initialize the attention mechanism"""
        # Salience weights for different features
        self.salience_weights = {
            "novelty": 0.4,       # Preference for new stimuli
            "relevance": 0.3,     # Relevance to current goals or context
            "intensity": 0.2,     # Signal strength or intensity
            "emotional": 0.1      # Emotional significance
        }
        
        # Current attentional state
        self.current_focus = None  # Currently attended item
        self.focus_duration = 0    # Time spent on current focus
        self.focus_history = deque(maxlen=10)  # Recent attentional targets
        
        # Attentional parameters
        self.focus_threshold = 0.6  # Minimum salience to capture attention
        self.habituation_rate = 0.1  # Rate of habituation to current stimulus
        self.recovery_rate = 0.05   # Rate of recovery from habituation
        self.switch_cost = 0.2      # Cost of switching attention
        
        # Internal state
        self.last_update_time = time.time()
        self.habituated_items = {}  # Track habituation to items
        self.goal_relevance = {}    # Track relevance to current goals
        
    def calculate_salience(self, items, item_properties=None):
        """
        Calculate salience scores for items based on their properties
        
        Args:
            items: List of items to evaluate
            item_properties: Dict mapping items to their properties
            
        Returns:
            List of (item, salience_score) tuples, sorted by salience
        """
        item_properties = item_properties or {}
        scored_items = []
        current_time = time.time()
        
        # Process each item
        for item in items:
            # Default properties if not provided
            properties = item_properties.get(item, {})
            
            # Calculate novelty based on recency in focus history
            novelty = 1.0
            for i, past_focus in enumerate(self.focus_history):
                if past_focus == item:
                    # More recent appearances reduce novelty more
                    recency_factor = (len(self.focus_history) - i) / len(self.focus_history)
                    novelty *= (1 - recency_factor * 0.8)
                    break
            
            # Apply habituation effect
            if item in self.habituated_items:
                habituation, last_seen = self.habituated_items[item]
                # Recovery based on time since last seen
                time_factor = min(1.0, (current_time - last_seen) * self.recovery_rate)
                habituation = max(0, habituation - time_factor)
                novelty *= (1 - habituation)
                
            # Calculate intensity from properties or default
            intensity = properties.get('intensity', 0.5)
            
            # Calculate emotional significance
            emotional = properties.get('emotional_value', 0.0)
            
            # Calculate relevance to current goals
            relevance = self.goal_relevance.get(item, properties.get('relevance', 0.5))
            
            # Switch cost if this would be a focus change
            switch_penalty = 0
            if self.current_focus is not None and item != self.current_focus:
                switch_penalty = self.switch_cost
            
            # Compute overall salience score
            salience = (
                self.salience_weights["novelty"] * novelty +
                self.salience_weights["intensity"] * intensity +
                self.salience_weights["emotional"] * emotional +
                self.salience_weights["relevance"] * relevance -
                switch_penalty
            )
            
            scored_items.append((item, salience))
        
        # Sort by salience score (highest first)
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items
    
    def update_focus(self, salient_items, forced=None, item_properties=None):
        """
        Update the focus of attention
        
        Args:
            salient_items: List of potentially salient items
            forced: Item to forcibly attend to (overrides salience calculation)
            item_properties: Dict mapping items to their properties
            
        Returns:
            Currently focused item
        """
        # Update timing
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # If a focus item is forced, use it
        if forced is not None:
            if forced != self.current_focus:
                # New focus
                if self.current_focus is not None:
                    self.focus_history.append(self.current_focus)
                self.current_focus = forced
                self.focus_duration = 0
            else:
                # Same focus
                self.focus_duration += time_delta
                
            # Update habituation
            self._update_habituation(forced, time_delta)
            return forced
            
        # No items to consider
        if not salient_items:
            return self.current_focus
            
        # Calculate salience for each item
        scored_items = self.calculate_salience(salient_items, item_properties)
        
        # Check if any item exceeds the focus threshold
        if scored_items and scored_items[0][1] >= self.focus_threshold:
            most_salient = scored_items[0][0]
            if most_salient != self.current_focus:
                # New focus
                if self.current_focus is not None:
                    self.focus_history.append(self.current_focus)
                self.current_focus = most_salient
                self.focus_duration = 0
            else:
                # Same focus
                self.focus_duration += time_delta
                
            # Update habituation
            self._update_habituation(most_salient, time_delta)
        elif self.current_focus is not None:
            # Maintain current focus with increased duration
            self.focus_duration += time_delta
            self._update_habituation(self.current_focus, time_delta)
            
        return self.current_focus
    
    def _update_habituation(self, item, time_delta):
        """
        Update habituation for an item
        
        Args:
            item: The item to update
            time_delta: Time elapsed since last update
        """
        current_time = time.time()
        
        # Get current habituation or initialize
        if item in self.habituated_items:
            habituation, _ = self.habituated_items[item]
        else:
            habituation = 0
            
        # Increase habituation for attended item
        habituation = min(1.0, habituation + time_delta * self.habituation_rate)
        
        # Store updated habituation
        self.habituated_items[item] = (habituation, current_time)
        
        # Recover habituation for other items
        for other_item in list(self.habituated_items.keys()):
            if other_item != item:
                h, last_time = self.habituated_items[other_item]
                recovery = time_delta * self.recovery_rate
                h = max(0, h - recovery)
                
                if h <= 0.01:
                    # Remove if habituation is negligible
                    del self.habituated_items[other_item]
                else:
                    self.habituated_items[other_item] = (h, last_time)
    
    def apply_top_down_bias(self, items, goal, bias_strength=0.4):
        """
        Apply top-down goal-directed bias to attention
        
        Args:
            items: List of items to bias
            goal: The current goal or intention
            bias_strength: Strength of the goal-directed bias (0-1)
            
        Returns:
            List of (item, biased_salience) tuples, sorted by biased salience
        """
        if not goal or not items:
            return []
            
        # First calculate standard bottom-up salience
        salience_scores = self.calculate_salience(items)
        
        # Calculate goal relevance for each item
        relevance_scores = {}
        for item in items:
            relevance = self._calculate_goal_relevance(item, goal)
            relevance_scores[item] = relevance
            # Store for future use
            self.goal_relevance[item] = relevance
        
        # Apply bias to salience scores
        biased_scores = []
        for item, salience in salience_scores:
            relevance = relevance_scores.get(item, 0)
            # Bias formula: combination of bottom-up salience and goal relevance
            biased_salience = salience * (1.0 - bias_strength) + relevance * bias_strength
            biased_scores.append((item, biased_salience))
        
        # Sort by biased salience score
        biased_scores.sort(key=lambda x: x[1], reverse=True)
        return biased_scores
    
    def _calculate_goal_relevance(self, item, goal):
        """
        Calculate relevance of an item to the current goal
        
        Args:
            item: The item to evaluate
            goal: The current goal
            
        Returns:
            Relevance score (0-1)
        """
        # In a real system, this would use sophisticated semantic matching
        # Here we use a simplified approach based on string matching
        
        # Convert goal and item to string representations
        if hasattr(goal, '__str__'):
            goal_str = str(goal).lower()
        else:
            goal_str = str(type(goal)).lower()
            
        if hasattr(item, '__str__'):
            item_str = str(item).lower()
        else:
            item_str = str(type(item)).lower()
            
        # Count word overlap
        goal_words = set(goal_str.split())
        item_words = set(item_str.split())
        
        if not goal_words or not item_words:
            return 0.1  # Default low relevance
            
        # Calculate Jaccard similarity
        intersection = len(goal_words.intersection(item_words))
        union = len(goal_words.union(item_words))
        
        if union == 0:
            return 0.1
            
        return intersection / union
    
    def focus_on(self, item):
        """
        Forcibly focus attention on a specific item
        
        Args:
            item: The item to focus on
            
        Returns:
            The focused item
        """
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Store previous focus in history
        if self.current_focus is not None and self.current_focus != item:
            self.focus_history.append(self.current_focus)
            
        # Set new focus
        self.current_focus = item
        self.focus_duration = 0
        
        # Update habituation
        self._update_habituation(item, time_delta)
        
        return item
    
    def get_attention_state(self):
        """
        Get the current state of the attention system
        
        Returns:
            Dict with current attentional state
        """
        return {
            "focus": self.current_focus,
            "duration": self.focus_duration,
            "history": list(self.focus_history),
            "habituation": {k: v[0] for k, v in self.habituated_items.items()}
        }
    
    def reset(self):
        """Reset the attention system"""
        self.current_focus = None
        self.focus_duration = 0
        self.focus_history.clear()
        self.habituated_items = {}
        self.goal_relevance = {}
        self.last_update_time = time.time()