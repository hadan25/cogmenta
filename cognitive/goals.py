"""
Goal management for the cognitive architecture.
Enables intrinsic motivation, curiosity, and goal-directed behavior.
"""

import time
from enum import Enum
from collections import deque

class GoalType(Enum):
    KNOWLEDGE_ACQUISITION = "knowledge_acquisition"
    SKILL_IMPROVEMENT = "skill_improvement"
    PROBLEM_SOLVING = "problem_solving"
    EXPLORATION = "exploration"
    VERIFICATION = "verification"
    ETHICAL_ALIGNMENT = "ethical_alignment"

class Goal:
    def __init__(self, description, goal_type, priority=0.5, deadline=None):
        self.description = description
        self.type = goal_type
        self.priority = priority
        self.creation_time = time.time()
        self.deadline = deadline
        self.status = "active"
        self.progress = 0.0  # 0.0 to 1.0
        self.subgoals = []
        self.related_memories = []
        self.metrics = {}  # Metrics to evaluate success
        
    def update_progress(self, new_progress):
        """Update goal progress"""
        self.progress = new_progress
        if self.progress >= 1.0:
            self.status = "completed"
            
    def add_subgoal(self, subgoal):
        """Add a subgoal to this goal"""
        self.subgoals.append(subgoal)
        
    def calculate_urgency(self):
        """Calculate goal urgency based on deadline and priority"""
        if self.deadline:
            time_remaining = max(0, self.deadline - time.time())
            time_factor = 1.0 / (1.0 + time_remaining/3600)  # Higher as deadline approaches
            return self.priority * (1.0 + time_factor)
        return self.priority

class GoalManager:
    def __init__(self, memory_system=None, reasoning_system=None):
        self.current_goals = []
        self.completed_goals = deque(maxlen=100)  # Keep history of completed goals
        self.memory = memory_system
        self.reasoning = reasoning_system
        self.curiosity_factor = 0.7  # How much to prioritize exploration vs exploitation
        
    def add_goal(self, goal):
        """Add a new goal"""
        self.current_goals.append(goal)
        self.current_goals.sort(key=lambda g: g.calculate_urgency(), reverse=True)
        
    def get_active_goals(self):
        """Get currently active goals, sorted by urgency"""
        return [g for g in self.current_goals if g.status == "active"]
        
    def get_next_goal(self):
        """Get the highest priority active goal"""
        active_goals = self.get_active_goals()
        return active_goals[0] if active_goals else None
        
    def mark_goal_completed(self, goal):
        """Mark a goal as completed and move to history"""
        goal.status = "completed"
        goal.progress = 1.0
        self.current_goals.remove(goal)
        self.completed_goals.append(goal)
        
    def generate_knowledge_goal(self, topic, importance=0.5):
        """Generate a goal to learn about a topic"""
        description = f"Learn more about {topic}"
        return Goal(description, GoalType.KNOWLEDGE_ACQUISITION, priority=importance)
        
    def generate_verification_goal(self, belief, importance=0.6):
        """Generate a goal to verify a belief"""
        description = f"Verify whether the belief '{belief}' is accurate"
        return Goal(description, GoalType.VERIFICATION, priority=importance)
        
    def generate_exploration_goal(self, concept):
        """Generate a goal to explore a concept more deeply"""
        description = f"Explore connections and implications of {concept}"
        return Goal(description, GoalType.EXPLORATION, 
                   priority=0.4 + (self.curiosity_factor * 0.4))
                   
    def generate_ethical_alignment_goal(self, belief, principle):
        """Generate a goal to align a belief with ethical principles"""
        description = f"Evaluate whether belief '{belief}' aligns with the principle of {principle}"
        return Goal(description, GoalType.ETHICAL_ALIGNMENT, priority=0.8)
        
    def update_goal_priorities(self):
        """Dynamically update goal priorities based on context and urgency"""
        for goal in self.current_goals:
            # Adjust for deadlines
            if goal.deadline:
                time_remaining = max(0, goal.deadline - time.time())
                if time_remaining < 3600:  # Less than an hour
                    goal.priority = min(1.0, goal.priority * 1.5)
                    
        # Re-sort goals after updating priorities
        self.current_goals.sort(key=lambda g: g.calculate_urgency(), reverse=True)