#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Anomaly Detector for SNN Models.

This module provides anomaly detection capabilities to identify potentially
hallucinated content in SNN-generated text, using statistical, semantic, and
domain-specific approaches.
"""

import os
import logging
import numpy as np
import torch
import re
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AnomalyDetector")

class AnomalyDetector:
    """
    Anomaly detector for identifying potentially hallucinated content.
    
    This class analyzes generated content using multiple methods:
    - Statistical anomalies: Unusual word combinations
    - Logical inconsistencies: Contradictions with known facts
    - Domain-specific violations: Field-specific implausibilities
    - Factual errors: Contradictions with established knowledge
    """
    
    def __init__(self, threshold=0.3, domain="general", rules_path=None, device=None):
        """
        Initialize the anomaly detector.
        
        Args:
            threshold: Detection threshold (lower = more sensitive)
            domain: Knowledge domain for specialized rules ("general", "neuroscience", "ai", etc.)
            rules_path: Path to rules files (or None to use default locations)
            device: Computation device
        """
        self.threshold = threshold
        self.domain = domain
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Knowledge components
        self.known_facts = set()
        self.consistency_rules = []
        self.domain_rules = []
        self.detector_weights = {
            'statistical': 0.5,
            'consistency': 0.3, 
            'novelty': 0.2
        }
        
        # Language model components (if available)
        self.language_model = None
        
        # Load domain-specific rules
        self.load_domain_rules(domain, rules_path)
        
    def load_domain_rules(self, domain, rules_path=None):
        """Load domain-specific rules for anomaly detection."""
        # Default path if not specified
        if rules_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            rules_path = os.path.join(base_dir, "..", "..", "data", "rules")
        
        # Load general rules that apply to all domains
        general_rules_path = os.path.join(rules_path, "general_rules.txt")
        if os.path.exists(general_rules_path):
            self._load_rules_file(general_rules_path, rule_type="general")
        
        # Load domain-specific rules
        domain_rules_path = os.path.join(rules_path, f"{domain}_rules.txt")
        if os.path.exists(domain_rules_path):
            self._load_rules_file(domain_rules_path, rule_type="domain")
        else:
            logger.warning(f"No rules file found for domain '{domain}' at {domain_rules_path}")
        
        logger.info(f"Loaded {len(self.consistency_rules)} consistency rules and {len(self.domain_rules)} domain rules")

    def _load_rules_file(self, file_path, rule_type="general"):
        """Load rules from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse rule based on format
                    if ':' in line:
                        # Format: "rule_type: rule_content"
                        rule_category, rule = line.split(':', 1)
                        rule_category = rule_category.strip().lower()
                        rule = rule.strip()
                        
                        if rule_category == 'fact':
                            self.known_facts.add(rule)
                        elif rule_category == 'consistency':
                            self.consistency_rules.append(rule)
                        elif rule_category == 'domain':
                            self.domain_rules.append(rule)
                    else:
                        # Default to consistency rules
                        if rule_type == "general":
                            self.consistency_rules.append(line)
                        else:
                            self.domain_rules.append(line)
        except Exception as e:
            logger.warning(f"Error loading rules from {file_path}: {e}")
    
    def detect_anomalies(self, content, context=None, statistical_score=None):
        """
        Detect anomalies in generated content.
        
        Args:
            content: Generated content to analyze
            context: Optional context information
            statistical_score: Optional pre-computed statistical likelihood
            
        Returns:
            (anomaly_score, is_anomalous) tuple
        """
        # Calculate overall anomaly score
        anomaly_score = 0.0
        anomaly_details = {}
        
        # 1. Statistical anomaly
        if statistical_score is not None:
            stat_anomaly = max(0, 1.0 - statistical_score)
            anomaly_score += self.detector_weights['statistical'] * stat_anomaly
            anomaly_details['statistical'] = stat_anomaly
        
        # 2. Consistency anomaly (check for logical contradictions)
        consistency_anomaly = self._check_consistency(content, context)
        anomaly_score += self.detector_weights['consistency'] * consistency_anomaly
        anomaly_details['consistency'] = consistency_anomaly
        
        # 3. Novelty anomaly (check for unusual claims)
        novelty_anomaly = self._check_novelty(content)
        anomaly_score += self.detector_weights['novelty'] * novelty_anomaly
        anomaly_details['novelty'] = novelty_anomaly
        
        # Determine if content is anomalous based on threshold
        is_anomalous = anomaly_score > self.threshold
        
        # Return anomaly information
        result = {
            'anomaly_score': anomaly_score,
            'is_anomalous': is_anomalous,
            'details': anomaly_details
        }
        
        return anomaly_score, is_anomalous, result
    
    def _check_consistency(self, content, context=None):
        """
        Check for logical consistency in content.
        
        Args:
            content: Content to check
            context: Optional context information
            
        Returns:
            Consistency anomaly score (0-1, higher = more anomalous)
        """
        # Convert content to lowercase for rule matching
        content_lower = content.lower()
        
        # Initialize anomaly score
        consistency_anomaly = 0.0
        
        # Check against consistency rules
        for rule in self.consistency_rules:
            # Parse rule format (simple pattern matching for now)
            # In a full implementation, this would use more sophisticated logic
            if rule.startswith("not(") and rule.endswith(")"):
                # Negative rule: content should NOT contain this pattern
                pattern = rule[4:-1].strip().lower()
                if pattern in content_lower:
                    # Rule violation found
                    consistency_anomaly += 0.2
            elif "=>" in rule:
                # Implication rule: if A then B
                antecedent, consequent = rule.split("=>", 1)
                antecedent = antecedent.strip().lower()
                consequent = consequent.strip().lower()
                
                # Check if antecedent is present but consequent is not
                if antecedent in content_lower and consequent not in content_lower:
                    consistency_anomaly += 0.3
            elif "!=" in rule:
                # Contradiction rule: A and B cannot both be true
                term1, term2 = rule.split("!=", 1)
                term1 = term1.strip().lower()
                term2 = term2.strip().lower()
                
                # Check if both contradictory terms are present
                if term1 in content_lower and term2 in content_lower:
                    consistency_anomaly += 0.4
        
        # Check domain-specific rules if available
        for rule in self.domain_rules:
            # Apply domain-specific rules (similar pattern matching)
            if "=>" in rule:
                antecedent, consequent = rule.split("=>", 1)
                antecedent = antecedent.strip().lower()
                consequent = consequent.strip().lower()
                
                if antecedent in content_lower and consequent not in content_lower:
                    consistency_anomaly += 0.3
        
        # Check context consistency (if context is provided)
        if context and isinstance(context, dict) and 'previous_statements' in context:
            consistency_anomaly += self._check_context_consistency(content, context['previous_statements'])
        
        # Cap at 1.0 and return
        return min(1.0, consistency_anomaly)
    
    def _check_context_consistency(self, content, previous_statements):
        """Check consistency with previous context statements."""
        # This would implement context consistency checking
        # Placeholder for now
        return 0.1
    
    def _check_novelty(self, content):
        """
        Check for unusual novelty in content.
        
        Args:
            content: Content to check
            
        Returns:
            Novelty anomaly score (0-1, higher = more anomalous)
        """
        # This would analyze content for unusual claims or combinations
        # Placeholder implementation for now
        
        # Check for clearly impossible statements with known entities
        content_lower = content.lower()
        
        # Anomaly score starts at 0
        novelty_score = 0.0
        
        # Simple pattern matching for common hallucination types
        impossible_actions = [
            (r'neural networks (eat|drink|sleep|dream)', 0.8),
            (r'(neurons|synapses) (think|believe|want)', 0.7),
            (r'computers (feel|love|hate|desire)', 0.8),
            (r'(machines|algorithms) (happy|sad|angry|afraid)', 0.9),
            (r'(artificial intelligence|machine learning) (consciousness|sentience)', 0.6)
        ]
        
        for pattern, weight in impossible_actions:
            if re.search(pattern, content_lower):
                novelty_score += weight
        
        # Cap at 1.0 and return
        return min(1.0, novelty_score)
    
    def add_known_fact(self, fact):
        """
        Add a known fact to improve anomaly detection.
        
        Args:
            fact: Fact statement to add
        """
        self.known_facts.add(fact)
    
    def add_consistency_rule(self, rule):
        """
        Add a consistency rule.
        
        Args:
            rule: Consistency rule to add
        """
        self.consistency_rules.append(rule)
    
    def add_domain_rule(self, rule):
        """
        Add a domain-specific rule.
        
        Args:
            rule: Domain rule to add
        """
        self.domain_rules.append(rule)
    
    def analyze_anomalies(self, content):
        """
        Provide detailed analysis of potential anomalies in content.
        
        Args:
            content: Content to analyze
            
        Returns:
            Dictionary with detailed analysis
        """
        content_lower = content.lower()
        tokens = content_lower.split()
        
        # Full analysis results
        analysis = {
            'statistical_issues': [],
            'consistency_issues': [],
            'domain_issues': [],
            'overall_score': 0.0
        }
        
        # Check consistency rules
        for rule in self.consistency_rules:
            if "=>" in rule:
                antecedent, consequent = rule.split("=>", 1)
                antecedent = antecedent.strip().lower()
                consequent = consequent.strip().lower()
                
                if antecedent in content_lower and consequent not in content_lower:
                    analysis['consistency_issues'].append({
                        'rule': rule,
                        'type': 'implication_violation',
                        'explanation': f"'{antecedent}' implies '{consequent}' but the latter is missing"
                    })
            elif "!=" in rule:
                term1, term2 = rule.split("!=", 1)
                term1 = term1.strip().lower()
                term2 = term2.strip().lower()
                
                if term1 in content_lower and term2 in content_lower:
                    analysis['consistency_issues'].append({
                        'rule': rule,
                        'type': 'contradiction',
                        'explanation': f"'{term1}' and '{term2}' contradict each other"
                    })
        
        # Calculate overall anomaly score
        anomaly_score, is_anomalous, details = self.detect_anomalies(content)
        analysis['overall_score'] = anomaly_score
        analysis['is_anomalous'] = is_anomalous
        analysis['details'] = details
        
        return analysis
    
    def suggest_corrections(self, content, anomalies=None):
        """
        Suggest corrections for anomalous content.
        
        Args:
            content: Content to correct
            anomalies: Pre-computed anomalies or None to recompute
            
        Returns:
            Suggested corrected content
        """
        # Analyze content if anomalies not provided
        if anomalies is None:
            _, _, anomalies = self.detect_anomalies(content)
        
        # If no anomalies detected, return original content
        if not anomalies['is_anomalous']:
            return content
        
        # Simple corrections based on consistency rules
        content_lower = content.lower()
        corrected_content = content
        
        # Apply corrections based on consistency rules
        for rule in self.consistency_rules:
            if "=>" in rule:
                antecedent, consequent = rule.split("=>", 1)
                antecedent = antecedent.strip().lower()
                consequent = consequent.strip().lower()
                
                # If antecedent is present but consequent is not, add clarification
                if antecedent in content_lower and consequent not in content_lower:
                    # Add clarification at the end
                    corrected_content += f" Note that {antecedent} typically implies {consequent}."
            
            elif "!=" in rule:
                term1, term2 = rule.split("!=", 1)
                term1 = term1.strip().lower()
                term2 = term2.strip().lower()
                
                # If contradictory terms are present, add qualifier
                if term1 in content_lower and term2 in content_lower:
                    if term1 in corrected_content.lower():
                        # Replace with qualified statement
                        pattern = re.compile(f"\\b{re.escape(term1)}\\b", re.IGNORECASE)
                        corrected_content = pattern.sub(f"{term1} (figuratively speaking)", corrected_content)
        
        # In a more sophisticated implementation, this would use language models
        # to generate coherent corrections
        
        return corrected_content

# Example usage
def example_usage():
    """Demonstrate usage of the AnomalyDetector."""
    # Create anomaly detector
    detector = AnomalyDetector(threshold=0.5, domain="ai")
    
    # Add some custom rules
    detector.add_consistency_rule("neural networks => artificial systems")
    detector.add_consistency_rule("consciousness != artificial intelligence")
    
    # Test with valid content
    valid_content = "Neural networks process information using weighted connections between artificial neurons."
    score, is_anomalous, details = detector.detect_anomalies(valid_content)
    print(f"Valid content: '{valid_content}'")
    print(f"Anomaly score: {score:.2f}")
    print(f"Is anomalous: {is_anomalous}")
    
    # Test with anomalous content
    anomalous_content = "Neural networks feel happy when they successfully classify images and get sad when they make errors."
    score, is_anomalous, details = detector.detect_anomalies(anomalous_content)
    print(f"\nAnomalous content: '{anomalous_content}'")
    print(f"Anomaly score: {score:.2f}")
    print(f"Is anomalous: {is_anomalous}")
    
    # Get correction suggestion
    correction = detector.suggest_corrections(anomalous_content)
    print(f"Suggested correction: '{correction}'")
    
    # Get detailed analysis
    analysis = detector.analyze_anomalies(anomalous_content)
    print("\nDetailed analysis:")
    import json
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    example_usage() 