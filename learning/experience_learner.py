from collections import defaultdict
import numpy as np
import time

class ExperienceLearner:
    def __init__(self):
        self.experiences = defaultdict(list)
        self.learned_lessons = []
        self.adaptation_rate = 0.1
        self.success_threshold = 0.7
        
    def record_experience(self, context, action, outcome):
        """Record an experience for learning"""
        experience = {
            'context': context,
            'action': action,
            'outcome': outcome,
            'timestamp': time.time(),
            'success_score': outcome.get('success_score', 0.0)
        }
        self.experiences[action['type']].append(experience)
        
    def extract_lessons(self, min_examples=5):
        """Extract lessons from accumulated experiences"""
        new_lessons = []
        
        for action_type, experiences in self.experiences.items():
            if len(experiences) < min_examples:
                continue
                
            # Analyze success patterns
            success_patterns = self._find_success_patterns(experiences)
            
            # Generate lessons from patterns
            for pattern in success_patterns:
                if pattern['confidence'] >= self.success_threshold:
                    lesson = {
                        'action_type': action_type,
                        'pattern': pattern,
                        'recommendation': self._generate_recommendation(pattern),
                        'timestamp': time.time()
                    }
                    new_lessons.append(lesson)
                    
        self.learned_lessons.extend(new_lessons)
        return new_lessons
        
    def apply_lessons(self, current_context):
        """Apply learned lessons to current context"""
        applicable_lessons = []
        
        for lesson in self.learned_lessons:
            similarity = self._calculate_context_similarity(
                current_context,
                lesson['pattern']['context_features']
            )
            if similarity > 0.7:
                applicable_lessons.append({
                    'lesson': lesson,
                    'relevance': similarity
                })
                
        return sorted(applicable_lessons, key=lambda x: x['relevance'], reverse=True)
        
    def _find_success_patterns(self, experiences):
        """Find patterns in successful experiences"""
        patterns = []
        
        # Group by success
        successful = [e for e in experiences if e['success_score'] > self.success_threshold]
        if not successful:
            return patterns
            
        # Extract common context features
        common_features = self._extract_common_features([e['context'] for e in successful])
        
        if common_features:
            pattern = {
                'context_features': common_features,
                'success_rate': len(successful) / len(experiences),
                'confidence': self._calculate_pattern_confidence(common_features, experiences),
                'sample_size': len(experiences)
            }
            patterns.append(pattern)
            
        return patterns
        
    def _extract_common_features(self, contexts):
        """Extract features common to multiple contexts"""
        if not contexts:
            return {}
            
        common = {}
        reference = contexts[0]
        
        for key, value in reference.items():
            if all(abs(c.get(key, 0) - value) < 0.2 for c in contexts):
                common[key] = value
                
        return common
        
    def _calculate_pattern_confidence(self, pattern, experiences):
        """Calculate confidence in a pattern"""
        matching = [e for e in experiences 
                   if all(abs(e['context'].get(k, 0) - v) < 0.2 
                         for k, v in pattern.items())]
                         
        if not matching:
            return 0.0
            
        success_rate = np.mean([e['success_score'] for e in matching])
        sample_factor = min(1.0, len(matching) / 10)  # Scale up with more examples
        
        return success_rate * sample_factor
        
    def _generate_recommendation(self, pattern):
        """Generate actionable recommendation from pattern"""
        return {
            'context_requirements': pattern['context_features'],
            'confidence': pattern['confidence'],
            'action_modifications': self._derive_action_modifications(pattern)
        }
        
    def _calculate_context_similarity(self, context1, context2):
        """Calculate similarity between two contexts"""
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
            
        similarities = []
        for key in common_keys:
            diff = abs(context1[key] - context2[key])
            similarities.append(1.0 / (1.0 + diff))
            
        return np.mean(similarities)
        
    def _derive_action_modifications(self, pattern):
        """Derive action modifications based on pattern"""
        mods = {}
        
        # Adjust parameters based on context features
        for feature, value in pattern['context_features'].items():
            if feature == 'attention_load' and value > 0.7:
                mods['reduce_complexity'] = True
            elif feature == 'processing_depth' and value < 0.3:
                mods['increase_depth'] = True
                
        return mods