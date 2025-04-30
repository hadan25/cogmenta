"""Configuration for metacognitive monitoring and control"""

METACOGNITION_CONFIG = {
    'monitoring': {
        'attention': {
            'load_threshold': 0.8,
            'focus_duration': 0.5,
            'switch_cost': 0.2
        },
        'learning': {
            'adaptation_rate': 0.1,
            'confidence_threshold': 0.7,
            'min_examples': 5
        },
        'reflection': {
            'interval': 100,  # Steps between reflections
            'min_confidence': 0.6,
            'improvement_threshold': 0.1
        }
    },
    'control': {
        'strategy_selection': {
            'exploration_rate': 0.2,
            'success_threshold': 0.7
        },
        'resource_allocation': {
            'attention_weight': 0.4,
            'memory_weight': 0.3,
            'processing_weight': 0.3
        }
    }
}
