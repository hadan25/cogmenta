# cogmenta_core/models/hybrid/formalism_router.py
import re
import logging
from enum import Enum

class FormalismType(Enum):
    PROLOG = "prolog"
    VECTOR_SYMBOLIC = "vector_symbolic"
    PROBABILISTIC = "probabilistic"
    DESCRIPTION_LOGIC = "description_logic"
    ASP = "answer_set_programming"

class FormalismRouter:
    """
    Routes inputs to the appropriate reasoning formalism
    based on the input characteristics.
    """
    
    def __init__(self, prolog_engine=None, vector_engine=None, 
                probabilistic_engine=None, dl_engine=None, asp_engine=None, meta_reasoner=None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize available reasoning engines
        self.engines = {
            FormalismType.PROLOG: prolog_engine,
            FormalismType.VECTOR_SYMBOLIC: vector_engine,
            FormalismType.PROBABILISTIC: probabilistic_engine,
            FormalismType.DESCRIPTION_LOGIC: dl_engine,
            FormalismType.ASP: asp_engine
        }
        
        # Define domain characteristics
        self.domain_patterns = {
            FormalismType.PROLOG: [
                # Mathematical Sciences
                r'equation', r'formula', r'theorem', r'proof',
                r'calculate', r'compute', r'solve', r'derive',
                r'greater than', r'less than', r'equal to',
                r'divisible', r'prime', r'factor', r'sum', r'product',
                # Physics Laws & Formulas
                r'force', r'mass', r'acceleration', r'velocity',
                r'energy', r'momentum', r'pressure', r'density',
                r'wavelength', r'frequency', r'amplitude',
                # Chemistry Calculations
                r'molarity', r'concentration', r'pH', r'equilibrium',
                r'reaction rate', r'atomic mass', r'molecular weight',
                # Formal Logic
                r'if.*then', r'prove', r'disprove', r'follows',
                r'implies', r'therefore', r'because', r'must be',
                r'cannot be', r'impossible', r'necessary',
                # ...rest of existing Prolog patterns...
                # Logical reasoning
                r'if.*then', r'prove', r'disprove', r'follows',
                r'implies', r'therefore', r'because', r'must be',
                r'cannot be', r'impossible', r'necessary',
                r'true', r'false', r'valid', r'invalid',
                r'reason', r'logic', r'proof', r'axiom',
                # Temporal logic
                r'before', r'after', r'during', r'until',
                r'always', r'never', r'eventually'
            ],
            
            FormalismType.VECTOR_SYMBOLIC: [
                # Biological Sciences
                r'species', r'organism', r'cell', r'gene', r'protein',
                r'ecosystem', r'evolution', r'adaptation', r'mutation',
                r'heredity', r'metabolism', r'reproduction',
                # Earth Sciences
                r'geology', r'climate', r'weather', r'atmosphere',
                r'tectonic', r'mineral', r'rock formation', r'erosion',
                r'ocean', r'river', r'mountain', r'volcano',
                # Psychology & Cognitive Science
                r'behavior', r'cognition', r'perception', r'memory',
                r'emotion', r'learning', r'development', r'personality',
                # Social Sciences
                r'society', r'culture', r'economics', r'politics',
                r'history', r'anthropology', r'sociology', r'linguistics',
                # Medical & Health Sciences
                r'disease', r'treatment', r'symptom', r'diagnosis',
                r'therapy', r'medicine', r'health', r'anatomy',
                # ...rest of existing Vector patterns...
                # Emotional/psychological
                r'like', r'love', r'hate', r'feel', r'think',
                r'believe', r'trust', r'fear', r'hope', r'want',
                r'desire', r'prefer', r'enjoy', r'dislike',
                # Relationships
                r'friend', r'enemy', r'ally', r'rival',
                r'parent', r'child', r'sibling', r'family',
                # Semantic similarity
                r'similar to', r'different from', r'reminds of',
                r'means', r'represents', r'suggests', r'implies',
                # Abstract concepts
                r'good', r'bad', r'beautiful', r'ugly',
                r'important', r'meaningful', r'significant'
            ],
            
            # Uncertain reasoning patterns (use Probabilistic)
            FormalismType.PROBABILISTIC: [
                r'probably', r'likely', r'unlikely', r'chance',
                r'might', r'may', r'could', r'possibly',
                r'uncertain', r'estimate', r'approximately'
            ],
            
            # Ontological patterns (use Description Logic)
            FormalismType.DESCRIPTION_LOGIC: [
                r'is a', r'kind of', r'type of', r'category',
                r'taxonomy', r'classify', r'property of',
                r'feature of', r'attribute', r'characteristic'
            ],
            
            # Common sense reasoning (use ASP)
            FormalismType.ASP: [
                r'typically', r'normally', r'usually',
                r'exception', r'unless', r'but not',
                r'generally', r'as a rule', r'by default'
            ]
        }
    
    def determine_formalism(self, text):
        """Enhanced formalism router that better utilizes vector capabilities"""
        text_lower = text.lower()
        
        # Score each formalism based on text characteristics
        formalism_scores = {formalism: 0 for formalism in FormalismType}
        
        # Vector formalism indicators: similarity, fuzzy matching, analogies
        vector_indicators = ['similar', 'like', 'resemble', 'related', 'comparable', 'reminds']
        if any(indicator in text_lower for indicator in vector_indicators):
            formalism_scores[FormalismType.VECTOR_SYMBOLIC] += 3
        
        # Check for questions about relations or similarity
        if ('how' in text_lower and 'related' in text_lower) or 'similar' in text_lower:
            formalism_scores[FormalismType.VECTOR_SYMBOLIC] += 2
            
        # Rest of the scoring logic...
        
        # Get formalism with highest score
        best_formalism = max(formalism_scores.items(), key=lambda x: x[1])
        
        if best_formalism[1] == 0:
            # Default case - choose based on input complexity
            word_count = len(text_lower.split())
            if word_count > 15:  # Longer inputs often have semantic complexity
                return FormalismType.VECTOR_SYMBOLIC
            else:
                return FormalismType.PROLOG
                
        return best_formalism[0]
    
    def get_engine(self, formalism_type):
        """
        Get the reasoning engine for the given formalism type.
        
        Args:
            formalism_type (FormalismType): Type of formalism
            
        Returns:
            Engine instance or None if not available
        """
        engine = self.engines.get(formalism_type)
        
        if engine is None:
            self.logger.warning(f"No engine available for {formalism_type.value}")
            
            # Fall back to available engine
            for fallback_type in [
                FormalismType.VECTOR_SYMBOLIC,
                FormalismType.PROLOG
            ]:
                fallback = self.engines.get(fallback_type)
                if fallback:
                    self.logger.info(f"Falling back to {fallback_type.value}")
                    return fallback
                    
        return engine
    
    def process(self, text):
        """
        Process text with the most appropriate formalism.
        
        Args:
            text (str): Input text to process
            
        Returns:
            dict: Processing results
        """
        # Determine the best formalism
        formalism = self.determine_formalism(text)
        
        # Get the appropriate engine
        engine = self.get_engine(formalism)
        
        if engine is None:
            return {
                'success': False,
                'formalism': formalism.value,
                'error': "No appropriate reasoning engine available"
            }
            
        # Process with the selected engine
        try:
            results = engine.process_text(text)
            
            return {
                'success': True,
                'formalism': formalism.value,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Error processing with {formalism.value}: {str(e)}")
            
            return {
                'success': False,
                'formalism': formalism.value,
                'error': str(e)
            }