#!/usr/bin/env python3
"""
Demonstration of the Conceptual Understanding Layer for Cogmenta Core.
Shows how the layer integrates concept embeddings, structured meaning maps,
meaning extraction, and cross-formalism translation.
"""
import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import time
import json
from pprint import pprint

# Import the necessary components
from concept_embeddings import ConceptEmbeddingSystem
from structured_meaning_map import StructuredMeaningMap
from meaning_extraction import MeaningExtractionSystem
from cross_formalism_translation import CrossFormalismTranslation
from conceptual_understanding_layer import ConceptualUnderstandingLayer

# Import other required components for integration
from models.symbolic.vector_symbolic import VectorSymbolicEngine
from models.symbolic.prolog_engine import PrologEngine
from processing.nlp_pipeline import NLPPipeline

def print_section(title):
    """Print a section heading."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def main():
    """
    Run the Conceptual Understanding Layer demonstration.
    """
    print_section("Initializing Components")
    
    # Initialize core components
    print("Initializing concept embedding system...")
    concept_system = ConceptEmbeddingSystem(embedding_dim=300)
    
    print("Initializing structured meaning map...")
    meaning_map = StructuredMeaningMap(concept_system)
    
    print("Initializing NLP pipeline...")
    nlp_pipeline = NLPPipeline()
    
    print("Initializing meaning extraction system...")
    meaning_extraction = MeaningExtractionSystem(
        concept_system=concept_system,
        meaning_map=meaning_map,
        nlp_pipeline=nlp_pipeline
    )
    
    print("Initializing vector symbolic engine...")
    vector_symbolic = VectorSymbolicEngine(dimension=300, sparsity=0.1)
    
    print("Initializing Prolog engine...")
    prolog_engine = PrologEngine()
    
    print("Initializing cross-formalism translation...")
    translation_layer = CrossFormalismTranslation(
        concept_system=concept_system,
        vector_symbolic=vector_symbolic,
        prolog_engine=prolog_engine,
        meaning_map=meaning_map
    )
    
    # Initialize the conceptual understanding layer
    print("Initializing conceptual understanding layer...")
    conceptual_layer = ConceptualUnderstandingLayer(
        concept_system=concept_system,
        meaning_map=meaning_map,
        meaning_extraction=meaning_extraction,
        translation_layer=translation_layer,
        vector_symbolic=vector_symbolic,
        prolog_engine=prolog_engine
    )
    
    # Initialize common concepts
    print("Initializing common concepts...")
    conceptual_layer.initialize_common_concepts()
    
    print_section("Processing Example Texts")
    
    # Example texts to process
    example_texts = [
        "Alice trusts Bob.",
        "Bob likes Alice but doesn't trust Charlie.",
        "Charlie fears Dave because Dave is aggressive.",
        "Everyone who trusts Alice also trusts Bob.",
        "Organizations that have authority must be trustworthy."
    ]
    
    # Process each example
    for i, text in enumerate(example_texts):
        print(f"\nExample {i+1}: '{text}'")
        
        # Process with conceptual understanding
        result = conceptual_layer.process_input(text)
        
        # Print key results
        print(f"- Processing time: {result['process_time']:.4f} seconds")
        print(f"- Concepts identified: {len(result['concepts'])}")
        if result['concepts']:
            print("  " + ", ".join([c['name'] for c in result['concepts']]))
        
        print(f"- Meaning summary: {result.get('meaning_summary', 'Not available')}")
        
        if result.get('meaning_graph_id'):
            print(f"- Created meaning graph: {result['meaning_graph_id']}")
        
        # Show propositions
        if result.get('meaning') and result['meaning'].get('propositions'):
            print("- Extracted propositions:")
            for i, prop in enumerate(result['meaning']['propositions'][:3]):  # Show first 3
                negated = " not" if prop.get('negated', False) else ""
                print(f"  {i+1}. {prop['subject']}{negated} {prop['predicate']} {prop['object']}")
            
            if len(result['meaning']['propositions']) > 3:
                print(f"  ... and {len(result['meaning']['propositions']) - 3} more")
        
        # Show translations if available
        if result.get('translations'):
            print("- Translations performed:")
            for trans_type, trans_result in result['translations'].items():
                print(f"  * {trans_type}: {len(trans_result)} results")
    
    print_section("Demonstrating Concept Queries")
    
    # Query for concepts
    query_terms = ["trust", "person", "organization", "fear"]
    
    for query in query_terms:
        print(f"\nQuerying for concept: '{query}'")
        result = conceptual_layer.query_concepts(query)
        
        # Show direct concept matches
        if result['concepts']:
            print("- Direct concept matches:")
            for concept in result['concepts']:
                print(f"  * {concept['name']} (similarity: {concept['similarity']:.2f})")
                if concept.get('metadata') and concept['metadata'].get('definition'):
                    print(f"    Definition: {concept['metadata']['definition']}")
        
        # Show related concepts
        if result['related_concepts']:
            print("- Related concepts:")
            for concept in result['related_concepts'][:5]:  # Show first 5
                print(f"  * {concept['name']} ({concept['relation']}, weight: {concept['weight']:.2f})")
            
            if len(result['related_concepts']) > 5:
                print(f"  ... and {len(result['related_concepts']) - 5} more")
        
        # Show activated concepts
        if result['activated_concepts']:
            print("- Currently activated concepts:")
            for concept in result['activated_concepts'][:3]:  # Show first 3
                print(f"  * {concept['name']} (activation: {concept['activation']:.2f})")
    
    print_section("Comparing Texts")
    
    # Compare texts for similarity
    text_pairs = [
        ("Alice trusts Bob.", "Bob is trusted by Alice."),
        ("Alice trusts Bob.", "Alice likes Bob."),
        ("Organizations must be trustworthy.", "Organizations should be reliable.")
    ]
    
    for text1, text2 in text_pairs:
        print(f"\nComparing texts:")
        print(f"1. '{text1}'")
        print(f"2. '{text2}'")
        
        comparison = conceptual_layer.compare_texts(text1, text2)
        
        print(f"- Overall similarity: {comparison['overall_similarity']:.2f}")
        print(f"- Proposition similarity: {comparison['proposition_similarity']:.2f}")
        print(f"- Concept similarity: {comparison['concept_similarity']:.2f}")
        
        if comparison.get('graph_similarity'):
            print(f"- Graph similarity: {comparison['graph_similarity']:.2f}")
        
        if comparison['shared_concepts']:
            print("- Shared concepts:", ", ".join(comparison['shared_concepts']))
    
    print_section("Cross-Formalism Translation")
    
    # Create a symbolic fact
    symbolic_fact = {
        'type': 'symbolic_fact',
        'subject': 'alice',
        'predicate': 'trusts',
        'object': 'bob',
        'confidence': 0.9
    }
    
    print("Original symbolic fact:")
    pprint(symbolic_fact)
    
    # Translate to vector representation
    print("\nTranslating to vector representation...")
    vector_fact = conceptual_layer.translate_between_formalisms(
        symbolic_fact, 'symbolic', 'vector'
    )
    
    print("Vector fact created (showing metadata, vector truncated):")
    vector_display = {k: v for k, v in vector_fact.items() if k != 'vector'}
    vector_display['vector'] = "[...]"  # Truncate for display
    pprint(vector_display)
    
    # Translate back to symbolic
    print("\nTranslating back to symbolic...")
    symbolic_fact2 = conceptual_layer.translate_between_formalisms(
        vector_fact, 'vector', 'symbolic'
    )
    
    print("Result of round-trip translation:")
    pprint(symbolic_fact2)
    
    # Translate to meaning graph
    print("\nTranslating to meaning graph...")
    graph_id = conceptual_layer.translate_between_formalisms(
        symbolic_fact, 'symbolic', 'graph'
    )
    
    if graph_id:
        print(f"Created meaning graph with ID: {graph_id}")
        
        # Get graph information
        graph = meaning_map.meaning_graphs.get(graph_id, {})
        print(f"Graph has {len(graph.get('nodes', []))} nodes and {len(graph.get('edges', []))} edges")
    
    print_section("Comprehensive Text Processing")
    
    # Process a more complex text
    complex_text = """
    In social networks, trust is a critical factor. Alice trusts Bob completely,
    and Bob trusts Charlie. However, Dave doesn't trust Charlie because of past experiences.
    Organizations must establish trust with their customers to be successful.
    """
    
    print(f"Processing text: '{complex_text.strip()}'")
    
    # Get comprehensive report
    report = conceptual_layer.process_text_with_conceptual_understanding(complex_text)
    
    # Display key information from report
    print("\nConceptual Understanding:")
    print(f"- Identified {len(report['conceptual_understanding']['concepts'])} concepts")
    print(f"- Meaning summary: {report['conceptual_understanding']['meaning_summary']}")
    
    print("\nStructured Representations:")
    for rep_type, rep_data in report['structured_representations'].items():
        print(f"- {rep_type}: {rep_data}")
    
    print("\nTranslations:")
    for trans_type, trans_data in report['translations'].items():
        print(f"- {trans_type}: {trans_data['count']} results")
    
    # Show performance metrics
    print_section("Performance Metrics")
    
    metrics = conceptual_layer.get_metrics()
    
    print(f"Total processed inputs: {metrics['total_processed']}")
    print(f"Average processing time: {metrics['avg_process_time']:.4f} seconds")
    
    if 'top_activated_concepts' in metrics:
        print("\nTop activated concepts:")
        for concept, activation in metrics['top_activated_concepts']:
            print(f"- {concept}: {activation:.2f}")
    
    if 'translation_stats' in metrics:
        print("\nTranslation statistics:")
        for trans_type, count in metrics['translation_stats']['by_type'].items():
            success_rate = metrics['translation_stats']['success_rate'].get(trans_type, {}).get('success_percentage', 0)
            print(f"- {trans_type}: {count} operations, {success_rate:.1f}% success rate")

if __name__ == "__main__":
    main()