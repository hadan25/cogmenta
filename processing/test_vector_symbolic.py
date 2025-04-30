# cogmenta_core/tests/test_vector_symbolic.py

from models.symbolic.vector_symbolic import VectorSymbolicEngine

def test_vector_symbolic_engine():
    """Test the vector symbolic engine on language facts."""
    vsa = VectorSymbolicEngine(dimension=300, sparsity=0.1)
    
    # Create some facts
    vsa.create_fact("alice", "likes", "bob", confidence=0.9)
    vsa.create_fact("bob", "trusts", "alice", confidence=0.8)
    vsa.create_fact("charlie", "fears", "dogs", confidence=0.7)
    
    # Query facts
    alice_facts = vsa.query_facts(subject="alice")
    bob_facts = vsa.query_facts(subject="bob")
    
    print("\n=== Alice Facts ===")
    for fact in alice_facts:
        print(f"{fact['subject']} {fact['predicate']} {fact['object']} (confidence: {fact['confidence']:.2f})")
    
    print("\n=== Bob Facts ===")
    for fact in bob_facts:
        print(f"{fact['subject']} {fact['predicate']} {fact['object']} (confidence: {fact['confidence']:.2f})")
    
    # Process a text query
    result = vsa.process_text("Does Alice like Bob?")
    print("\n=== Query Result ===")
    print(result['response'])
    
    # Test similarity matching
    vsa.create_fact("robert", "likes", "pizza", confidence=0.9)
    vsa.create_fact("robb", "plays", "guitar", confidence=0.8)
    
    print("\n=== Similarity Between Bob and Robert ===")
    sim = vsa.similarity("bob", "robert")
    print(f"Similarity: {sim:.2f}")
    
    # Test query with similar words
    similar_results = vsa.query_facts(subject="bobby", threshold=0.6)
    
    print("\n=== Results for similar query 'bobby' ===")
    for fact in similar_results:
        print(f"{fact['subject']} {fact['predicate']} {fact['object']} (confidence: {fact['confidence']:.2f})")

if __name__ == "__main__":
    test_vector_symbolic_engine()