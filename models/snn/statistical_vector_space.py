#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Statistical Vector Space for SNN Models.

This module provides a statistical vector space implementation that learns
co-occurrence statistics and provides likelihood estimations for sequences.
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import defaultdict, Counter
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StatisticalVectorSpace")

class StatisticalVectorSpace:
    """
    Statistical vector space component that tracks frequencies and distributions.
    
    Features:
    - Learning co-occurrence statistics
    - Estimating likelihood of sequences
    - Providing statistical grounding for symbolic operations
    - N-gram analysis for contextual patterns
    """
    
    def __init__(self, dim=300, device=None, window_size=5, min_count=2):
        """
        Initialize the statistical vector space.
        
        Args:
            dim: Dimensionality of statistical vectors
            device: Computation device
            window_size: Context window size for co-occurrence learning
            min_count: Minimum count to include token in vocabulary
        """
        self.dim = dim
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        self.min_count = min_count
        
        # Core statistical components
        self.token_counts = Counter()
        self.cooccurrence_matrix = defaultdict(float)
        self.conditional_probs = {}
        self.joint_probs = {}
        
        # N-gram components
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        
        # Domain language models
        self.domain_specific_patterns = {}
        
        # Vector space
        self.token_vectors = {}
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.embedding_matrix = None
        
        # Metadata
        self.total_tokens = 0
        self.vocabulary_size = 0
        self.is_trained = False
    
    def preprocess_text(self, text):
        """Preprocess text for tokenization."""
        # Convert to lowercase
        text = text.lower()
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Basic tokenization by splitting on spaces
        tokens = text.split()
        
        return tokens
    
    def learn_from_corpus(self, corpus, epochs=1):
        """
        Learn statistics from text corpus.
        
        Args:
            corpus: List of text documents
            epochs: Number of training epochs
        """
        logger.info(f"Learning statistics from corpus of {len(corpus)} documents")
        
        # First pass: count tokens
        for doc in corpus:
            tokens = self.preprocess_text(doc)
            self.token_counts.update(tokens)
            self.total_tokens += len(tokens)
        
        # Filter low-frequency tokens
        vocab = {token for token, count in self.token_counts.items() if count >= self.min_count}
        self.vocabulary_size = len(vocab)
        
        # Create token mappings
        self.token_to_idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        
        logger.info(f"Vocabulary size after filtering: {self.vocabulary_size}")
        
        # Second pass: learn co-occurrences and n-grams
        for epoch in range(epochs):
            logger.info(f"Training epoch {epoch+1}/{epochs}")
            
            for doc in corpus:
                tokens = self.preprocess_text(doc)
                
                # Process only tokens in vocabulary
                tokens = [t for t in tokens if t in self.token_to_idx]
                
                if not tokens:
                    continue
                
                # Update unigrams
                self.unigrams.update(tokens)
                
                # Update bigrams
                for i in range(len(tokens) - 1):
                    bigram = (tokens[i], tokens[i+1])
                    self.bigrams[bigram] += 1
                
                # Update trigrams
                for i in range(len(tokens) - 2):
                    trigram = (tokens[i], tokens[i+1], tokens[i+2])
                    self.trigrams[trigram] += 1
                
                # Update co-occurrence matrix
                for i, token1 in enumerate(tokens):
                    if token1 not in self.token_to_idx:
                        continue
                        
                    # Consider window of +/- window_size tokens
                    window_start = max(0, i - self.window_size)
                    window_end = min(len(tokens), i + self.window_size + 1)
                    
                    for j in range(window_start, window_end):
                        if i == j:
                            continue
                            
                        token2 = tokens[j]
                        if token2 not in self.token_to_idx:
                            continue
                        
                        # Weight by distance (closer tokens have stronger co-occurrence)
                        weight = 1.0 / max(1, abs(i - j))
                        
                        # Update co-occurrence
                        idx1 = self.token_to_idx[token1]
                        idx2 = self.token_to_idx[token2]
                        self.cooccurrence_matrix[(idx1, idx2)] += weight
        
        # Calculate probabilities
        self._calculate_probabilities()
        
        # Learn vector representations
        self._learn_vectors()
        
        self.is_trained = True
        logger.info(f"Learned statistics for {self.vocabulary_size} tokens")
        return self
    
    def _calculate_probabilities(self):
        """Calculate conditional and joint probabilities from co-occurrence data."""
        # Calculate total co-occurrences for normalization
        total_cooccurrences = sum(self.cooccurrence_matrix.values())
        
        # Calculate conditional probabilities P(token2|token1)
        for (idx1, idx2), count in self.cooccurrence_matrix.items():
            # Get tokens
            token1 = self.idx_to_token[idx1]
            token2 = self.idx_to_token[idx2]
            
            # P(token2|token1) = Count(token1, token2) / Count(token1)
            self.conditional_probs[(token1, token2)] = count / max(1, self.unigrams[token1])
            
            # P(token1, token2) = Count(token1, token2) / Total
            self.joint_probs[(token1, token2)] = count / max(1, total_cooccurrences)
    
    def _learn_vectors(self):
        """Learn vector representations based on statistics (simplified GloVe-like approach)."""
        # Initialize random vectors
        self.embedding_matrix = torch.randn(self.vocabulary_size, self.dim, device=self.device) * 0.1
        
        # In a full implementation, we would use GloVe or Word2Vec training
        # For this simplified version, we'll use co-occurrence statistics directly
        
        # Convert co-occurrence matrix to dense format for processing
        cooc_matrix = torch.zeros((self.vocabulary_size, self.vocabulary_size), device=self.device)
        for (idx1, idx2), value in self.cooccurrence_matrix.items():
            cooc_matrix[idx1, idx2] = value
        
        # Apply log and normalize
        log_cooc = torch.log(cooc_matrix + 1.0)
        
        # Simple dimensionality reduction using SVD
        try:
            U, S, V = torch.svd(log_cooc)
            # Take top components
            self.embedding_matrix = U[:, :self.dim] * torch.sqrt(S[:self.dim].unsqueeze(0))
            
            # Now map to token_vectors
            for token, idx in self.token_to_idx.items():
                self.token_vectors[token] = self.embedding_matrix[idx]
                
            logger.info(f"Learned {self.dim}-dimensional vectors using SVD")
        except Exception as e:
            logger.warning(f"SVD failed: {e}. Using random vectors instead.")
            # Fallback to random vectors
            for token, idx in self.token_to_idx.items():
                self.token_vectors[token] = torch.randn(self.dim, device=self.device) * 0.1
    
    def calculate_likelihood(self, sequence):
        """
        Calculate likelihood of a token sequence based on learned statistics.
        
        Args:
            sequence: String or list of tokens
            
        Returns:
            Likelihood score between 0 and 1
        """
        if not self.is_trained:
            logger.warning("Cannot calculate likelihood: model not trained")
            return 0.5
            
        # Preprocess input
        if isinstance(sequence, str):
            tokens = self.preprocess_text(sequence)
        else:
            tokens = sequence
            
        if len(tokens) <= 1:
            return 1.0
        
        # Use different approaches based on sequence length
        if len(tokens) == 2:
            return self._calculate_bigram_likelihood(tokens)
        elif len(tokens) == 3:
            return self._calculate_trigram_likelihood(tokens)
        else:
            return self._calculate_ngram_likelihood(tokens)
    
    def _calculate_bigram_likelihood(self, tokens):
        """Calculate likelihood using bigram statistics."""
        token1, token2 = tokens
        
        # Get conditional probability P(token2|token1)
        cond_prob = self.conditional_probs.get((token1, token2), 0)
        
        # If we've never seen this bigram, use a small non-zero value
        if cond_prob == 0:
            # Fall back to unigram probability with a penalty
            if token2 in self.unigrams:
                cond_prob = 0.01 * (self.unigrams[token2] / max(1, self.total_tokens))
            else:
                cond_prob = 0.001  # Very unlikely
        
        return cond_prob
    
    def _calculate_trigram_likelihood(self, tokens):
        """Calculate likelihood using trigram statistics."""
        # Check if we've seen this trigram
        trigram = tuple(tokens)
        if trigram in self.trigrams:
            # Direct trigram probability
            tri_count = self.trigrams[trigram]
            bigram_count = self.bigrams.get((tokens[0], tokens[1]), 0)
            
            if bigram_count > 0:
                return tri_count / bigram_count
        
        # Fall back to bigram approach
        bigram_probs = []
        for i in range(len(tokens) - 1):
            bigram_prob = self._calculate_bigram_likelihood(tokens[i:i+2])
            bigram_probs.append(bigram_prob)
        
        # Geometric mean of bigram probabilities
        return np.exp(np.mean(np.log([max(p, 1e-10) for p in bigram_probs])))
    
    def _calculate_ngram_likelihood(self, tokens):
        """Calculate likelihood for longer sequences using a sliding window approach."""
        # Use a sliding window of trigrams and bigrams
        log_probs = []
        
        # Trigram probabilities
        for i in range(len(tokens) - 2):
            trigram = tuple(tokens[i:i+3])
            if trigram in self.trigrams:
                bigram = (tokens[i], tokens[i+1])
                tri_prob = self.trigrams[trigram] / max(1, self.bigrams[bigram])
                log_probs.append(np.log(max(tri_prob, 1e-10)))
            else:
                # Fall back to bigram
                bigram = (tokens[i], tokens[i+1])
                bi_prob = self.bigrams.get(bigram, 0) / max(1, self.unigrams[tokens[i]])
                log_probs.append(np.log(max(bi_prob, 1e-10)))
        
        # Add final bigram if needed
        if len(tokens) >= 2:
            final_bigram = (tokens[-2], tokens[-1])
            bi_prob = self.bigrams.get(final_bigram, 0) / max(1, self.unigrams[tokens[-2]])
            log_probs.append(np.log(max(bi_prob, 1e-10)))
        
        # Calculate overall probability
        if log_probs:
            avg_log_prob = sum(log_probs) / len(log_probs)
            likelihood = np.exp(avg_log_prob)
            
            # Scale to 0-1 range
            return min(1.0, max(0.0, likelihood * 5))  # Scaling factor to make more useful
        else:
            return 0.001  # Very unlikely
    
    def get_vector(self, token):
        """Get statistical vector for a token."""
        if not self.is_trained:
            return torch.zeros(self.dim, device=self.device)
            
        # Preprocess token
        token = token.lower() if isinstance(token, str) else token
        
        # Return vector if available
        if token in self.token_vectors:
            return self.token_vectors[token]
            
        # For unknown tokens, check similarity with known tokens
        # For simplicity, we'll return zeros, but in a full implementation
        # we would have subword handling
        return torch.zeros(self.dim, device=self.device)
    
    def combine_vectors(self, tokens):
        """
        Combine vectors for a sequence of tokens.
        
        Args:
            tokens: List of tokens or string
            
        Returns:
            Combined vector representation
        """
        if not self.is_trained:
            return torch.zeros(self.dim, device=self.device)
            
        # Preprocess input
        if isinstance(tokens, str):
            tokens = self.preprocess_text(tokens)
        
        # Get vectors for tokens
        vectors = [self.get_vector(token) for token in tokens]
        
        if not vectors:
            return torch.zeros(self.dim, device=self.device)
        
        # Simple average of vectors
        return torch.mean(torch.stack([v for v in vectors if torch.any(v)]), dim=0)
    
    def most_similar(self, token, n=5):
        """Find most similar tokens based on vector similarity."""
        if not self.is_trained or token not in self.token_vectors:
            return []
            
        query_vector = self.token_vectors[token]
        
        similarities = []
        for other_token, other_vector in self.token_vectors.items():
            if other_token == token:
                continue
                
            # Calculate cosine similarity
            similarity = torch.dot(query_vector, other_vector) / (
                torch.norm(query_vector) * torch.norm(other_vector) + 1e-8
            )
            similarities.append((other_token, similarity.item()))
        
        # Sort by similarity
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]
    
    def predict_next_token(self, context, n=3):
        """Predict most likely next tokens given context."""
        if not self.is_trained:
            return []
            
        # Preprocess context
        if isinstance(context, str):
            context = self.preprocess_text(context)
        
        if not context:
            # With no context, return most common tokens
            return [token for token, _ in self.unigrams.most_common(n)]
        
        # For single token context
        if len(context) == 1:
            token = context[0]
            candidates = []
            
            # Check all bigrams starting with this token
            for (t1, t2), count in self.bigrams.items():
                if t1 == token:
                    candidates.append((t2, count / max(1, self.unigrams[token])))
            
            # Sort and return top n
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [token for token, _ in candidates[:n]]
        
        # For longer context, use last two tokens for trigram prediction
        last_two = context[-2:]
        candidates = []
        
        # Check all trigrams starting with these tokens
        for (t1, t2, t3), count in self.trigrams.items():
            if (t1, t2) == tuple(last_two):
                bigram_count = self.bigrams.get((t1, t2), 0)
                prob = count / max(1, bigram_count)
                candidates.append((t3, prob))
        
        # If no trigram candidates, fall back to bigram with last token
        if not candidates:
            for (t1, t2), count in self.bigrams.items():
                if t1 == context[-1]:
                    candidates.append((t2, count / max(1, self.unigrams[t1])))
        
        # Sort and return top n
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [token for token, _ in candidates[:n]]
    
    def detect_anomalies(self, sequence, threshold=0.1):
        """
        Detect anomalous token sequences based on statistical likelihood.
        
        Args:
            sequence: Token sequence to check
            threshold: Likelihood threshold below which tokens are anomalous
            
        Returns:
            List of (token, position, likelihood) for anomalous tokens
        """
        if not self.is_trained:
            return []
            
        # Preprocess input
        if isinstance(sequence, str):
            tokens = self.preprocess_text(sequence)
        else:
            tokens = sequence
        
        anomalies = []
        
        # Check each position in the sequence
        for i in range(1, len(tokens)):
            # Get context (previous token)
            context = tokens[i-1]
            token = tokens[i]
            
            # Calculate likelihood of this token given context
            if (context, token) in self.conditional_probs:
                likelihood = self.conditional_probs[(context, token)]
            else:
                # Never seen this bigram
                likelihood = 0.001
            
            # Check if anomalous
            if likelihood < threshold:
                anomalies.append((token, i, likelihood))
        
        return anomalies
    
    def save(self, file_path):
        """Save the statistical model to a file."""
        data = {
            'token_counts': dict(self.token_counts),
            'unigrams': dict(self.unigrams),
            'bigrams': dict(self.bigrams),
            'trigrams': dict(self.trigrams),
            'token_to_idx': self.token_to_idx,
            'idx_to_token': self.idx_to_token,
            'vocabulary_size': self.vocabulary_size,
            'total_tokens': self.total_tokens,
            'dim': self.dim,
            'window_size': self.window_size,
            'min_count': self.min_count
        }
        
        # We need to convert defaultdict to dict and tensors to lists
        if self.embedding_matrix is not None:
            data['embedding_matrix'] = self.embedding_matrix.cpu().numpy().tolist()
        
        torch.save(data, file_path)
        logger.info(f"Statistical model saved to {file_path}")
    
    @classmethod
    def load(cls, file_path, device=None):
        """Load a statistical model from a file."""
        data = torch.load(file_path)
        
        # Create instance with saved parameters
        instance = cls(
            dim=data['dim'],
            device=device,
            window_size=data['window_size'],
            min_count=data['min_count']
        )
        
        # Restore data
        instance.token_counts = Counter(data['token_counts'])
        instance.unigrams = Counter(data['unigrams'])
        instance.bigrams = Counter(data['bigrams'])
        instance.trigrams = Counter(data['trigrams'])
        instance.token_to_idx = data['token_to_idx']
        instance.idx_to_token = data['idx_to_token']
        instance.vocabulary_size = data['vocabulary_size']
        instance.total_tokens = data['total_tokens']
        
        # Restore embedding matrix
        if 'embedding_matrix' in data:
            instance.embedding_matrix = torch.tensor(
                data['embedding_matrix'],
                device=instance.device
            )
            
            # Reconstruct token vectors
            for token, idx in instance.token_to_idx.items():
                instance.token_vectors[token] = instance.embedding_matrix[idx]
        
        instance.is_trained = True
        logger.info(f"Statistical model loaded from {file_path}")
        return instance


# Example usage
def example_usage():
    """Demonstrate usage of the statistical vector space."""
    # Create statistical vector space
    stat_space = StatisticalVectorSpace(dim=50)
    
    # Sample corpus
    corpus = [
        "neural networks process information efficiently",
        "deep learning models can recognize patterns",
        "machine learning algorithms require training data",
        "neural networks can learn complex patterns from data",
        "artificial intelligence systems use neural networks",
        "deep neural networks have multiple layers",
        "machine learning models can make predictions"
    ]
    
    # Train on corpus
    stat_space.learn_from_corpus(corpus)
    
    # Calculate likelihood of a valid sequence
    valid_seq = "neural networks process information"
    valid_likelihood = stat_space.calculate_likelihood(valid_seq)
    print(f"Likelihood of '{valid_seq}': {valid_likelihood:.4f}")
    
    # Calculate likelihood of an invalid sequence
    invalid_seq = "neural networks eat pizza"
    invalid_likelihood = stat_space.calculate_likelihood(invalid_seq)
    print(f"Likelihood of '{invalid_seq}': {invalid_likelihood:.4f}")
    
    # Get most similar words
    similarities = stat_space.most_similar("neural", n=3)
    print(f"Most similar to 'neural': {similarities}")
    
    # Predict next token
    next_tokens = stat_space.predict_next_token("neural networks", n=3)
    print(f"After 'neural networks', expect: {next_tokens}")
    
    # Detect anomalies
    anomalies = stat_space.detect_anomalies("neural networks eat data")
    if anomalies:
        print(f"Anomalies detected: {anomalies}")
    else:
        print("No anomalies detected")


if __name__ == "__main__":
    example_usage() 