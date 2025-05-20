#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Tokenizer for SNN Models.

This module provides subword tokenization capabilities for the SNN models,
supporting BPE (Byte-Pair Encoding) and WordPiece tokenization methods.
"""

import os
import re
import json
import pickle
import logging
import collections
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdvancedTokenizer")

class AdvancedTokenizer:
    """
    Advanced tokenizer with subword tokenization capabilities.
    
    Features:
    - BPE (Byte-Pair Encoding) tokenization
    - WordPiece tokenization
    - Support for special tokens
    - Vocabulary management
    - Token merging for decoding
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        method: str = "bpe",
        special_tokens: List[str] = None,
        device: Optional[str] = None,
        model_type: str = "generic"
    ):
        """
        Initialize the advanced tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            method: Tokenization method ("bpe" or "wordpiece")
            special_tokens: Special tokens to include in vocabulary
            device: Device to run operations on ('cpu', 'cuda', etc.) or None for auto-detection
            model_type: Type of SNN model for specific tokenization strategies
        """
        # Set device for GPU acceleration - removed but kept for future implementation
        # self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        # logger.info(f"Tokenizer using device: {self.device}")
        
        self.vocab_size = vocab_size
        self.method = method.lower()
        self.model_type = model_type.lower()
        
        # Default special tokens if not provided
        if special_tokens is None:
            # Core special tokens for all models
            self.special_tokens = [
                "<PAD>",  # Padding token
                "<UNK>",  # Unknown token
                "<BOS>",  # Beginning of sequence
                "<EOS>",  # End of sequence
                "<MASK>", # Mask token for training
                "<SEP>",  # Separator token
                "<CLS>"   # Classification token
            ]
            
            # Add model-specific special tokens
            if self.model_type == "memory":
                self.special_tokens.extend([
                    "<RECALL>",  # Memory recall token
                    "<STORE>",   # Memory storage token
                    "<FORGET>"   # Memory forgetting token
                ])
            elif self.model_type == "perceptual":
                self.special_tokens.extend([
                    "<VISUAL>",  # Visual feature token
                    "<AUDIO>",   # Audio feature token
                    "<TACTILE>"  # Tactile feature token
                ])
            elif self.model_type == "reasoning":
                self.special_tokens.extend([
                    "<IF>",      # Conditional token
                    "<THEN>",    # Consequence token
                    "<ELSE>",    # Alternative token
                    "<AND>",     # Logical AND token
                    "<OR>",      # Logical OR token
                    "<NOT>"      # Logical NOT token
                ])
            elif self.model_type == "decision":
                self.special_tokens.extend([
                    "<OPTION>",   # Decision option token
                    "<VALUE>",    # Value assessment token
                    "<CHOICE>",   # Final choice token
                    "<UTILITY>"   # Utility calculation token
                ])
            elif self.model_type == "affective":
                self.special_tokens.extend([
                    "<EMOTION>",  # Emotion indicator token
                    "<MOOD>",     # Mood state token
                    "<VALENCE>",  # Positive/negative valence token
                    "<AROUSAL>"   # Emotional intensity token
                ])
            elif self.model_type == "metacognitive":
                self.special_tokens.extend([
                    "<MONITOR>",   # Self-monitoring token
                    "<EVALUATE>",  # Evaluation token
                    "<REGULATE>",  # Self-regulation token
                    "<CONFIDENCE>" # Confidence assessment token
                ])
            elif self.model_type == "statistical":
                self.special_tokens.extend([
                    "<PATTERN>",   # Pattern recognition token
                    "<FREQUENCY>", # Frequency assessment token 
                    "<PREDICT>",   # Prediction token
                    "<INFER>"      # Statistical inference token
                ])
        else:
            self.special_tokens = special_tokens
        
        # Initialize vocabulary
        self.token_to_id = {}  # Token to ID mapping
        self.id_to_token = {}  # ID to token mapping
        self.byte_pairs = {}   # Byte pair merges for BPE
        self.word_freqs = {}   # Word frequencies for training
        
        # Add special tokens to vocabulary
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        # Initialize tokenization regex patterns
        self._init_patterns()
        
        # Training state
        self.is_trained = False
    
    def _init_patterns(self):
        """Initialize regex patterns for tokenization"""
        # Common word boundary pattern
        self.word_pattern = re.compile(r'\b\w+\b|[^\w\s]')
        
        # Unicode character pattern
        self.unicode_pattern = re.compile(r'.')
        
        # Whitespace pattern
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Lowercase pattern (for case insensitivity)
        self.lowercase_pattern = re.compile(r'[A-Z]')
        
        # Model-specific patterns
        if self.model_type == "memory":
            # Pattern for memory-related tokens
            self.memory_pattern = re.compile(r'\b(recall|remember|forget|store)\b', re.IGNORECASE)
        
        elif self.model_type == "perceptual":
            # Pattern for perceptual features
            self.perceptual_pattern = re.compile(r'\b(see|hear|feel|touch|visual|audio|tactile)\b', re.IGNORECASE)
        
        elif self.model_type == "reasoning":
            # Pattern for logical operators
            self.logical_pattern = re.compile(r'\b(if|then|else|and|or|not)\b', re.IGNORECASE)
            
        elif self.model_type == "decision":
            # Pattern for decision-making terms
            self.decision_pattern = re.compile(r'\b(choose|decide|option|select|prefer|value|utility)\b', re.IGNORECASE)
            
        elif self.model_type == "affective":
            # Pattern for emotion and mood terms
            self.affective_pattern = re.compile(r'\b(happy|sad|angry|fear|joy|disgust|surprise|emotion|mood|feel)\b', re.IGNORECASE)
            
        elif self.model_type == "metacognitive":
            # Pattern for metacognitive terms
            self.metacognitive_pattern = re.compile(r'\b(think|evaluate|monitor|regulate|confidence|certain|uncertain)\b', re.IGNORECASE)
            
        elif self.model_type == "statistical":
            # Pattern for statistical and probabilistic terms
            self.statistical_pattern = re.compile(r'\b(pattern|frequency|likely|probability|predict|infer|correlation)\b', re.IGNORECASE)
    
    def train(self, texts: List[str], min_freq: int = 2, num_merges: int = 10000):
        """
        Train the tokenizer on a corpus of texts.
        
        Args:
            texts: List of text samples for training
            min_freq: Minimum frequency for a token to be included
            num_merges: Number of merge operations for BPE
            
        Returns:
            Number of tokens in the final vocabulary
        """
        logger.info(f"Training tokenizer with method: {self.method}, target vocab size: {self.vocab_size}")
        
        # Collect word frequencies
        self.word_freqs = collections.Counter()
        for text in texts:
            words = self._split_text(text)
            self.word_freqs.update(words)
        
        # Filter low-frequency words
        self.word_freqs = {word: count for word, count in self.word_freqs.items()
                          if count >= min_freq}
        
        if self.method == "bpe":
            # Train BPE tokenizer
            return self._train_bpe(num_merges)
        elif self.method == "wordpiece":
            # Train WordPiece tokenizer
            return self._train_wordpiece()
        else:
            logger.warning(f"Unknown method: {self.method}, falling back to BPE")
            return self._train_bpe(num_merges)
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into words for initial tokenization"""
        # Replace whitespace
        text = re.sub(self.whitespace_pattern, ' ', text)
        # Find all words and punctuation
        words = self.word_pattern.findall(text)
        return words
    
    def _train_bpe(self, num_merges: int) -> int:
        """
        Train BPE tokenizer.
        
        Args:
            num_merges: Number of merge operations
            
        Returns:
            Vocabulary size
        """
        logger.info("Training BPE tokenizer...")
        
        # Start with characters as initial vocabulary
        char_vocab = set()
        for word in self.word_freqs.keys():
            for char in word:
                char_vocab.add(char)
        
        # Initialize vocabulary with special tokens and characters
        self.token_to_id = {token: i for i, token in enumerate(self.special_tokens)}
        next_id = len(self.special_tokens)
        
        for char in sorted(char_vocab):
            self.token_to_id[char] = next_id
            next_id += 1
        
        # Update id_to_token mapping
        self.id_to_token = {id: token for token, id in self.token_to_id.items()}
        
        # Initialize words as sequences of characters
        words = {word: list(word) for word in self.word_freqs.keys()}
        
        # Perform BPE merge operations
        merges = {}
        for i in range(min(num_merges, self.vocab_size - len(self.token_to_id))):
            # Count pair frequencies
            pair_freqs = collections.Counter()
            for word, freq in self.word_freqs.items():
                word_pieces = words[word]
                if len(word_pieces) < 2:
                    continue
                
                for j in range(len(word_pieces) - 1):
                    pair = (word_pieces[j], word_pieces[j + 1])
                    pair_freqs[pair] += freq
            
            if not pair_freqs:
                break
                
            # Find most frequent pair
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            
            # Create new token
            new_token = ''.join(best_pair)
            
            # Add to vocabulary if not already present
            if new_token not in self.token_to_id:
                self.token_to_id[new_token] = next_id
                self.id_to_token[next_id] = new_token
                next_id += 1
            
            # Record the merge
            merges[best_pair] = new_token
            
            # Apply the merge to all words
            for word in words:
                word_pieces = words[word]
                idx = 0
                while idx < len(word_pieces) - 1:
                    if (word_pieces[idx], word_pieces[idx + 1]) == best_pair:
                        word_pieces[idx] = new_token
                        word_pieces.pop(idx + 1)
                    else:
                        idx += 1
            
            logger.debug(f"Merge #{i+1}/{num_merges}: {best_pair} -> {new_token}")
            
            if len(self.token_to_id) >= self.vocab_size:
                logger.info(f"Reached target vocabulary size: {self.vocab_size}")
                break
        
        # Store merges for encoding
        self.byte_pairs = merges
        
        # Mark as trained
        self.is_trained = True
        
        logger.info(f"BPE training complete. Vocabulary size: {len(self.token_to_id)}")
        return len(self.token_to_id)
    
    def _train_wordpiece(self) -> int:
        """
        Train WordPiece tokenizer.
        
        Returns:
            Vocabulary size
        """
        logger.info("Training WordPiece tokenizer...")
        
        # Start with characters and special tokens in vocabulary
        char_vocab = set()
        for word in self.word_freqs.keys():
            for char in word:
                char_vocab.add(char)
        
        # Initialize vocabulary with special tokens and characters
        self.token_to_id = {token: i for i, token in enumerate(self.special_tokens)}
        next_id = len(self.special_tokens)
        
        for char in sorted(char_vocab):
            self.token_to_id[char] = next_id
            next_id += 1
        
        # Add common prefixes and suffixes
        # We'll start with whole words, then iteratively split based on likelihood scores
        
        # Initialize with most common words
        word_scores = {word: freq for word, freq in self.word_freqs.items()}
        
        # Add most frequent words directly to vocabulary
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        remaining_slots = self.vocab_size - len(self.token_to_id)
        
        for word, _ in sorted_words[:remaining_slots]:
            if word not in self.token_to_id:
                self.token_to_id[word] = next_id
                self.id_to_token[next_id] = word
                next_id += 1
        
        # Mark as trained
        self.is_trained = True
        
        logger.info(f"WordPiece training complete. Vocabulary size: {len(self.token_to_id)}")
        return len(self.token_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
        """
        # Apply model-specific preprocessing
        text = self._preprocess_text(text)
        
        if self.method == "bpe":
            return self._bpe_encode(text)
        elif self.method == "wordpiece":
            return self._wordpiece_encode(text)
        else:
            return self._fallback_encode(text)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before tokenization for model-specific features.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Replace whitespace
        text = re.sub(self.whitespace_pattern, ' ', text)
        
        # Apply model-specific preprocessing
        if self.model_type == "memory":
            # Replace memory-related terms with special tokens
            text = re.sub(r'\b(recall|remember)\b', ' <RECALL> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(store|save)\b', ' <STORE> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(forget|delete)\b', ' <FORGET> ', text, flags=re.IGNORECASE)
            
        elif self.model_type == "perceptual":
            # Replace perceptual terms with special tokens
            text = re.sub(r'\b(see|visual|image)\b', ' <VISUAL> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(hear|audio|sound)\b', ' <AUDIO> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(feel|touch|tactile)\b', ' <TACTILE> ', text, flags=re.IGNORECASE)
            
        elif self.model_type == "reasoning":
            # Replace logical operators with special tokens
            text = re.sub(r'\b(if)\b', ' <IF> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(then)\b', ' <THEN> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(else)\b', ' <ELSE> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(and)\b', ' <AND> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(or)\b', ' <OR> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(not)\b', ' <NOT> ', text, flags=re.IGNORECASE)
            
        elif self.model_type == "decision":
            # Replace decision-making terms with special tokens
            text = re.sub(r'\b(option|choice|alternative)\b', ' <OPTION> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(value|worth|importance)\b', ' <VALUE> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(choose|select|decide)\b', ' <CHOICE> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(utility|benefit|cost)\b', ' <UTILITY> ', text, flags=re.IGNORECASE)
            
        elif self.model_type == "affective":
            # Replace emotion-related terms with special tokens
            text = re.sub(r'\b(emotion|feeling)\b', ' <EMOTION> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(mood|temperament)\b', ' <MOOD> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(positive|negative|valence)\b', ' <VALENCE> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(intensity|arousal|activation)\b', ' <AROUSAL> ', text, flags=re.IGNORECASE)
            
        elif self.model_type == "metacognitive":
            # Replace metacognitive terms with special tokens
            text = re.sub(r'\b(monitor|observe|watch)\b', ' <MONITOR> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(evaluate|judge|assess)\b', ' <EVALUATE> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(regulate|control|adjust)\b', ' <REGULATE> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(confidence|certainty|uncertainty)\b', ' <CONFIDENCE> ', text, flags=re.IGNORECASE)
            
        elif self.model_type == "statistical":
            # Replace statistical terms with special tokens
            text = re.sub(r'\b(pattern|structure|regularity)\b', ' <PATTERN> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(frequency|occurrence|count)\b', ' <FREQUENCY> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(predict|forecast|anticipate)\b', ' <PREDICT> ', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(infer|deduce|conclude|induce)\b', ' <INFER> ', text, flags=re.IGNORECASE)
        
        return text
    
    def _bpe_encode(self, text: str) -> List[int]:
        """
        Encode text using BPE tokenization.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        token_ids = []
        
        # Split text into words
        words = self._split_text(text)
        
        for word in words:
            # Handle unknown words or empty strings
            if not word:
                continue
                
            # Start with character sequence
            word_pieces = list(word)
            
            # Apply merges
            idx = 0
            while idx < len(word_pieces) - 1:
                current_pair = (word_pieces[idx], word_pieces[idx + 1])
                
                if current_pair in self.byte_pairs:
                    # Apply merge
                    merged_token = self.byte_pairs[current_pair]
                    word_pieces[idx] = merged_token
                    word_pieces.pop(idx + 1)
                    
                    # Reset to check for further merges
                    if idx > 0:
                        idx -= 1
                else:
                    # Move to next position
                    idx += 1
            
            # Convert to token IDs
            for piece in word_pieces:
                token_id = self.token_to_id.get(piece, self.token_to_id["<UNK>"])
                token_ids.append(token_id)
        
        return token_ids
    
    def _wordpiece_encode(self, text: str) -> List[int]:
        """
        Encode text using WordPiece tokenization.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        token_ids = []
        
        # Split text into words
        words = self._split_text(text)
        
        for word in words:
            # Handle unknown words or empty strings
            if not word:
                continue
                
            # Check if word is in vocabulary
            if word in self.token_to_id:
                token_ids.append(self.token_to_id[word])
                continue
            
            # Greedy longest-match-first algorithm
            chars = list(word)
            idx = 0
            subwords = []
            
            while idx < len(chars):
                # Try to find the longest subword starting from the current position
                current = chars[idx]
                longest_token = None
                longest_end = idx
                
                for j in range(idx + 1, len(chars) + 1):
                    subword = ''.join(chars[idx:j])
                    if subword in self.token_to_id:
                        longest_token = subword
                        longest_end = j
                
                if longest_token:
                    subwords.append(longest_token)
                    idx = longest_end
                else:
                    # Unknown character, add as individual token
                    subwords.append(current)
                    idx += 1
            
            # Convert subwords to token IDs
            for subword in subwords:
                token_id = self.token_to_id.get(subword, self.token_to_id["<UNK>"])
                token_ids.append(token_id)
        
        return token_ids
    
    def _fallback_encode(self, text: str) -> List[int]:
        """
        Fallback character-level encoding.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        token_ids = []
        
        for char in text:
            if char in self.token_to_id:
                token_ids.append(self.token_to_id[char])
            else:
                token_ids.append(self.token_to_id["<UNK>"])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        if not token_ids:
            return ""
        
        tokens = [self.id_to_token.get(id, "<UNK>") for id in token_ids]
        
        if self.method == "bpe":
            return self._bpe_decode(tokens)
        elif self.method == "wordpiece":
            return self._wordpiece_decode(tokens)
        else:
            return ''.join(tokens)
    
    def _bpe_decode(self, tokens: List[str]) -> str:
        """
        Decode BPE tokens to text.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Decoded text
        """
        # Simply concatenate all tokens (BPE merges preserve character sequences)
        text = ''.join(tokens)
        
        # Remove special tokens
        for special in self.special_tokens:
            text = text.replace(special, '')
        
        return text
    
    def _wordpiece_decode(self, tokens: List[str]) -> str:
        """
        Decode WordPiece tokens to text.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Decoded text
        """
        # Remove special tokens
        filtered_tokens = [t for t in tokens if t not in self.special_tokens]
        
        # Join tokens with space
        text = ' '.join(filtered_tokens)
        
        # Remove any internal spaces added by decoding
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens (without converting to IDs).
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        token_ids = self.encode(text)
        return [self.id_to_token.get(id, "<UNK>") for id in token_ids]
    
    def save(self, filepath: str) -> bool:
        """
        Save tokenizer to file.
        
        Args:
            filepath: Path to save tokenizer
            
        Returns:
            Success status
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        try:
            # Prepare data to save
            tokenizer_data = {
                'config': {
                    'vocab_size': self.vocab_size,
                    'method': self.method,
                    'special_tokens': self.special_tokens
                },
                'token_to_id': self.token_to_id,
                'id_to_token': {int(k): v for k, v in self.id_to_token.items()},
                'byte_pairs': {str(k): v for k, v in self.byte_pairs.items()},
                'is_trained': self.is_trained
            }
            
            # Save to file based on extension
            file_ext = os.path.splitext(filepath)[1].lower()
            
            if file_ext == '.json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
            elif file_ext in ['.pkl', '.pickle']:
                with open(filepath, 'wb') as f:
                    pickle.dump(tokenizer_data, f)
            else:
                # Default to JSON if extension is unknown
                with open(f"{filepath}.json", 'w', encoding='utf-8') as f:
                    json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
                    filepath = f"{filepath}.json"
            
            logger.info(f"Tokenizer saved to {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving tokenizer: {e}")
            return False
    
    def load(self, filepath: str) -> bool:
        """
        Load tokenizer from file.
        
        Args:
            filepath: Path to load tokenizer from
            
        Returns:
            Success status
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return False
        
        try:
            # Load based on file extension
            file_ext = os.path.splitext(filepath)[1].lower()
            
            if file_ext == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    tokenizer_data = json.load(f)
            elif file_ext in ['.pkl', '.pickle']:
                with open(filepath, 'rb') as f:
                    tokenizer_data = pickle.load(f)
            else:
                logger.error(f"Unsupported file extension: {file_ext}")
                return False
            
            # Load configuration
            config = tokenizer_data.get('config', {})
            self.vocab_size = config.get('vocab_size', self.vocab_size)
            self.method = config.get('method', self.method)
            self.special_tokens = config.get('special_tokens', self.special_tokens)
            
            # Load mappings
            self.token_to_id = tokenizer_data.get('token_to_id', {})
            self.id_to_token = {int(k): v for k, v in tokenizer_data.get('id_to_token', {}).items()}
            
            # Load byte pairs (handling tuple keys)
            byte_pairs_str = tokenizer_data.get('byte_pairs', {})
            self.byte_pairs = {}
            for k, v in byte_pairs_str.items():
                try:
                    # Handle string representation of tuple
                    if k.startswith('(') and k.endswith(')'):
                        # This is a crude way to parse the tuple string, but it works for simple cases
                        parts = k.strip('()').split(',')
                        if len(parts) == 2:
                            key = (parts[0].strip(" '"), parts[1].strip(" '"))
                            self.byte_pairs[key] = v
                    else:
                        # Just store as-is
                        self.byte_pairs[k] = v
                except Exception:
                    # If parsing fails, use string key
                    self.byte_pairs[k] = v
            
            # Load training state
            self.is_trained = tokenizer_data.get('is_trained', False)
            
            logger.info(f"Tokenizer loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            return False

# Utility function to create a tokenizer
def create_tokenizer(method="bpe", vocab_size=50000, model_type="generic"):
    """
    Factory function to create and return a new tokenizer instance.
    
    Args:
        method: Tokenization method ("bpe" or "wordpiece")
        vocab_size: Maximum vocabulary size
        model_type: Type of SNN model for specific tokenization strategies
    
    Returns:
        New AdvancedTokenizer instance
    """
    logger.info(f"Creating new tokenizer with method: {method}, vocab_size: {vocab_size}, model_type: {model_type}")
    return AdvancedTokenizer(vocab_size=vocab_size, method=method, model_type=model_type)

# Simple demo/test function
def test_tokenizer():
    """
    Test function to demonstrate the tokenizer's capabilities.
    """
    test_texts = [
        "This is a simple test to see if the tokenizer works properly.",
        "Memory recall and storage mechanisms are critical for long-term learning.",
        "Visual perception combines with auditory processing for multimodal understanding.",
        "If logical reasoning is applied, then better decisions can be made.",
        "Choose between these options based on their utility and value.",
        "Emotions and mood affect how we perceive and respond to the world.",
        "Monitoring our thinking process helps evaluate and regulate cognition.",
        "Statistical patterns help predict future events by inferring frequencies."
    ]
    
    # Test each model type with relevant text
    model_types = [
        "generic", "memory", "perceptual", "reasoning", 
        "decision", "affective", "metacognitive", "statistical"
    ]
    
    print("Testing AdvancedTokenizer with different model types:")
    print("-" * 60)
    
    for i, model_type in enumerate(model_types):
        # Create tokenizer for this model type
        tokenizer = create_tokenizer(method="bpe", vocab_size=1000, model_type=model_type)
        
        # Use a text that's relevant for this model type
        test_text = test_texts[min(i, len(test_texts) - 1)]
        
        # Tokenize the text
        tokens = tokenizer.tokenize(test_text)
        token_ids = tokenizer.encode(test_text)
        decoded = tokenizer.decode(token_ids)
        
        # Print results
        print(f"\nModel type: {model_type}")
        print(f"Test text: {test_text}")
        print(f"Tokens: {tokens[:10]}...")
        print(f"Token IDs: {token_ids[:10]}...")
        print(f"Decoded: {decoded}")
        print("-" * 60)
    
    print("Tokenizer test completed.")

if __name__ == "__main__":
    test_tokenizer() 