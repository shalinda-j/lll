"""
Tokenizer module implementing Layers 0-3: Binary Foundation to ASCII Mapping.

Layer 0: Binary Foundation - all data as bit arrays (0/1)
Layer 1: Byte Encoding - binary (8 bits) → byte integer (0-255)
Layer 2: ASCII Mapping - byte integers to ASCII characters (0-127)
Layer 3: Frequency Sorting - characters sorted by frequency
"""

from typing import List, Dict, Tuple, Optional
import torch


class BinaryTokenizer:
    """
    Tokenizer implementing the binary-to-ASCII pipeline (Layers 0-3).

    This tokenizer processes text through:
    1. Converting characters to binary representation
    2. Grouping bits into bytes
    3. Mapping to ASCII character IDs
    4. Optionally sorting by frequency for efficient encoding
    """

    def __init__(self, vocab_size: int = 128):
        """
        Initialize the tokenizer.

        Args:
            vocab_size: Size of vocabulary (default 128 for ASCII)
        """
        self.vocab_size = vocab_size

        # Build character mappings (Layer 2: ASCII Mapping)
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

        # Initialize with printable ASCII characters
        for i in range(vocab_size):
            char = chr(i)
            self.char_to_id[char] = i
            self.id_to_char[i] = char

        # Frequency tracking for Layer 3
        self.char_frequencies: Dict[str, int] = {chr(i): 0 for i in range(vocab_size)}

        # Word boundary characters (for Zone C)
        self.word_boundaries = {' ', '\n', '\t', '.', ',', '!', '?', ';', ':', '(', ')'}

    # ==================== LAYER 0: BINARY FOUNDATION ====================

    def char_to_bits(self, char: str) -> List[int]:
        """
        Convert a single character to its 8-bit binary representation.

        Layer 0: Binary Foundation
        Every character is represented as 8 bits (0 or 1).

        Args:
            char: Single character to convert

        Returns:
            List of 8 integers (0 or 1)
        """
        if len(char) != 1:
            raise ValueError("Input must be a single character")

        byte_val = ord(char)
        bits = []
        for i in range(7, -1, -1):
            bits.append((byte_val >> i) & 1)
        return bits

    def text_to_bits(self, text: str) -> List[int]:
        """
        Convert entire text to a flat list of bits.

        Args:
            text: Input string

        Returns:
            Flat list of bits (0 or 1)
        """
        bits = []
        for char in text:
            bits.extend(self.char_to_bits(char))
        return bits

    # ==================== LAYER 1: BYTE ENCODING ====================

    def bits_to_byte(self, bits: List[int]) -> int:
        """
        Convert 8 bits to a byte integer (0-255).

        Layer 1: Byte Encoding
        Groups of 8 bits become byte integers.

        Args:
            bits: List of exactly 8 bits

        Returns:
            Integer in range 0-255
        """
        if len(bits) != 8:
            raise ValueError(f"Expected 8 bits, got {len(bits)}")

        byte_val = 0
        for i, bit in enumerate(bits):
            byte_val |= (bit & 1) << (7 - i)
        return byte_val

    def bits_to_bytes(self, bits: List[int]) -> List[int]:
        """
        Convert a list of bits to a list of byte integers.

        Args:
            bits: List of bits (length must be multiple of 8)

        Returns:
            List of byte integers (0-255)
        """
        if len(bits) % 8 != 0:
            # Pad with zeros if needed
            bits = bits + [0] * (8 - len(bits) % 8)

        bytes_list = []
        for i in range(0, len(bits), 8):
            bytes_list.append(self.bits_to_byte(bits[i:i+8]))
        return bytes_list

    # ==================== LAYER 2: ASCII MAPPING ====================

    def byte_to_ascii_id(self, byte_val: int) -> int:
        """
        Map a byte integer to ASCII character ID.

        Layer 2: ASCII Mapping
        Maps byte integers (0-127) to character IDs.
        Values > 127 are clamped to valid ASCII range.

        Args:
            byte_val: Integer in range 0-255

        Returns:
            ASCII character ID (0-127)
        """
        return min(byte_val, self.vocab_size - 1)

    def bytes_to_ascii(self, bytes_list: List[int]) -> List[int]:
        """
        Convert bytes to ASCII character IDs.

        Args:
            bytes_list: List of byte integers

        Returns:
            List of ASCII character IDs
        """
        return [self.byte_to_ascii_id(b) for b in bytes_list]

    # ==================== LAYER 3: FREQUENCY SORTING ====================

    def update_frequencies(self, text: str) -> None:
        """
        Update character frequency counts from text.

        Layer 3: Frequency Sorting
        Track character frequencies for potential vocabulary optimization.

        Args:
            text: Input text to analyze
        """
        for char in text:
            if char in self.char_frequencies:
                self.char_frequencies[char] += 1

    def get_frequency_sorted_vocab(self) -> List[Tuple[str, int]]:
        """
        Get vocabulary sorted by character frequency.

        Returns:
            List of (character, frequency) tuples, sorted by frequency (descending)
        """
        sorted_vocab = sorted(
            self.char_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_vocab

    # ==================== MAIN ENCODING/DECODING METHODS ====================

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text to tensor of character IDs.

        This is the main entry point for tokenization.
        Goes through the full pipeline: text → bits → bytes → ASCII IDs.

        Args:
            text: Input string

        Returns:
            Tensor of shape (seq_len,) with character IDs
        """
        if not text:
            return torch.tensor([], dtype=torch.long)

        # Update frequencies for Layer 3
        self.update_frequencies(text)

        # Direct encoding (skip bit manipulation for efficiency)
        char_ids = []
        for char in text:
            char_id = self.char_to_id.get(char, 0)  # Default to 0 for unknown
            char_ids.append(char_id)

        return torch.tensor(char_ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        """
        Decode tensor of character IDs back to text.

        Args:
            ids: Tensor of character IDs

        Returns:
            Decoded string
        """
        if ids.numel() == 0:
            return ""

        ids_list = ids.tolist()
        chars = []
        for char_id in ids_list:
            char = self.id_to_char.get(int(char_id), chr(int(char_id)))
            chars.append(char)
        return ''.join(chars)

    def encode_with_bits(self, text: str) -> Tuple[torch.Tensor, List[int]]:
        """
        Full encoding pipeline returning both IDs and raw bits.

        Useful for debugging and understanding the binary foundation.

        Args:
            text: Input string

        Returns:
            Tuple of (character_ids tensor, raw bits list)
        """
        bits = self.text_to_bits(text)
        bytes_list = self.bits_to_bytes(bits)
        ascii_ids = self.bytes_to_ascii(bytes_list)

        return torch.tensor(ascii_ids, dtype=torch.long), bits

    # ==================== WORD BOUNDARY DETECTION (for Zone C) ====================

    def find_word_boundaries(self, ids: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Find word boundaries in a sequence of character IDs.

        Used by Zone C (Word Building) to segment character sequences into words.

        Args:
            ids: Tensor of character IDs

        Returns:
            List of (start_idx, end_idx) tuples for each word
        """
        if ids.numel() == 0:
            return []

        ids_list = ids.tolist()
        boundaries = []
        word_start = None

        for i, char_id in enumerate(ids_list):
            char = self.id_to_char.get(int(char_id), ' ')

            if char in self.word_boundaries:
                if word_start is not None:
                    boundaries.append((word_start, i))
                    word_start = None
            else:
                if word_start is None:
                    word_start = i

        # Handle last word
        if word_start is not None:
            boundaries.append((word_start, len(ids_list)))

        return boundaries

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        return f"BinaryTokenizer(vocab_size={self.vocab_size})"