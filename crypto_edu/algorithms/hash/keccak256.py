"""
Keccak-256 Implementation with educational components and visualization capabilities.

This module provides both a pure Python implementation of Keccak-256 (used in Ethereum)
for educational purposes and a wrapper around hashlib's sha3_256 for production use.
"""

import hashlib
from typing import List, Tuple, Union
import numpy as np

# Keccak-256 Constants
KECCAK_ROUNDS = 24
KECCAK_LANE_WIDTH = 64  # bits
KECCAK_STATE_SIZE = 5 * 5 * KECCAK_LANE_WIDTH  # 1600 bits

# Round constants
RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008
]

# Rotation offsets
R = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14]
]

class Keccak256:
    def __init__(self, use_builtin: bool = True):
        """
        Initialize Keccak-256 hasher.
        
        Args:
            use_builtin: If True, uses Python's builtin hashlib implementation.
                        If False, uses the educational pure Python implementation.
        """
        self.use_builtin = use_builtin
        self.rate = 1088  # bits (1600 - 512)
        self.capacity = 512  # bits
        self.output_length = 256  # bits
        
        # State is a 5x5 array of 64-bit lanes
        self.state = [[0 for _ in range(5)] for _ in range(5)]
        
    def _reset_state(self):
        """Reset the state to all zeros."""
        self.state = [[0 for _ in range(5)] for _ in range(5)]
    
    def _keccak_f(self):
        """
        Apply the Keccak-f[1600] permutation to the state.
        This is the core function of Keccak.
        """
        lanes = self.state
        
        # 24 rounds
        for round_idx in range(KECCAK_ROUNDS):
            # Theta step
            C = [lanes[x][0] ^ lanes[x][1] ^ lanes[x][2] ^ lanes[x][3] ^ lanes[x][4] for x in range(5)]
            D = [C[(x-1) % 5] ^ self._rotate_left(C[(x+1) % 5], 1) for x in range(5)]
            lanes = [[lanes[x][y] ^ D[x] for y in range(5)] for x in range(5)]
            
            # Rho and Pi steps
            x, y = 1, 0
            current = lanes[x][y]
            for t in range(24):
                x, y = y, (2*x + 3*y) % 5
                current, lanes[x][y] = lanes[x][y], self._rotate_left(current, R[x][y])
            
            # Chi step
            lanes = [[lanes[x][y] ^ ((~lanes[(x+1) % 5][y]) & lanes[(x+2) % 5][y]) for y in range(5)] for x in range(5)]
            
            # Iota step
            lanes[0][0] ^= RC[round_idx]
        
        self.state = lanes
    
    def _rotate_left(self, x: int, n: int) -> int:
        """
        Rotate a 64-bit integer left by n bits.
        
        Args:
            x: 64-bit integer
            n: Number of bits to rotate
            
        Returns:
            Rotated integer
        """
        n = n % 64
        return ((x << n) | (x >> (64 - n))) & ((1 << 64) - 1)
    
    def _pad(self, message: bytes) -> List[int]:
        """
        Pad the message according to Keccak padding rule.
        
        Args:
            message: Input message bytes
            
        Returns:
            List of padded blocks (each block is rate bits)
        """
        # Convert to bits
        bits = ''.join(format(b, '08b') for b in message)
        
        # Pad with 10*1 pattern
        bits += '1'
        while (len(bits) + 1) % self.rate != 0:
            bits += '0'
        bits += '1'
        
        # Convert back to bytes and split into blocks
        padded_bytes = []
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                padded_bytes.append(int(bits[i:i+8], 2))
            else:
                padded_bytes.append(int(bits[i:] + '0' * (8 - len(bits[i:])), 2))
        
        # Split into blocks of rate bits (rate/8 bytes)
        blocks = []
        rate_bytes = self.rate // 8
        for i in range(0, len(padded_bytes), rate_bytes):
            blocks.append(padded_bytes[i:i+rate_bytes])
        
        return blocks
    
    def _absorb(self, block: List[int]):
        """
        Absorb a block into the state.
        
        Args:
            block: Block of rate bits
        """
        # Convert block to state format
        rate_bytes = self.rate // 8
        for i in range(min(len(block), rate_bytes)):
            byte_idx = i
            x = byte_idx % 5
            y = byte_idx // 5
            lane_idx = byte_idx // 8
            bit_idx = (byte_idx % 8) * 8
            
            # XOR the byte into the state
            self.state[x][y] ^= block[i] << bit_idx
        
        # Apply the permutation
        self._keccak_f()
    
    def _squeeze(self) -> bytes:
        """
        Squeeze the output from the state.
        
        Returns:
            Output hash bytes
        """
        output_bytes = bytearray()
        output_bits = self.output_length
        rate_bytes = self.rate // 8
        
        while output_bits > 0:
            # Extract bytes from the state
            for i in range(min(rate_bytes, output_bits // 8)):
                byte_idx = i
                x = byte_idx % 5
                y = byte_idx // 5
                lane_idx = byte_idx // 8
                bit_idx = (byte_idx % 8) * 8
                
                # Extract byte from state
                output_bytes.append((self.state[x][y] >> bit_idx) & 0xFF)
                output_bits -= 8
            
            if output_bits > 0:
                self._keccak_f()
        
        return bytes(output_bytes)
    
    def hash(self, message: Union[str, bytes]) -> str:
        """
        Compute Keccak-256 hash of a message.
        
        Args:
            message: Input message as string or bytes
            
        Returns:
            Hexadecimal string of the hash
        """
        if isinstance(message, str):
            message = message.encode()
            
        if self.use_builtin:
            # Note: hashlib.sha3_256 is FIPS 202 SHA3-256, not Keccak-256 used in Ethereum
            # For Ethereum's Keccak-256, we'd need a specialized library
            # This is just for demonstration
            return hashlib.sha3_256(message).hexdigest()
        
        # Reset state
        self._reset_state()
        
        # Pad message
        blocks = self._pad(message)
        
        # Absorb phase
        for block in blocks:
            self._absorb(block)
        
        # Squeeze phase
        output = self._squeeze()
        
        return output.hex()
    
    def demonstrate_avalanche(self, message: str) -> Tuple[str, str, float]:
        """
        Demonstrate the avalanche effect by changing one bit and comparing hashes.
        
        Args:
            message: Input message
            
        Returns:
            Tuple of (original hash, modified hash, bit difference percentage)
        """
        # Get original hash
        orig_hash = self.hash(message)
        
        # Modify one bit in the message
        message_bytes = bytearray(message.encode())
        message_bytes[0] ^= 1  # Flip least significant bit of first byte
        mod_message = bytes(message_bytes)
        mod_hash = self.hash(mod_message)
        
        # Compare bits
        orig_bits = bin(int(orig_hash, 16))[2:].zfill(256)
        mod_bits = bin(int(mod_hash, 16))[2:].zfill(256)
        diff_bits = sum(a != b for a, b in zip(orig_bits, mod_bits))
        diff_percentage = (diff_bits / 256) * 100
        
        return orig_hash, mod_hash, diff_percentage
    
    def get_state_visualization(self) -> List[List[int]]:
        """
        Get a visualization of the current state.
        
        Returns:
            5x5 grid of state values
        """
        return self.state

# Example usage and tests
if __name__ == "__main__":
    # Test vector
    test_vector = "abc"
    
    # Test both implementations
    keccak_builtin = Keccak256(use_builtin=True)
    keccak_pure = Keccak256(use_builtin=False)
    
    builtin_hash = keccak_builtin.hash(test_vector)
    pure_hash = keccak_pure.hash(test_vector)
    
    print(f"Builtin implementation: {builtin_hash}")
    print(f"Pure Python implementation: {pure_hash}")
    
    # Note: The hashes may not match exactly because hashlib.sha3_256 is FIPS 202 SHA3-256,
    # not Keccak-256 used in Ethereum. For exact Ethereum Keccak-256, use a specialized library.
    
    # Demonstrate avalanche effect
    orig_hash, mod_hash, diff_percent = keccak_pure.demonstrate_avalanche("Hello, World!")
    print(f"\nOriginal message hash: {orig_hash}")
    print(f"Modified message hash: {mod_hash}")
    print(f"Bit difference: {diff_percent:.2f}%")
    
    # Visualize state (just for demonstration)
    print("\nState visualization (after hashing):")
    state = keccak_pure.get_state_visualization()
    for row in state:
        print(" ".join(f"{lane:016x}" for lane in row))