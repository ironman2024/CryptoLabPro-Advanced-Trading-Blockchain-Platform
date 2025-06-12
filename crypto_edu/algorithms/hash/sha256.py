"""
SHA-256 Implementation with educational components and visualization capabilities.

This module provides both a pure Python implementation of SHA-256 for educational purposes
and a wrapper around hashlib's SHA-256 for production use.
"""

import hashlib
import struct
import binascii
from typing import List, Tuple, Union
import numpy as np

# SHA-256 Constants
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

def rotr(x: int, n: int, size: int = 32) -> int:
    """Rotate right: (x >>> n)"""
    return ((x >> n) | (x << (size - n))) & ((1 << size) - 1)

def ch(x: int, y: int, z: int) -> int:
    """Ch mixing function"""
    return (x & y) ^ (~x & z)

def maj(x: int, y: int, z: int) -> int:
    """Maj mixing function"""
    return (x & y) ^ (x & z) ^ (y & z)

def sigma0(x: int) -> int:
    """Σ0 function"""
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)

def sigma1(x: int) -> int:
    """Σ1 function"""
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)

def gamma0(x: int) -> int:
    """γ0 function"""
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)

def gamma1(x: int) -> int:
    """γ1 function"""
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)

class SHA256:
    def __init__(self, use_builtin: bool = True):
        """
        Initialize SHA-256 hasher.
        
        Args:
            use_builtin: If True, uses Python's builtin hashlib implementation.
                        If False, uses the educational pure Python implementation.
        """
        self.use_builtin = use_builtin
        if not use_builtin:
            self.h = [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
            ]
    
    def _pad_message(self, message: bytes) -> bytes:
        """
        Pad the message according to SHA-256 specification.
        """
        length = len(message) * 8
        message += b'\x80'
        while (len(message) + 8) % 64 != 0:
            message += b'\x00'
        message += length.to_bytes(8, 'big')
        return message

    def _process_block(self, block: bytes) -> None:
        """
        Process a 512-bit block according to SHA-256 specification.
        """
        w = [0] * 64
        # Break block into 16 32-bit big-endian words
        for i in range(16):
            w[i] = int.from_bytes(block[i*4:(i+1)*4], 'big')
        
        # Extend first 16 words into remaining 48 words
        for i in range(16, 64):
            w[i] = (gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16]) & 0xffffffff
        
        # Initialize hash value for this block
        a, b, c, d, e, f, g, h = self.h
        
        # Main loop
        for i in range(64):
            t1 = h + sigma1(e) + ch(e, f, g) + K[i] + w[i]
            t2 = sigma0(a) + maj(a, b, c)
            h = g
            g = f
            f = e
            e = (d + t1) & 0xffffffff
            d = c
            c = b
            b = a
            a = (t1 + t2) & 0xffffffff
        
        # Update hash values
        self.h[0] = (self.h[0] + a) & 0xffffffff
        self.h[1] = (self.h[1] + b) & 0xffffffff
        self.h[2] = (self.h[2] + c) & 0xffffffff
        self.h[3] = (self.h[3] + d) & 0xffffffff
        self.h[4] = (self.h[4] + e) & 0xffffffff
        self.h[5] = (self.h[5] + f) & 0xffffffff
        self.h[6] = (self.h[6] + g) & 0xffffffff
        self.h[7] = (self.h[7] + h) & 0xffffffff

    def hash(self, message: Union[str, bytes]) -> str:
        """
        Compute SHA-256 hash of a message.
        
        Args:
            message: Input message as string or bytes
            
        Returns:
            Hexadecimal string of the hash
        """
        if isinstance(message, str):
            message = message.encode()
            
        if self.use_builtin:
            return hashlib.sha256(message).hexdigest()
        
        # Pad message
        padded = self._pad_message(message)
        
        # Process message in 512-bit blocks
        for i in range(0, len(padded), 64):
            self._process_block(padded[i:i+64])
            
        # Produce final hash value
        return ''.join(f'{h:08x}' for h in self.h)

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

    def get_intermediate_states(self, message: str) -> List[List[int]]:
        """
        Get intermediate states during hash computation for educational purposes.
        
        Args:
            message: Input message
            
        Returns:
            List of intermediate hash states after each block
        """
        if isinstance(message, str):
            message = message.encode()
            
        if self.use_builtin:
            # Can't get intermediate states with builtin
            return []
            
        # Reset hash state
        self.h = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        ]
        
        # Pad message
        padded = self._pad_message(message)
        
        # Process message in 512-bit blocks and record states
        states = []
        for i in range(0, len(padded), 64):
            self._process_block(padded[i:i+64])
            states.append(self.h.copy())
            
        return states

def find_collision_example(prefix_bits: int = 8) -> Tuple[str, str, str]:
    """
    Find a small collision example for educational purposes.
    This is NOT cryptographically secure - it's just for demonstration!
    
    Args:
        prefix_bits: Number of bits to match for the collision
        
    Returns:
        Tuple of (message1, message2, common_prefix)
    """
    hasher = SHA256()
    seen = {}
    counter = 0
    mask = (1 << prefix_bits) - 1
    
    while True:
        message = f"test{counter}".encode()
        full_hash = hasher.hash(message)
        prefix = int(full_hash[:prefix_bits//4], 16)
        
        if prefix in seen and seen[prefix] != message:
            return (
                seen[prefix].decode(),
                message.decode(),
                full_hash[:prefix_bits//4]
            )
        
        seen[prefix] = message
        counter += 1
        if counter > 10000:  # Prevent infinite loops
            return None

# Example usage and tests
if __name__ == "__main__":
    # Test vector from NIST
    test_vector = "abc"
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    
    # Test both implementations
    sha256_builtin = SHA256(use_builtin=True)
    sha256_pure = SHA256(use_builtin=False)
    
    assert sha256_builtin.hash(test_vector) == expected
    assert sha256_pure.hash(test_vector) == expected
    
    # Demonstrate avalanche effect
    orig_hash, mod_hash, diff_percent = sha256_pure.demonstrate_avalanche("Hello, World!")
    print(f"Original message hash: {orig_hash}")
    print(f"Modified message hash: {mod_hash}")
    print(f"Bit difference: {diff_percent:.2f}%")
    
    # Show intermediate states
    states = sha256_pure.get_intermediate_states("Hello, World!")
    print("\nIntermediate states:")
    for i, state in enumerate(states):
        print(f"Block {i+1}: {' '.join(f'{h:08x}' for h in state)}")
    
    # Find small collision example
    collision = find_collision_example(8)
    if collision:
        msg1, msg2, prefix = collision
        print(f"\nFound messages with {len(prefix)*4}-bit hash collision:")
        print(f"Message 1: {msg1}")
        print(f"Message 2: {msg2}")
        print(f"Common hash prefix: {prefix}")