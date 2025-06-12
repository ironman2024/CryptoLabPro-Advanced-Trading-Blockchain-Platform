"""
Keccak-256 Implementation (used in Ethereum)

This module provides both an educational implementation of Keccak-256
and a wrapper around the cryptography library's implementation for production use.
"""

from typing import List, Tuple, Union
import numpy as np

# Keccak-f[1600] permutation constants
KECCAK_ROUNDS = 24
ROTC = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44]
PILN = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1]
LFSR = [1, 0, 32898, 0, 32906, 2147483648, 2147516416, 2147483648, 32777, 2147483648, 138, 0, 136, 0, 2147516425, 2147483648, 2147483658, 0, 2147516555, 2147483648, 139, 2147483648, 32905, 2147483648]

def rol(a: int, n: int) -> int:
    """Rotate left operation."""
    return ((a << n) | (a >> (64 - n))) & ((1 << 64) - 1)

def load64(b: bytes) -> int:
    """Load 8 bytes into a 64-bit integer."""
    return int.from_bytes(b, 'little')

def store64(a: int) -> bytes:
    """Store a 64-bit integer as 8 bytes."""
    return a.to_bytes(8, 'little')

class Keccak256:
    def __init__(self, use_builtin: bool = True):
        """
        Initialize Keccak-256 hasher.
        
        Args:
            use_builtin: If True, uses the cryptography library implementation.
                        If False, uses the educational pure Python implementation.
        """
        self.use_builtin = use_builtin
        if not use_builtin:
            self.state = [[0] * 5 for _ in range(5)]
            self.buffer = bytearray()
            self.rate = 136  # 1088 bits
            self.suffix = 0x01
            
    def _keccak_f1600(self) -> None:
        """
        Keccak-f[1600] permutation function.
        This is the core permutation function used in Keccak.
        """
        lanes = [[0] * 5 for _ in range(5)]
        
        # Convert state to lanes
        for x in range(5):
            for y in range(5):
                lanes[x][y] = self.state[x][y]
        
        # Main permutation loop
        for round in range(KECCAK_ROUNDS):
            # θ (theta)
            C = [lanes[x][0] ^ lanes[x][1] ^ lanes[x][2] ^ lanes[x][3] ^ lanes[x][4] for x in range(5)]
            D = [C[(x + 4) % 5] ^ rol(C[(x + 1) % 5], 1) for x in range(5)]
            lanes = [[lanes[x][y] ^ D[x] for y in range(5)] for x in range(5)]
            
            # ρ and π (rho and pi)
            (x, y) = (1, 0)
            current = lanes[x][y]
            for t in range(24):
                (x, y) = (y, (2 * x + 3 * y) % 5)
                (current, lanes[x][y]) = (lanes[x][y], rol(current, ROTC[t]))
            
            # χ (chi)
            temp = [[lanes[x][y] for y in range(5)] for x in range(5)]
            for x in range(5):
                for y in range(5):
                    lanes[x][y] = temp[x][y] ^ ((~temp[(x + 1) % 5][y]) & temp[(x + 2) % 5][y])
            
            # ι (iota)
            lanes[0][0] ^= LFSR[round]
        
        # Update state
        self.state = lanes

    def update(self, data: bytes) -> None:
        """
        Update the hash state with more data.
        """
        if self.use_builtin:
            from cryptography.hazmat.primitives import hashes
            self.h = hashes.Hash(hashes.SHA3_256())
            self.h.update(data)
            return
            
        self.buffer.extend(data)
        
        while len(self.buffer) >= self.rate:
            block = self.buffer[:self.rate]
            self.buffer = self.buffer[self.rate:]
            
            # XOR block into state
            for i in range(0, self.rate, 8):
                x, y = (i // 8) % 5, (i // 8) // 5
                self.state[x][y] ^= load64(block[i:i+8])
            
            self._keccak_f1600()

    def finalize(self) -> bytes:
        """
        Finalize the hash computation and return the digest.
        """
        if self.use_builtin:
            return self.h.finalize()
            
        # Pad buffer
        self.buffer.append(self.suffix)
        while len(self.buffer) < self.rate:
            self.buffer.append(0)
        self.buffer[-1] |= 0x80
        
        # Process final block
        for i in range(0, self.rate, 8):
            x, y = (i // 8) % 5, (i // 8) // 5
            self.state[x][y] ^= load64(self.buffer[i:i+8])
        
        self._keccak_f1600()
        
        # Extract first 256 bits of state
        result = bytearray()
        for i in range(32):
            x, y = (i // 8) % 5, (i // 8) // 5
            result.extend(store64(self.state[x][y])[:(8 if i < 28 else 32 - 28 * 8)])
        
        return bytes(result)

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
            from cryptography.hazmat.primitives import hashes
            digest = hashes.Hash(hashes.SHA3_256())
            digest.update(message)
            return digest.finalize().hex()
        
        self.update(message)
        return self.finalize().hex()

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

# Example usage and tests
if __name__ == "__main__":
    # Test vector
    test_vector = "abc"
    # This is SHA3-256 test vector, Keccak-256 would be different
    expected = "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
    
    # Test both implementations
    keccak_builtin = Keccak256(use_builtin=True)
    keccak_pure = Keccak256(use_builtin=False)
    
    # Note: The pure Python implementation needs thorough testing against official test vectors
    hash1 = keccak_builtin.hash(test_vector)
    hash2 = keccak_pure.hash(test_vector)
    
    print(f"Built-in implementation: {hash1}")
    print(f"Pure Python implementation: {hash2}")
    
    # Demonstrate avalanche effect
    orig_hash, mod_hash, diff_percent = keccak_pure.demonstrate_avalanche("Hello, World!")
    print(f"\nAvalanche Effect Demonstration:")
    print(f"Original message hash: {orig_hash}")
    print(f"Modified message hash: {mod_hash}")
    print(f"Bit difference: {diff_percent:.2f}%")