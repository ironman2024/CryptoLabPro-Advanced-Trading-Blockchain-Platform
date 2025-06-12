# Cryptographic Hash Functions

This directory contains educational implementations of various cryptographic hash functions used in blockchain technology.

## Implementations

### SHA-256
The SHA-256 implementation provides both:
- A pure Python implementation for educational purposes
- A wrapper around Python's hashlib for production use

Features:
- Full SHA-256 algorithm implementation
- Avalanche effect demonstration
- Collision finding examples (for educational purposes)
- Comprehensive test vectors

### Keccak-256
The Keccak-256 implementation (used in Ethereum) includes:
- A pure Python implementation of the Keccak-f[1600] permutation
- A wrapper around the cryptography library for production use

Features:
- Complete Keccak state manipulation
- Padding and absorption phases
- Avalanche effect demonstration
- Test vectors

## Educational Components

### Avalanche Effect
Both implementations include methods to demonstrate the avalanche effect:
```python
from hashing.sha256 import SHA256

hasher = SHA256()
orig_hash, mod_hash, diff_percent = hasher.demonstrate_avalanche("Hello, World!")
print(f"Original hash: {orig_hash}")
print(f"Modified hash: {mod_hash}")
print(f"Bit difference: {diff_percent}%")
```

### Finding Collisions
The SHA-256 implementation includes a method to find partial hash collisions for educational purposes:
```python
from hashing.sha256 import find_collision_example

msg1, msg2, prefix = find_collision_example(8)  # Find 8-bit collision
print(f"Messages with same {len(prefix)*4}-bit prefix:")
print(f"Message 1: {msg1}")
print(f"Message 2: {msg2}")
print(f"Common prefix: {prefix}")
```

## Usage in Blockchain

These hash functions are fundamental to blockchain technology:

1. **Block Hashing**: SHA-256 is used in Bitcoin to:
   - Hash block headers
   - Calculate proof-of-work
   - Create transaction IDs

2. **Ethereum Usage**: Keccak-256 is used in Ethereum for:
   - Contract address generation
   - Transaction signing
   - State trie hashing

## Security Considerations

For production use:
- Always use the built-in implementations (`use_builtin=True`)
- The pure Python implementations are for learning only
- Never use the collision finding methods in production

## Performance Comparison

Typical performance metrics on modern hardware:
- Built-in SHA-256: ~150 MB/s
- Pure Python SHA-256: ~1 MB/s
- Built-in Keccak-256: ~100 MB/s
- Pure Python Keccak-256: ~0.5 MB/s

## References

- [NIST FIPS 180-4](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf)
- [Keccak Reference](https://keccak.team/files/Keccak-reference-3.0.pdf)
- [Bitcoin Wiki: SHA-256](https://en.bitcoin.it/wiki/SHA-256)
- [Ethereum Wiki: Ethash](https://eth.wiki/en/concepts/ethash/ethash)