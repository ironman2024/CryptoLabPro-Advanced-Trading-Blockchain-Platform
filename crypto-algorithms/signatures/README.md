# Digital Signature Algorithms

This directory contains implementations of digital signature algorithms used in blockchain technology.

## Implementations

### ECDSA (Elliptic Curve Digital Signature Algorithm)
The ECDSA implementation provides:
- Pure Python implementation for educational purposes
- Wrapper around cryptography library for production use
- secp256k1 curve support (used in Bitcoin)

Features:
- Key pair generation
- Signature creation and verification
- Signature malleability demonstration
- Comprehensive test vectors

### Schnorr Signatures (BIP340)
The Schnorr signature implementation includes:
- Pure Python implementation following BIP340 spec
- Wrapper around libsecp256k1 for production use
- Batch verification capabilities

Features:
- Key pair generation with x-only public keys
- Tagged hashing for domain separation
- Batch signature verification
- Test vectors from BIP340

## Educational Components

### ECDSA Signature Malleability
```python
from signatures.ecdsa import ECDSA

signer = ECDSA()
private_key, public_key = signer.generate_keypair()

# Demonstrate signature malleability
result = signer.demonstrate_signature_malleability("Hello", private_key)
print(f"Original s: {result['original_signature']['s']}")
print(f"Modified s: {result['modified_signature']['s']}")
print(f"Both valid: {result['original_signature']['valid']} and {result['modified_signature']['valid']}")
```

### Schnorr Batch Verification
```python
from signatures.schnorr import SchnorrSignature

schnorr = SchnorrSignature()

# Generate multiple signatures
messages = ["Message 1", "Message 2", "Message 3"]
keypairs = [schnorr.generate_keypair() for _ in range(len(messages))]
signatures = [schnorr.sign(m, priv) for m, (priv, _) in zip(messages, keypairs)]
public_keys = [pub for _, pub in keypairs]

# Verify in batch
valid, speedup = schnorr.demonstrate_batch_verification(messages, signatures, public_keys)
print(f"Batch verification speedup: {speedup:.2f}x")
```

## Usage in Blockchain

1. **ECDSA in Bitcoin**:
   - Transaction signing
   - Address generation
   - Message signing

2. **Schnorr in Bitcoin (since Taproot)**:
   - More efficient signatures
   - Native multisig support
   - Batch verification

## Security Considerations

For production use:
- Always use the built-in implementations (`use_builtin=True`)
- The pure Python implementations are for learning only
- Use secure random number generation
- Protect private keys
- Be aware of signature malleability

## Performance Comparison

Typical performance metrics on modern hardware:

ECDSA:
- Built-in: ~1000 verifications/s
- Pure Python: ~10 verifications/s

Schnorr:
- Built-in: ~2000 verifications/s
- Pure Python: ~15 verifications/s
- Batch verification: 2-5x speedup for multiple signatures

## References

- [Bitcoin BIP340 (Schnorr)](https://github.com/bitcoin/bips/blob/master/bip-0340.mediawiki)
- [SEC 2: Recommended Elliptic Curve Domain Parameters](https://www.secg.org/sec2-v2.pdf)
- [Bitcoin Wiki: Secp256k1](https://en.bitcoin.it/wiki/Secp256k1)
- [Schnorr Signatures for Bitcoin](https://github.com/sipa/bips/blob/bip-taproot/bip-0340.mediawiki)