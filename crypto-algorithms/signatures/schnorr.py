"""
Schnorr Signature Implementation (BIP340)

This module provides an educational implementation of the Schnorr signature scheme
as specified in BIP340 for Bitcoin. It includes both a pure Python implementation
and a wrapper around the libsecp256k1 library when available.
"""

from typing import Tuple, Optional, Union
import hashlib
import random
from dataclasses import dataclass
import hmac

# secp256k1 curve parameters (same as ECDSA)
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

def tagged_hash(tag: str, msg: bytes) -> bytes:
    """
    Create a tagged hash as specified in BIP340.
    """
    tag_hash = hashlib.sha256(tag.encode()).digest()
    return hashlib.sha256(tag_hash + tag_hash + msg).digest()

class SchnorrSignature:
    def __init__(self, use_builtin: bool = True):
        """
        Initialize Schnorr signature implementation.
        
        Args:
            use_builtin: If True, attempts to use libsecp256k1.
                        If False, uses pure Python implementation.
        """
        self.use_builtin = use_builtin
        if use_builtin:
            try:
                import secp256k1
                self.secp256k1 = secp256k1
            except ImportError:
                print("Warning: libsecp256k1 not available, falling back to pure Python")
                self.use_builtin = False

    def _point_add(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[int, int]:
        """Add two points on the curve."""
        if p1 is None:
            return p2
        if p2 is None:
            return p1
        
        x1, y1 = p1
        x2, y2 = p2
        
        if x1 == x2:
            if (y1 + y2) % P == 0:
                return None
            # Point doubling
            lam = (3 * x1 * x1) * pow(2 * y1, -1, P) % P
        else:
            # Point addition
            lam = (y2 - y1) * pow(x2 - x1, -1, P) % P
        
        x3 = (lam * lam - x1 - x2) % P
        y3 = (lam * (x1 - x3) - y1) % P
        
        return (x3, y3)

    def _scalar_mult(self, k: int, p: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """Multiply point by scalar using double-and-add."""
        if p is None:
            p = (Gx, Gy)
            
        result = None
        addend = p
        
        while k:
            if k & 1:
                result = self._point_add(result, addend)
            addend = self._point_add(addend, addend)
            k >>= 1
        
        return result

    def _lift_x(self, x: int) -> Optional[Tuple[int, int]]:
        """
        Convert an x coordinate to a point on the curve.
        Returns None if x is not on the curve.
        """
        if not (0 <= x < P):
            return None
            
        # Compute y² = x³ + 7
        y_sq = (pow(x, 3, P) + 7) % P
        
        # Try to compute y
        y = pow(y_sq, (P + 1) // 4, P)
        
        if pow(y, 2, P) != y_sq:
            return None
            
        # Return the point with even y
        return (x, y if y % 2 == 0 else P - y)

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate a new Schnorr keypair.
        
        Returns:
            Tuple of (private_key, public_key) in bytes
        """
        if self.use_builtin:
            privkey = self.secp256k1.PrivateKey()
            pubkey = privkey.pubkey
            return (
                privkey.private_key,
                pubkey.serialize()[1:]  # Remove prefix byte
            )
        
        while True:
            # Generate random private key
            d = random.randrange(1, N)
            
            # Calculate public key point
            P = self._scalar_mult(d)
            
            # Check if y coordinate is even (BIP340 requirement)
            if P[1] % 2 == 0:
                return (
                    d.to_bytes(32, 'big'),
                    P[0].to_bytes(32, 'big')  # x-only pubkey
                )
            # If y is odd, negate the private key
            d = N - d

    def sign(self, message: Union[str, bytes], private_key: bytes,
            aux_rand: Optional[bytes] = None) -> bytes:
        """
        Create a Schnorr signature according to BIP340.
        
        Args:
            message: Message to sign
            private_key: 32-byte private key
            aux_rand: Optional 32-byte auxiliary random data
            
        Returns:
            64-byte signature (R.x || s)
        """
        if isinstance(message, str):
            message = message.encode()
            
        if self.use_builtin:
            privkey = self.secp256k1.PrivateKey(private_key)
            sig = privkey.schnorr_sign(message, aux_rand)
            return sig
            
        # Convert inputs to integers
        d = int.from_bytes(private_key, 'big')
        
        # Generate auxiliary random data if not provided
        if aux_rand is None:
            aux_rand = random.randbytes(32)
            
        # Calculate public key point
        P = self._scalar_mult(d)
        
        # BIP340 requires even y coordinate
        if P[1] % 2 == 1:
            d = N - d
            P = self._scalar_mult(d)
            
        # Calculate t = d xor hash(P.x || aux_rand)
        P_bytes = P[0].to_bytes(32, 'big')
        t = d ^ int.from_bytes(tagged_hash("BIPSchnorrDerive", P_bytes + aux_rand), 'big')
        
        # Calculate R = t·G
        R = self._scalar_mult(t)
        
        # Calculate e = hash(R.x || P.x || message)
        R_bytes = R[0].to_bytes(32, 'big')
        e = int.from_bytes(tagged_hash("BIPSchnorrSign", R_bytes + P_bytes + message), 'big')
        
        # Calculate s = (t + e·d) mod n
        s = (t + e * d) % N
        
        # Return 64-byte signature
        return R_bytes + s.to_bytes(32, 'big')

    def verify(self, message: Union[str, bytes], signature: bytes,
              public_key: bytes) -> bool:
        """
        Verify a Schnorr signature according to BIP340.
        
        Args:
            message: Original message
            signature: 64-byte signature (R.x || s)
            public_key: 32-byte public key (x-only)
            
        Returns:
            True if signature is valid
        """
        if isinstance(message, str):
            message = message.encode()
            
        if len(signature) != 64 or len(public_key) != 32:
            return False
            
        if self.use_builtin:
            try:
                pubkey = self.secp256k1.PublicKey(b'\x02' + public_key, True)
                return pubkey.schnorr_verify(message, signature)
            except Exception:
                return False
        
        # Extract R.x and s from signature
        Rx = int.from_bytes(signature[:32], 'big')
        s = int.from_bytes(signature[32:], 'big')
        
        # Extract P.x from public key
        Px = int.from_bytes(public_key, 'big')
        
        # Verify s is in [0, n-1]
        if s >= N:
            return False
            
        # Lift x coordinates to curve points
        R = self._lift_x(Rx)
        P = self._lift_x(Px)
        
        if R is None or P is None:
            return False
            
        # Calculate e = hash(R.x || P.x || message)
        e = int.from_bytes(
            tagged_hash("BIPSchnorrSign",
                       Rx.to_bytes(32, 'big') +
                       Px.to_bytes(32, 'big') +
                       message),
            'big'
        )
        
        # Verify R = s·G - e·P
        R_calc = self._point_add(
            self._scalar_mult(s),
            self._scalar_mult(N - e, P)
        )
        
        if R_calc is None:
            return False
            
        # Compare x coordinates
        return R_calc[0] == Rx

    def demonstrate_batch_verification(self, messages: list, signatures: list,
                                    public_keys: list) -> Tuple[bool, float]:
        """
        Demonstrate batch verification of multiple signatures.
        
        Args:
            messages: List of messages
            signatures: List of signatures
            public_keys: List of public keys
            
        Returns:
            Tuple of (all_valid, speedup_factor)
        """
        import time
        
        if not (len(messages) == len(signatures) == len(public_keys)):
            raise ValueError("Input lists must have same length")
            
        # Time individual verification
        start = time.time()
        individual_valid = all(
            self.verify(m, s, p)
            for m, s, p in zip(messages, signatures, public_keys)
        )
        individual_time = time.time() - start
        
        if not self.use_builtin:
            # Basic batch verification (educational - not constant time!)
            start = time.time()
            
            try:
                # Random scalars for linear combination
                scalars = [random.randrange(1, N) for _ in messages]
                
                # Calculate sum(a_i * s_i) * G
                s_sum = sum(
                    (a * int.from_bytes(s[32:], 'big')) % N
                    for a, s in zip(scalars, signatures)
                ) % N
                R_sum = self._scalar_mult(s_sum)
                
                # Calculate sum(a_i * e_i * P_i) + sum(a_i * R_i)
                points_sum = None
                for a, m, s, p in zip(scalars, messages, signatures, public_keys):
                    # Get points
                    R = self._lift_x(int.from_bytes(s[:32], 'big'))
                    P = self._lift_x(int.from_bytes(p, 'big'))
                    if R is None or P is None:
                        return False, 0
                        
                    # Calculate e
                    e = int.from_bytes(
                        tagged_hash("BIPSchnorrSign",
                                  s[:32] + p + m),
                        'big'
                    )
                    
                    # Add (a*e*P + a*R) to sum
                    points_sum = self._point_add(
                        points_sum,
                        self._point_add(
                            self._scalar_mult((a * e) % N, P),
                            self._scalar_mult(a, R)
                        )
                    )
                
                batch_valid = (R_sum == points_sum)
                batch_time = time.time() - start
                
                return (
                    individual_valid and batch_valid,
                    individual_time / batch_time if batch_time > 0 else 1
                )
                
            except Exception:
                return False, 0
                
        else:
            # Use libsecp256k1's batch verification if available
            try:
                start = time.time()
                batch_valid = self.secp256k1.schnorr_batch_verify(
                    messages, signatures, public_keys
                )
                batch_time = time.time() - start
                
                return (
                    individual_valid and batch_valid,
                    individual_time / batch_time if batch_time > 0 else 1
                )
            except Exception:
                return False, 0

# Example usage and tests
if __name__ == "__main__":
    # Test both implementations
    for impl in [True, False]:
        schnorr = SchnorrSignature(use_builtin=impl)
        
        # Generate keypair
        private_key, public_key = schnorr.generate_keypair()
        
        # Test message
        message = "Hello, Schnorr!"
        
        # Sign message
        signature = schnorr.sign(message, private_key)
        
        # Verify signature
        valid = schnorr.verify(message, signature, public_key)
        
        print(f"\nTesting {'built-in' if impl else 'pure Python'} implementation:")
        print(f"Private key: {private_key.hex()}")
        print(f"Public key (x-only): {public_key.hex()}")
        print(f"Signature: {signature.hex()}")
        print(f"Signature valid: {valid}")
        
        # Demonstrate batch verification
        if not impl:
            # Generate multiple signatures
            n_sigs = 5
            messages = [f"Message {i}" for i in range(n_sigs)]
            keypairs = [schnorr.generate_keypair() for _ in range(n_sigs)]
            signatures = [
                schnorr.sign(m, priv)
                for m, (priv, _) in zip(messages, keypairs)
            ]
            public_keys = [pub for _, pub in keypairs]
            
            valid, speedup = schnorr.demonstrate_batch_verification(
                messages, signatures, public_keys
            )
            print(f"\nBatch verification demo:")
            print(f"All signatures valid: {valid}")
            print(f"Speedup factor: {speedup:.2f}x")