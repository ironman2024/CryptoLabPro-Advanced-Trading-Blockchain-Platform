"""
ECDSA (Elliptic Curve Digital Signature Algorithm) Implementation

This module provides both an educational implementation of ECDSA
and a wrapper around the cryptography library for production use.
Focuses on the secp256k1 curve used in Bitcoin.
"""

from typing import Tuple, Optional, Union
import hashlib
import random
from dataclasses import dataclass

# secp256k1 curve parameters
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

@dataclass
class Point:
    """Represents a point on the secp256k1 curve."""
    x: int
    y: int
    infinity: bool = False

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Point':
        """Create a point from its compressed byte representation."""
        if len(data) != 33 and len(data) != 65:
            raise ValueError("Invalid point encoding length")
        
        if data[0] == 0x00:
            return cls(0, 0, infinity=True)
        
        if len(data) == 33:
            # Compressed format
            if data[0] not in (0x02, 0x03):
                raise ValueError("Invalid compressed point")
            x = int.from_bytes(data[1:], 'big')
            # Solve y^2 = x^3 + 7
            y_squared = (pow(x, 3, P) + 7) % P
            y = pow(y_squared, (P + 1) // 4, P)
            if bool(data[0] & 1) != bool(y & 1):
                y = P - y
            return cls(x, y)
        else:
            # Uncompressed format
            if data[0] != 0x04:
                raise ValueError("Invalid uncompressed point")
            x = int.from_bytes(data[1:33], 'big')
            y = int.from_bytes(data[33:], 'big')
            return cls(x, y)

    def to_bytes(self, compressed: bool = True) -> bytes:
        """Convert point to bytes (compressed or uncompressed format)."""
        if self.infinity:
            return b'\x00'
        
        if compressed:
            return bytes([0x02 + (self.y & 1)]) + self.x.to_bytes(32, 'big')
        else:
            return b'\x04' + self.x.to_bytes(32, 'big') + self.y.to_bytes(32, 'big')

class ECDSA:
    def __init__(self, use_builtin: bool = True):
        """
        Initialize ECDSA with secp256k1 curve.
        
        Args:
            use_builtin: If True, uses the cryptography library.
                        If False, uses the educational implementation.
        """
        self.use_builtin = use_builtin
        if use_builtin:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import ec
            self.curve = ec.SECP256K1()
            self.hash_algorithm = hashes.SHA256()
    
    def _point_add(self, p1: Point, p2: Point) -> Point:
        """Add two points on the curve."""
        if p1.infinity:
            return p2
        if p2.infinity:
            return p1
        
        if p1.x == p2.x:
            if (p1.y + p2.y) % P == 0:
                return Point(0, 0, infinity=True)
            # Point doubling
            lam = (3 * p1.x * p1.x) * pow(2 * p1.y, -1, P) % P
        else:
            # Point addition
            lam = (p2.y - p1.y) * pow(p2.x - p1.x, -1, P) % P
        
        x3 = (lam * lam - p1.x - p2.x) % P
        y3 = (lam * (p1.x - x3) - p1.y) % P
        
        return Point(x3, y3)

    def _scalar_mult(self, k: int, p: Point) -> Point:
        """Multiply point by scalar using double-and-add."""
        result = Point(0, 0, infinity=True)
        addend = p
        
        while k:
            if k & 1:
                result = self._point_add(result, addend)
            addend = self._point_add(addend, addend)
            k >>= 1
        
        return result

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate a new ECDSA keypair.
        
        Returns:
            Tuple of (private_key, public_key) in bytes
        """
        if self.use_builtin:
            from cryptography.hazmat.primitives.asymmetric import ec
            private_key = ec.generate_private_key(ec.SECP256K1())
            public_key = private_key.public_key()
            return (
                private_key.private_numbers().private_value.to_bytes(32, 'big'),
                public_key.public_bytes(
                    encoding=ec.Encoding.X962,
                    format=ec.PublicFormat.CompressedPoint
                )
            )
        
        # Generate private key
        private_key = random.randrange(1, N)
        
        # Calculate public key
        public_point = self._scalar_mult(private_key, Point(Gx, Gy))
        
        return (
            private_key.to_bytes(32, 'big'),
            public_point.to_bytes(compressed=True)
        )

    def sign(self, message: Union[str, bytes], private_key: bytes) -> Tuple[bytes, bytes]:
        """
        Sign a message using ECDSA.
        
        Args:
            message: Message to sign
            private_key: Private key bytes
            
        Returns:
            Tuple of (r, s) signature components in bytes
        """
        if isinstance(message, str):
            message = message.encode()
            
        if self.use_builtin:
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.hazmat.primitives import hashes
            
            private_key_obj = ec.derive_private_key(
                int.from_bytes(private_key, 'big'),
                ec.SECP256K1()
            )
            signature = private_key_obj.sign(
                message,
                ec.ECDSA(hashes.SHA256())
            )
            # Extract r and s from DER format
            from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
            r, s = decode_dss_signature(signature)
            return (r.to_bytes(32, 'big'), s.to_bytes(32, 'big'))
        
        # Calculate message hash
        z = int.from_bytes(hashlib.sha256(message).digest(), 'big')
        
        # Convert private key to int
        d = int.from_bytes(private_key, 'big')
        
        while True:
            # Generate random k
            k = random.randrange(1, N)
            
            # Calculate r = x coordinate of k*G
            r_point = self._scalar_mult(k, Point(Gx, Gy))
            r = r_point.x % N
            if r == 0:
                continue
            
            # Calculate s = k^(-1)(z + rd) mod N
            s = (pow(k, -1, N) * (z + r * d)) % N
            if s == 0:
                continue
            
            # Return signature
            return (r.to_bytes(32, 'big'), s.to_bytes(32, 'big'))

    def verify(self, message: Union[str, bytes], signature: Tuple[bytes, bytes], 
              public_key: bytes) -> bool:
        """
        Verify an ECDSA signature.
        
        Args:
            message: Original message
            signature: Tuple of (r, s) signature components
            public_key: Public key bytes
            
        Returns:
            True if signature is valid
        """
        if isinstance(message, str):
            message = message.encode()
            
        if self.use_builtin:
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
            from cryptography.exceptions import InvalidSignature
            
            try:
                # Convert public key bytes to PublicKey object
                public_key_obj = ec.EllipticCurvePublicKey.from_encoded_point(
                    ec.SECP256K1(),
                    public_key
                )
                
                # Encode signature in DER format
                r, s = (int.from_bytes(sig, 'big') for sig in signature)
                der_signature = encode_dss_signature(r, s)
                
                # Verify
                public_key_obj.verify(
                    der_signature,
                    message,
                    ec.ECDSA(hashes.SHA256())
                )
                return True
            except (ValueError, InvalidSignature):
                return False
        
        # Convert signature components to integers
        r = int.from_bytes(signature[0], 'big')
        s = int.from_bytes(signature[1], 'big')
        
        # Verify signature range
        if not (0 < r < N and 0 < s < N):
            return False
        
        # Calculate message hash
        z = int.from_bytes(hashlib.sha256(message).digest(), 'big')
        
        # Calculate u1 and u2
        w = pow(s, -1, N)
        u1 = (z * w) % N
        u2 = (r * w) % N
        
        # Calculate u1*G + u2*Q
        public_point = Point.from_bytes(public_key)
        point = self._point_add(
            self._scalar_mult(u1, Point(Gx, Gy)),
            self._scalar_mult(u2, public_point)
        )
        
        if point.infinity:
            return False
            
        # Verify that r equals x coordinate modulo N
        return point.x % N == r

    def demonstrate_signature_malleability(self, message: str, private_key: bytes) -> dict:
        """
        Demonstrate ECDSA signature malleability by creating equivalent signatures.
        
        Args:
            message: Message to sign
            private_key: Private key bytes
            
        Returns:
            Dictionary with original and modified signatures
        """
        # Get original signature
        r, s = self.sign(message, private_key)
        
        # Create equivalent signature using N - s
        s_int = int.from_bytes(s, 'big')
        s_alt = (N - s_int).to_bytes(32, 'big')
        
        # Verify both signatures
        public_key = self._scalar_mult(
            int.from_bytes(private_key, 'big'),
            Point(Gx, Gy)
        ).to_bytes()
        
        return {
            'message': message,
            'original_signature': {
                'r': r.hex(),
                's': s.hex(),
                'valid': self.verify(message, (r, s), public_key)
            },
            'modified_signature': {
                'r': r.hex(),
                's': s_alt.hex(),
                'valid': self.verify(message, (r, s_alt), public_key)
            }
        }

# Example usage and tests
if __name__ == "__main__":
    # Test both implementations
    for impl in [True, False]:
        ecdsa = ECDSA(use_builtin=impl)
        
        # Generate keypair
        private_key, public_key = ecdsa.generate_keypair()
        
        # Test message
        message = "Hello, Blockchain!"
        
        # Sign message
        signature = ecdsa.sign(message, private_key)
        
        # Verify signature
        valid = ecdsa.verify(message, signature, public_key)
        
        print(f"\nTesting {'built-in' if impl else 'pure Python'} implementation:")
        print(f"Private key: {private_key.hex()}")
        print(f"Public key: {public_key.hex()}")
        print(f"Signature (r,s): ({signature[0].hex()}, {signature[1].hex()})")
        print(f"Signature valid: {valid}")
        
        # Demonstrate signature malleability
        if not impl:
            result = ecdsa.demonstrate_signature_malleability(message, private_key)
            print("\nSignature Malleability Demonstration:")
            print(f"Original s: {result['original_signature']['s']}")
            print(f"Modified s: {result['modified_signature']['s']}")
            print(f"Both valid: {result['original_signature']['valid']} and {result['modified_signature']['valid']}")