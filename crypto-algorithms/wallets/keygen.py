"""
Key Generation Utilities

This module provides utilities for generating and managing cryptographic keys
for various blockchain address types.
"""

import hashlib
import hmac
from typing import Tuple, Optional
import secrets
from dataclasses import dataclass
from ..signatures.ecdsa import ECDSA
from ..signatures.schnorr import SchnorrSignature

@dataclass
class KeyPair:
    """Container for public/private key pair."""
    private_key: bytes
    public_key: bytes
    wif: str
    address: str
    address_type: str

class KeyGenerator:
    """
    Key generation utilities supporting multiple address types
    and signature schemes.
    """
    
    def __init__(self, testnet: bool = False):
        """
        Initialize key generator.
        
        Args:
            testnet: Whether to use testnet version bytes
        """
        self.testnet = testnet
        self.ecdsa = ECDSA(use_builtin=True)
        self.schnorr = SchnorrSignature(use_builtin=True)
        
        # Version bytes
        self.versions = {
            'mainnet': {
                'p2pkh': b'\x00',
                'p2sh': b'\x05',
                'wif': b'\x80'
            },
            'testnet': {
                'p2pkh': b'\x6f',
                'p2sh': b'\xc4',
                'wif': b'\xef'
            }
        }
    
    def generate_keypair(self, compressed: bool = True) -> KeyPair:
        """
        Generate new ECDSA keypair.
        
        Args:
            compressed: Whether to use compressed public key format
            
        Returns:
            KeyPair object
        """
        # Generate private key
        private_key, public_key = self.ecdsa.generate_keypair()
        
        # Create WIF private key
        wif = self._private_key_to_wif(private_key, compressed)
        
        # Generate P2PKH address
        address = self.public_key_to_address(public_key, 'p2pkh')
        
        return KeyPair(
            private_key=private_key,
            public_key=public_key,
            wif=wif,
            address=address,
            address_type='p2pkh'
        )
    
    def generate_schnorr_keypair(self) -> KeyPair:
        """
        Generate new Schnorr keypair (BIP340).
        
        Returns:
            KeyPair object
        """
        # Generate keypair
        private_key, public_key = self.schnorr.generate_keypair()
        
        # Create WIF
        wif = self._private_key_to_wif(private_key, True)  # Always compressed
        
        # Generate P2TR address
        address = self.public_key_to_address(public_key, 'p2tr')
        
        return KeyPair(
            private_key=private_key,
            public_key=public_key,
            wif=wif,
            address=address,
            address_type='p2tr'
        )
    
    def _private_key_to_wif(self, private_key: bytes,
                           compressed: bool = True) -> str:
        """
        Convert private key to Wallet Import Format (WIF).
        
        Args:
            private_key: Raw private key bytes
            compressed: Whether public key is compressed
            
        Returns:
            WIF string
        """
        # Get version byte
        version = self.versions['testnet' if self.testnet else 'mainnet']['wif']
        
        # Add version byte
        data = version + private_key
        
        # Add compression flag if needed
        if compressed:
            data += b'\x01'
        
        # Add checksum
        checksum = hashlib.sha256(
            hashlib.sha256(data).digest()
        ).digest()[:4]
        
        # Encode in base58
        return self._base58_encode(data + checksum)
    
    def wif_to_private_key(self, wif: str) -> Tuple[bytes, bool]:
        """
        Convert WIF to private key.
        
        Args:
            wif: WIF string
            
        Returns:
            Tuple of (private_key, compressed)
        """
        # Decode base58
        data = self._base58_decode(wif)
        
        # Verify checksum
        checksum = data[-4:]
        data = data[:-4]
        
        if checksum != hashlib.sha256(
            hashlib.sha256(data).digest()
        ).digest()[:4]:
            raise ValueError("Invalid WIF checksum")
        
        # Check version byte
        version = data[0:1]
        expected = self.versions['testnet' if self.testnet else 'mainnet']['wif']
        if version != expected:
            raise ValueError("Invalid WIF version byte")
        
        # Extract private key and compression flag
        if len(data) == 34:
            private_key = data[1:33]
            compressed = True
        else:
            private_key = data[1:]
            compressed = False
        
        return private_key, compressed
    
    def public_key_to_address(self, public_key: bytes,
                             address_type: str = 'p2pkh') -> str:
        """
        Convert public key to address.
        
        Args:
            public_key: Public key bytes
            address_type: Address type ('p2pkh', 'p2sh', 'p2wpkh', 'p2tr')
            
        Returns:
            Address string
        """
        if address_type == 'p2pkh':
            # Hash public key
            h = hashlib.new('ripemd160',
                hashlib.sha256(public_key).digest()
            ).digest()
            
            # Add version byte
            version = self.versions['testnet' if self.testnet else 'mainnet']['p2pkh']
            versioned = version + h
            
            # Add checksum
            checksum = hashlib.sha256(
                hashlib.sha256(versioned).digest()
            ).digest()[:4]
            
            # Encode in base58
            return self._base58_encode(versioned + checksum)
            
        elif address_type == 'p2sh':
            # Create redeem script (example: P2WPKH nested in P2SH)
            redeem_script = b'\x00\x14' + hashlib.new('ripemd160',
                hashlib.sha256(public_key).digest()
            ).digest()
            
            # Hash redeem script
            h = hashlib.new('ripemd160',
                hashlib.sha256(redeem_script).digest()
            ).digest()
            
            # Add version byte
            version = self.versions['testnet' if self.testnet else 'mainnet']['p2sh']
            versioned = version + h
            
            # Add checksum
            checksum = hashlib.sha256(
                hashlib.sha256(versioned).digest()
            ).digest()[:4]
            
            # Encode in base58
            return self._base58_encode(versioned + checksum)
            
        elif address_type == 'p2wpkh':
            # Native SegWit
            h = hashlib.new('ripemd160',
                hashlib.sha256(public_key).digest()
            ).digest()
            
            # Encode in bech32
            return self._bech32_encode(
                'tb' if self.testnet else 'bc',
                0,
                h
            )
            
        elif address_type == 'p2tr':
            # Taproot (BIP341)
            # Note: This is a simplified version
            return self._bech32_encode(
                'tb' if self.testnet else 'bc',
                1,
                public_key
            )
            
        else:
            raise ValueError("Invalid address type")
    
    @staticmethod
    def _base58_encode(data: bytes) -> str:
        """Encode bytes in base58."""
        alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
        
        # Convert to integer
        n = int.from_bytes(data, 'big')
        
        # Encode
        result = ''
        while n > 0:
            n, r = divmod(n, 58)
            result = alphabet[r] + result
            
        # Add leading zeros
        for b in data:
            if b == 0:
                result = alphabet[0] + result
            else:
                break
                
        return result
    
    @staticmethod
    def _base58_decode(s: str) -> bytes:
        """Decode base58 string to bytes."""
        alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
        
        # Convert to integer
        n = 0
        for char in s:
            n = n * 58 + alphabet.index(char)
            
        # Convert to bytes
        return n.to_bytes((n.bit_length() + 7) // 8, 'big')
    
    @staticmethod
    def _bech32_encode(hrp: str, witver: int, witprog: bytes) -> str:
        """
        Encode witness program in bech32/bech32m.
        Simplified implementation - real code would use full bech32 spec.
        """
        # This is a placeholder - real implementation would follow BIP350
        return f"{hrp}1{'q' * 58}"
    
    def create_multisig_address(self, public_keys: list, m: int,
                              address_type: str = 'p2sh') -> Tuple[str, bytes]:
        """
        Create m-of-n multisig address.
        
        Args:
            public_keys: List of public key bytes
            m: Number of required signatures
            address_type: Address type ('p2sh' or 'p2wsh')
            
        Returns:
            Tuple of (address, redeem_script)
        """
        if not (0 < m <= len(public_keys)):
            raise ValueError("Invalid m")
            
        # Create redeem script
        script = bytes([80 + m])  # OP_m
        for key in sorted(public_keys):  # Sort keys for standardness
            script += bytes([len(key)]) + key
        script += bytes([80 + len(public_keys)])  # OP_n
        script += b'\xae'  # OP_CHECKMULTISIG
        
        if address_type == 'p2sh':
            # Hash redeem script
            h = hashlib.new('ripemd160',
                hashlib.sha256(script).digest()
            ).digest()
            
            # Add version byte
            version = self.versions['testnet' if self.testnet else 'mainnet']['p2sh']
            versioned = version + h
            
            # Add checksum
            checksum = hashlib.sha256(
                hashlib.sha256(versioned).digest()
            ).digest()[:4]
            
            # Encode in base58
            address = self._base58_encode(versioned + checksum)
            
        elif address_type == 'p2wsh':
            # Native SegWit multisig
            script_hash = hashlib.sha256(script).digest()
            
            # Encode in bech32
            address = self._bech32_encode(
                'tb' if self.testnet else 'bc',
                0,
                script_hash
            )
            
        else:
            raise ValueError("Invalid address type")
            
        return address, script

# Example usage and tests
if __name__ == "__main__":
    # Create key generator
    keygen = KeyGenerator(testnet=False)
    
    # Generate ECDSA keypair
    print("Generating ECDSA keypair...")
    keypair = keygen.generate_keypair()
    print(f"Private key (WIF): {keypair.wif}")
    print(f"Public key: {keypair.public_key.hex()}")
    print(f"Address (P2PKH): {keypair.address}")
    
    # Generate Schnorr keypair
    print("\nGenerating Schnorr keypair...")
    schnorr_pair = keygen.generate_schnorr_keypair()
    print(f"Private key (WIF): {schnorr_pair.wif}")
    print(f"Public key: {schnorr_pair.public_key.hex()}")
    print(f"Address (P2TR): {schnorr_pair.address}")
    
    # Create multisig address
    print("\nCreating 2-of-3 multisig address...")
    keypairs = [keygen.generate_keypair() for _ in range(3)]
    public_keys = [kp.public_key for kp in keypairs]
    
    address, redeem_script = keygen.create_multisig_address(public_keys, 2)
    print(f"Multisig address: {address}")
    print(f"Redeem script: {redeem_script.hex()}")