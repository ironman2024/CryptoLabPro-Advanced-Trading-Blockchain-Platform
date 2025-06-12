"""
Hierarchical Deterministic (HD) Wallet Implementation

This module provides an implementation of BIP32, BIP39, and BIP44
for hierarchical deterministic wallet generation.
"""

import hashlib
import hmac
from typing import List, Tuple, Optional
import secrets
from dataclasses import dataclass
from ..signatures.ecdsa import ECDSA

# BIP39 wordlist (first few words for example)
BIP39_WORDLIST = [
    "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract",
    "absurd", "abuse", "access", "accident", "account", "accuse", "achieve", "acid",
    # ... (full list would be 2048 words)
]

@dataclass
class ExtendedKey:
    """Extended key structure for BIP32."""
    key: bytes
    chain_code: bytes
    depth: int
    index: int
    parent_fingerprint: bytes
    is_private: bool

class HDWallet:
    """
    Hierarchical Deterministic Wallet implementation following BIP32/39/44.
    """
    
    def __init__(self):
        """Initialize HD wallet."""
        self.ecdsa = ECDSA(use_builtin=True)
    
    def generate_mnemonic(self, strength: int = 128) -> str:
        """
        Generate BIP39 mnemonic phrase.
        
        Args:
            strength: Entropy length in bits (128-256)
            
        Returns:
            Mnemonic phrase as space-separated words
        """
        if strength not in [128, 160, 192, 224, 256]:
            raise ValueError("Invalid entropy length")
            
        # Generate random entropy
        entropy = secrets.token_bytes(strength // 8)
        
        # Add checksum
        entropy_bits = bin(int.from_bytes(entropy, 'big'))[2:].zfill(strength)
        checksum_length = strength // 32
        checksum = bin(int.from_bytes(
            hashlib.sha256(entropy).digest(), 'big'
        ))[2:].zfill(256)[:checksum_length]
        
        # Combine entropy and checksum
        combined = entropy_bits + checksum
        
        # Split into 11-bit groups and convert to words
        words = []
        for i in range(0, len(combined), 11):
            index = int(combined[i:i+11], 2)
            words.append(BIP39_WORDLIST[index])
            
        return ' '.join(words)
    
    def mnemonic_to_seed(self, mnemonic: str, passphrase: str = "") -> bytes:
        """
        Convert mnemonic to seed following BIP39.
        
        Args:
            mnemonic: Space-separated mnemonic phrase
            passphrase: Optional passphrase for extra security
            
        Returns:
            64-byte seed
        """
        # Normalize strings
        mnemonic = mnemonic.normalize('NFKD')
        passphrase = passphrase.normalize('NFKD')
        
        # Add salt
        salt = ('mnemonic' + passphrase).encode('utf-8')
        
        # Generate seed using PBKDF2
        return hashlib.pbkdf2_hmac(
            'sha512',
            mnemonic.encode('utf-8'),
            salt,
            iterations=2048
        )
    
    def _derive_master_key(self, seed: bytes) -> ExtendedKey:
        """
        Derive master key from seed following BIP32.
        
        Args:
            seed: 64-byte seed
            
        Returns:
            Master extended key
        """
        # Generate master key material
        hmac_obj = hmac.new(b'Bitcoin seed', seed, hashlib.sha512)
        master_key = hmac_obj.digest()
        
        # Split into key and chain code
        key = master_key[:32]
        chain_code = master_key[32:]
        
        return ExtendedKey(
            key=key,
            chain_code=chain_code,
            depth=0,
            index=0,
            parent_fingerprint=bytes([0] * 4),
            is_private=True
        )
    
    def _derive_child_key(self, parent: ExtendedKey, index: int,
                         hardened: bool = False) -> ExtendedKey:
        """
        Derive child key following BIP32.
        
        Args:
            parent: Parent extended key
            index: Child index
            hardened: Whether to use hardened derivation
            
        Returns:
            Child extended key
        """
        if hardened:
            index += 0x80000000
            
        # Data to HMAC
        if hardened:
            data = b'\x00' + parent.key
        else:
            if parent.is_private:
                # Get public key
                public_key = self.ecdsa._scalar_mult(
                    int.from_bytes(parent.key, 'big')
                ).to_bytes(compressed=True)
                data = public_key
            else:
                data = parent.key
                
        data += index.to_bytes(4, 'big')
        
        # Generate child key material
        hmac_obj = hmac.new(parent.chain_code, data, hashlib.sha512)
        child_material = hmac_obj.digest()
        
        # Split into key and chain code
        child_key = child_material[:32]
        child_chain = child_material[32:]
        
        if parent.is_private:
            # Add to parent key (mod n)
            parent_key = int.from_bytes(parent.key, 'big')
            child_key = int.from_bytes(child_key, 'big')
            final_key = (parent_key + child_key) % self.ecdsa.N
            key = final_key.to_bytes(32, 'big')
        else:
            # Add to parent public key point
            parent_point = self.ecdsa.Point.from_bytes(parent.key)
            child_point = self.ecdsa._scalar_mult(
                int.from_bytes(child_key, 'big')
            )
            final_point = self.ecdsa._point_add(parent_point, child_point)
            key = final_point.to_bytes(compressed=True)
        
        # Calculate parent fingerprint
        if parent.is_private:
            parent_pub = self.ecdsa._scalar_mult(
                int.from_bytes(parent.key, 'big')
            ).to_bytes(compressed=True)
        else:
            parent_pub = parent.key
        fingerprint = hashlib.new('ripemd160',
            hashlib.sha256(parent_pub).digest()
        ).digest()[:4]
        
        return ExtendedKey(
            key=key,
            chain_code=child_chain,
            depth=parent.depth + 1,
            index=index,
            parent_fingerprint=fingerprint,
            is_private=parent.is_private
        )
    
    def derive_path(self, seed: bytes, path: str) -> ExtendedKey:
        """
        Derive key from HD path (e.g. "m/44'/0'/0'/0/0").
        
        Args:
            seed: Master seed
            path: Derivation path
            
        Returns:
            Derived extended key
        """
        if not path.startswith('m/'):
            raise ValueError("Invalid path format")
            
        # Start with master key
        key = self._derive_master_key(seed)
        
        # Derive each level
        components = path.split('/')[1:]
        for comp in components:
            hardened = comp.endswith("'")
            index = int(comp[:-1] if hardened else comp)
            key = self._derive_child_key(key, index, hardened)
            
        return key
    
    def get_address(self, public_key: bytes, address_type: str = 'p2pkh') -> str:
        """
        Generate address from public key.
        
        Args:
            public_key: Compressed public key bytes
            address_type: Address type ('p2pkh', 'p2sh', 'bech32')
            
        Returns:
            Address string
        """
        # Hash public key
        h = hashlib.new('ripemd160',
            hashlib.sha256(public_key).digest()
        ).digest()
        
        if address_type == 'p2pkh':
            # Add version byte (0x00 for mainnet)
            versioned = b'\x00' + h
            
            # Add checksum
            checksum = hashlib.sha256(
                hashlib.sha256(versioned).digest()
            ).digest()[:4]
            
            # Encode in base58
            address = self._base58_encode(versioned + checksum)
            
        elif address_type == 'p2sh':
            # Add version byte (0x05 for mainnet)
            versioned = b'\x05' + h
            
            # Add checksum
            checksum = hashlib.sha256(
                hashlib.sha256(versioned).digest()
            ).digest()[:4]
            
            # Encode in base58
            address = self._base58_encode(versioned + checksum)
            
        elif address_type == 'bech32':
            # Implement bech32 encoding here
            # This is a simplified version
            address = 'bc1' + h.hex()
            
        else:
            raise ValueError("Invalid address type")
            
        return address
    
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

class Wallet:
    """
    High-level wallet interface combining HD wallet functionality
    with transaction signing and balance tracking.
    """
    
    def __init__(self, mnemonic: Optional[str] = None):
        """
        Initialize wallet.
        
        Args:
            mnemonic: Optional mnemonic phrase. If None, generates new one.
        """
        self.hd = HDWallet()
        
        if mnemonic is None:
            self.mnemonic = self.hd.generate_mnemonic()
        else:
            self.mnemonic = mnemonic
            
        self.seed = self.hd.mnemonic_to_seed(self.mnemonic)
        self.accounts = {}  # address -> (private_key, public_key)
        
    def create_account(self, index: int = 0) -> str:
        """
        Create new account (address) using BIP44.
        
        Args:
            index: Account index
            
        Returns:
            Address string
        """
        # BIP44 path: m/44'/0'/index'/0/0
        path = f"m/44'/0'/{index}'/0/0"
        
        # Derive private key
        derived = self.hd.derive_path(self.seed, path)
        private_key = derived.key
        
        # Get public key
        public_key = self.hd.ecdsa._scalar_mult(
            int.from_bytes(private_key, 'big')
        ).to_bytes(compressed=True)
        
        # Generate address
        address = self.hd.get_address(public_key)
        
        # Store account
        self.accounts[address] = (private_key, public_key)
        
        return address
    
    def sign_transaction(self, from_address: str, to_address: str,
                        amount: float, fee: float) -> Optional[Transaction]:
        """
        Sign a transaction.
        
        Args:
            from_address: Sender address
            to_address: Recipient address
            amount: Amount to send
            fee: Transaction fee
            
        Returns:
            Signed transaction or None if error
        """
        if from_address not in self.accounts:
            return None
            
        private_key, _ = self.accounts[from_address]
        
        # Create transaction
        tx = Transaction(
            from_address=from_address,
            to_address=to_address,
            amount=amount,
            fee=fee,
            nonce=0  # Should get from network
        )
        
        # Sign transaction
        message = tx.get_hash().encode()
        tx.signature = self.hd.ecdsa.sign(message, private_key)
        
        return tx
    
    def export_private_key(self, address: str) -> Optional[str]:
        """
        Export private key in WIF format.
        
        Args:
            address: Address to export key for
            
        Returns:
            WIF private key or None if address not found
        """
        if address not in self.accounts:
            return None
            
        private_key, _ = self.accounts[address]
        
        # Add version byte (0x80 for mainnet)
        versioned = b'\x80' + private_key
        
        # Add compression flag
        versioned += b'\x01'
        
        # Add checksum
        checksum = hashlib.sha256(
            hashlib.sha256(versioned).digest()
        ).digest()[:4]
        
        # Encode in base58
        return self.hd._base58_encode(versioned + checksum)

# Example usage and tests
if __name__ == "__main__":
    # Create new wallet
    wallet = Wallet()
    print(f"Mnemonic: {wallet.mnemonic}")
    
    # Create accounts
    for i in range(3):
        address = wallet.create_account(i)
        print(f"\nAccount {i}:")
        print(f"Address: {address}")
        print(f"Private key (WIF): {wallet.export_private_key(address)}")
    
    # Sign transaction
    address1 = wallet.create_account(0)
    address2 = wallet.create_account(1)
    
    tx = wallet.sign_transaction(address1, address2, 1.0, 0.1)
    if tx:
        print("\nSigned Transaction:")
        print(f"From: {tx.from_address}")
        print(f"To: {tx.to_address}")
        print(f"Amount: {tx.amount}")
        print(f"Fee: {tx.fee}")
        print(f"Signature: {tx.signature.hex()}")