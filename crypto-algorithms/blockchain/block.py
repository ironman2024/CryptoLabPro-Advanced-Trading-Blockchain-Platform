"""
Block Implementation

This module provides a basic implementation of a blockchain block structure
with support for both Bitcoin-style and Ethereum-style blocks.
"""

from dataclasses import dataclass
from typing import List, Optional, Union, Dict
import time
import json
import hashlib
from ..hashing.sha256 import SHA256
from ..hashing.keccak256 import Keccak256

@dataclass
class Transaction:
    """Basic transaction structure."""
    from_address: str
    to_address: str
    amount: float
    fee: float
    nonce: int
    signature: Optional[bytes] = None
    
    def to_dict(self) -> dict:
        """Convert transaction to dictionary."""
        return {
            'from': self.from_address,
            'to': self.to_address,
            'amount': self.amount,
            'fee': self.fee,
            'nonce': self.nonce,
            'signature': self.signature.hex() if self.signature else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Transaction':
        """Create transaction from dictionary."""
        return cls(
            from_address=data['from'],
            to_address=data['to'],
            amount=data['amount'],
            fee=data['fee'],
            nonce=data['nonce'],
            signature=bytes.fromhex(data['signature']) if data['signature'] else None
        )
    
    def get_hash(self) -> str:
        """Calculate transaction hash."""
        # Create a copy without signature for hashing
        tx_dict = self.to_dict()
        tx_dict['signature'] = None
        return Keccak256().hash(json.dumps(tx_dict, sort_keys=True))

class MerkleTree:
    """Merkle Tree implementation for transaction hashing."""
    
    @staticmethod
    def hash_pair(left: str, right: str) -> str:
        """Hash a pair of hex strings."""
        combined = bytes.fromhex(left) + bytes.fromhex(right)
        return SHA256().hash(combined)
    
    @classmethod
    def build(cls, transactions: List[Transaction]) -> List[List[str]]:
        """
        Build a Merkle Tree from transactions.
        
        Returns:
            List of levels in the tree, each level containing hash strings
        """
        if not transactions:
            return [[SHA256().hash(b'')]]
            
        # Get transaction hashes
        current_level = [tx.get_hash() for tx in transactions]
        
        # Ensure even number of elements
        if len(current_level) % 2 == 1:
            current_level.append(current_level[-1])
            
        levels = [current_level]
        
        # Build tree bottom-up
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                next_level.append(cls.hash_pair(
                    current_level[i],
                    current_level[i + 1] if i + 1 < len(current_level) else current_level[i]
                ))
            current_level = next_level
            levels.insert(0, current_level)
            
        return levels
    
    @classmethod
    def get_root(cls, transactions: List[Transaction]) -> str:
        """Get Merkle root hash of transactions."""
        tree = cls.build(transactions)
        return tree[0][0] if tree else SHA256().hash(b'')
    
    @classmethod
    def get_proof(cls, transactions: List[Transaction], tx_index: int) -> List[Dict[str, str]]:
        """
        Get Merkle proof for a transaction.
        
        Returns:
            List of dictionaries with position ('left' or 'right') and hash
        """
        if not transactions or tx_index >= len(transactions):
            return []
            
        tree = cls.build(transactions)
        proof = []
        
        # Start from bottom level
        current_index = tx_index
        
        for level in tree[1:]:  # Skip root level
            is_right = current_index % 2 == 0
            pair_index = current_index - 1 if is_right else current_index + 1
            
            if pair_index < len(level):
                proof.append({
                    'position': 'left' if is_right else 'right',
                    'hash': level[pair_index]
                })
            
            current_index //= 2
            
        return proof
    
    @classmethod
    def verify_proof(cls, tx_hash: str, proof: List[Dict[str, str]], root: str) -> bool:
        """Verify a Merkle proof."""
        current_hash = tx_hash
        
        for step in proof:
            if step['position'] == 'left':
                current_hash = cls.hash_pair(step['hash'], current_hash)
            else:
                current_hash = cls.hash_pair(current_hash, step['hash'])
                
        return current_hash == root

@dataclass
class BlockHeader:
    """Block header structure."""
    version: int
    previous_hash: str
    merkle_root: str
    timestamp: int
    difficulty_target: int
    nonce: int
    
    def to_dict(self) -> dict:
        """Convert header to dictionary."""
        return {
            'version': self.version,
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'timestamp': self.timestamp,
            'difficulty_target': self.difficulty_target,
            'nonce': self.nonce
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BlockHeader':
        """Create header from dictionary."""
        return cls(**data)
    
    def get_hash(self) -> str:
        """Calculate block header hash."""
        header_dict = self.to_dict()
        return SHA256().hash(SHA256().hash(json.dumps(header_dict, sort_keys=True)))

class Block:
    """Main block class supporting both Bitcoin-style and Ethereum-style blocks."""
    
    def __init__(self, version: int = 1, previous_hash: str = '0' * 64):
        """
        Initialize a new block.
        
        Args:
            version: Block version
            previous_hash: Hash of previous block
        """
        self.transactions: List[Transaction] = []
        self.header = BlockHeader(
            version=version,
            previous_hash=previous_hash,
            merkle_root='0' * 64,
            timestamp=int(time.time()),
            difficulty_target=0,
            nonce=0
        )
        self.state_root: Optional[str] = None  # For Ethereum-style blocks
        self.receipts_root: Optional[str] = None  # For Ethereum-style blocks
        
    def add_transaction(self, transaction: Transaction) -> None:
        """Add a transaction to the block."""
        self.transactions.append(transaction)
        self.header.merkle_root = MerkleTree.get_root(self.transactions)
        
    def get_hash(self) -> str:
        """Get block hash."""
        return self.header.get_hash()
    
    def mine(self, difficulty: int) -> None:
        """
        Mine the block with proof of work.
        
        Args:
            difficulty: Number of leading zeros required in hash
        """
        self.header.difficulty_target = difficulty
        target = '0' * difficulty
        
        while not self.get_hash().startswith(target):
            self.header.nonce += 1
            
    def verify(self) -> bool:
        """
        Verify block integrity.
        
        Returns:
            True if block is valid
        """
        # Verify merkle root
        if self.header.merkle_root != MerkleTree.get_root(self.transactions):
            return False
            
        # Verify proof of work
        if not self.get_hash().startswith('0' * self.header.difficulty_target):
            return False
            
        # Verify transactions
        for tx in self.transactions:
            if not self.verify_transaction(tx):
                return False
                
        return True
    
    def verify_transaction(self, transaction: Transaction) -> bool:
        """
        Verify a single transaction.
        
        Args:
            transaction: Transaction to verify
            
        Returns:
            True if transaction is valid
        """
        # Basic verification - in practice, would check signatures,
        # account balances, nonces, etc.
        return (
            transaction.amount > 0 and
            transaction.fee >= 0 and
            transaction.nonce >= 0
        )
    
    def to_dict(self) -> dict:
        """Convert block to dictionary."""
        return {
            'header': self.header.to_dict(),
            'transactions': [tx.to_dict() for tx in self.transactions],
            'state_root': self.state_root,
            'receipts_root': self.receipts_root
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Block':
        """Create block from dictionary."""
        block = cls()
        block.header = BlockHeader.from_dict(data['header'])
        block.transactions = [Transaction.from_dict(tx) for tx in data['transactions']]
        block.state_root = data['state_root']
        block.receipts_root = data['receipts_root']
        return block

# Example usage and tests
if __name__ == "__main__":
    # Create a block
    block = Block(version=1)
    
    # Add some transactions
    transactions = [
        Transaction("Alice", "Bob", 1.0, 0.1, 1),
        Transaction("Bob", "Charlie", 0.5, 0.1, 1),
        Transaction("Charlie", "Alice", 0.25, 0.1, 1)
    ]
    
    for tx in transactions:
        block.add_transaction(tx)
    
    # Mine the block
    print("Mining block...")
    block.mine(4)  # Require 4 leading zeros
    print(f"Block mined with nonce: {block.header.nonce}")
    print(f"Block hash: {block.get_hash()}")
    
    # Verify the block
    print(f"Block is valid: {block.verify()}")
    
    # Demonstrate Merkle proofs
    print("\nMerkle Tree Demonstration:")
    tx = transactions[1]  # Second transaction
    tx_hash = tx.get_hash()
    merkle_proof = MerkleTree.get_proof(transactions, 1)
    merkle_root = block.header.merkle_root
    
    print(f"Transaction hash: {tx_hash}")
    print(f"Merkle root: {merkle_root}")
    print(f"Proof: {merkle_proof}")
    print(f"Proof verifies: {MerkleTree.verify_proof(tx_hash, merkle_proof, merkle_root)}")