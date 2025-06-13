"""
Blockchain Implementation

This module provides a basic blockchain implementation with support for
fork resolution and chain reorganization.
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import time
from .block import Block, Transaction
from ..hashing.sha256 import SHA256

class Blockchain:
    """Main blockchain class with fork handling and reorganization."""
    
    def __init__(self):
        """Initialize blockchain with genesis block."""
        self.blocks: Dict[str, Block] = {}  # hash -> block
        self.block_heights: Dict[int, List[str]] = defaultdict(list)  # height -> [hashes]
        self.main_chain: List[str] = []  # List of block hashes in main chain
        self.pending_transactions: List[Transaction] = []  # Mempool
        self.difficulty: int = 4  # Number of leading zeros required
        self.target_block_time: int = 600  # Target 10 minutes between blocks
        self.difficulty_adjustment_interval: int = 2016  # Blocks between difficulty adjustments
        
        # Create genesis block
        genesis = Block(version=1)
        genesis_hash = genesis.get_hash()
        self.blocks[genesis_hash] = genesis
        self.block_heights[0].append(genesis_hash)
        self.main_chain.append(genesis_hash)
    
    def add_block(self, block: Block) -> bool:
        """
        Add a block to the blockchain.
        
        Args:
            block: Block to add
            
        Returns:
            True if block was added successfully
        """
        # Verify block
        if not block.verify():
            return False
            
        # Check if we have the previous block
        if block.header.previous_hash not in self.blocks:
            return False
            
        # Check if block already exists
        block_hash = block.get_hash()
        if block_hash in self.blocks:
            return False
            
        # Get block height
        prev_height = self.get_block_height(block.header.previous_hash)
        if prev_height is None:
            return False
        height = prev_height + 1
        
        # Store block
        self.blocks[block_hash] = block
        self.block_heights[height].append(block_hash)
        
        # Check if this creates a longer chain
        if self.is_better_chain(block_hash):
            self.reorganize_chain(block_hash)
            
        return True
    
    def get_block_height(self, block_hash: str) -> Optional[int]:
        """Get height of a block."""
        for height, hashes in self.block_heights.items():
            if block_hash in hashes:
                return height
        return None
    
    def get_chain_work(self, tip_hash: str) -> int:
        """
        Calculate total chain work (sum of difficulties).
        Simple implementation - in practice would use big integers.
        """
        work = 0
        current_hash = tip_hash
        
        while current_hash in self.blocks:
            block = self.blocks[current_hash]
            work += 2 ** block.header.difficulty_target
            current_hash = block.header.previous_hash
            
        return work
    
    def is_better_chain(self, tip_hash: str) -> bool:
        """
        Check if chain ending in tip_hash is better than current main chain.
        Uses total chain work for comparison.
        """
        if not self.main_chain:
            return True
            
        current_work = self.get_chain_work(self.main_chain[-1])
        new_work = self.get_chain_work(tip_hash)
        
        return new_work > current_work
    
    def get_fork_point(self, chain_tip: str) -> Optional[str]:
        """Find the last common block between main chain and fork."""
        fork_chain = []
        current_hash = chain_tip
        
        # Build fork chain
        while current_hash in self.blocks:
            fork_chain.append(current_hash)
            current_hash = self.blocks[current_hash].header.previous_hash
            
        # Find common ancestor
        for block_hash in fork_chain:
            if block_hash in self.main_chain:
                return block_hash
                
        return None
    
    def reorganize_chain(self, new_tip: str) -> None:
        """
        Reorganize chain to make the fork the new main chain.
        
        Args:
            new_tip: Hash of new chain tip
        """
        fork_point = self.get_fork_point(new_tip)
        if fork_point is None:
            return
            
        # Build new chain
        new_chain = []
        current_hash = new_tip
        
        while current_hash != fork_point:
            new_chain.insert(0, current_hash)
            current_hash = self.blocks[current_hash].header.previous_hash
            
        # Find fork point index in main chain
        fork_index = self.main_chain.index(fork_point)
        
        # Update main chain
        self.main_chain = self.main_chain[:fork_index + 1] + new_chain
        
        # Adjust difficulty if needed
        self.adjust_difficulty()
    
    def adjust_difficulty(self) -> None:
        """
        Adjust mining difficulty based on block times.
        Simple implementation - real networks use more sophisticated algorithms.
        """
        if len(self.main_chain) % self.difficulty_adjustment_interval != 0:
            return
            
        # Get start and end blocks of the period
        start_block = self.blocks[self.main_chain[-self.difficulty_adjustment_interval]]
        end_block = self.blocks[self.main_chain[-1]]
        
        # Calculate actual time taken
        time_taken = end_block.header.timestamp - start_block.header.timestamp
        target_time = self.target_block_time * self.difficulty_adjustment_interval
        
        # Adjust difficulty
        if time_taken < target_time // 4:
            self.difficulty += 1
        elif time_taken > target_time * 4:
            self.difficulty = max(1, self.difficulty - 1)
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """
        Add transaction to mempool.
        
        Args:
            transaction: Transaction to add
            
        Returns:
            True if transaction was added
        """
        # Basic verification
        if not self.verify_transaction(transaction):
            return False
            
        self.pending_transactions.append(transaction)
        return True
    
    def verify_transaction(self, transaction: Transaction) -> bool:
        """
        Verify transaction.
        Simple implementation - real networks do much more verification.
        """
        return (
            transaction.amount > 0 and
            transaction.fee >= 0 and
            transaction.nonce >= 0
        )
    
    def create_block(self, miner_address: str) -> Block:
        """
        Create a new block with pending transactions.
        
        Args:
            miner_address: Address to receive mining reward
            
        Returns:
            New block ready for mining
        """
        # Sort transactions by fee
        transactions = sorted(
            self.pending_transactions,
            key=lambda tx: tx.fee,
            reverse=True
        )
        
        # Create coinbase transaction
        coinbase = Transaction(
            from_address="0" * 64,
            to_address=miner_address,
            amount=50.0,  # Block reward
            fee=0.0,
            nonce=len(self.main_chain)
        )
        
        # Create block
        block = Block(
            version=1,
            previous_hash=self.main_chain[-1]
        )
        
        # Add transactions
        block.add_transaction(coinbase)
        for tx in transactions[:999]:  # Limit block size
            block.add_transaction(tx)
            
        return block
    
    def mine_block(self, miner_address: str) -> Optional[Block]:
        """
        Create and mine a new block.
        
        Args:
            miner_address: Address to receive mining reward
            
        Returns:
            Mined block if successful
        """
        block = self.create_block(miner_address)
        
        try:
            block.mine(self.difficulty)
            if self.add_block(block):
                # Remove included transactions from mempool
                included_hashes = {tx.get_hash() for tx in block.transactions}
                self.pending_transactions = [
                    tx for tx in self.pending_transactions
                    if tx.get_hash() not in included_hashes
                ]
                return block
        except Exception as e:
            print(f"Mining failed: {e}")
            
        return None
    
    def get_balance(self, address: str) -> float:
        """
        Calculate balance of an address.
        Simple implementation - real networks use UTXOs or account state.
        """
        balance = 0.0
        
        # Process all transactions in main chain
        for block_hash in self.main_chain:
            block = self.blocks[block_hash]
            for tx in block.transactions:
                if tx.to_address == address:
                    balance += tx.amount
                if tx.from_address == address:
                    balance -= (tx.amount + tx.fee)
                    
        return balance
    
    def get_block_by_height(self, height: int) -> Optional[Block]:
        """Get block at specific height in main chain."""
        if 0 <= height < len(self.main_chain):
            return self.blocks[self.main_chain[height]]
        return None
    
    def get_latest_block(self) -> Block:
        """Get the latest block in main chain."""
        return self.blocks[self.main_chain[-1]]
    
    def get_chain_stats(self) -> dict:
        """Get blockchain statistics."""
        latest_block = self.get_latest_block()
        return {
            'height': len(self.main_chain) - 1,
            'difficulty': self.difficulty,
            'last_block_time': latest_block.header.timestamp,
            'pending_transactions': len(self.pending_transactions),
            'total_blocks': len(self.blocks),
            'fork_blocks': sum(len(hashes) for hashes in self.block_heights.values()) - len(self.main_chain)
        }