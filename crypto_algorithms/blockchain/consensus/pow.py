"""
Proof of Work (PoW) consensus implementation
"""

import hashlib
import time
from typing import Tuple, Any


class ProofOfWork:
    """
    Proof of Work consensus mechanism implementation.
    """
    
    def __init__(self, initial_difficulty: int = 4,
                 target_block_time: int = 10,
                 difficulty_adjustment_interval: int = 10):
        """
        Initialize PoW consensus.
        
        Args:
            initial_difficulty: Number of leading zeros required in block hash
            target_block_time: Target time between blocks in seconds
            difficulty_adjustment_interval: Number of blocks between difficulty adjustments
        """
        self.difficulty = initial_difficulty
        self.target_block_time = target_block_time
        self.adjustment_interval = difficulty_adjustment_interval
        self.last_adjustment_time = time.time()
        self.blocks_since_adjustment = 0
        
    def mine_block(self, block: Any) -> Tuple[bool, int, float]:
        """
        Mine a block using proof of work.
        
        Args:
            block: Block to mine
            
        Returns:
            Tuple of (success, hash_attempts, energy_used)
        """
        nonce = 0
        start_time = time.time()
        target = '0' * self.difficulty
        
        while True:
            block.header.nonce = nonce
            block_hash = block.get_hash()
            
            if block_hash.startswith(target):
                end_time = time.time()
                time_taken = end_time - start_time
                energy = self._calculate_energy_usage(nonce, time_taken)
                return True, nonce, energy
                
            nonce += 1
            
            # Prevent infinite loops in testing
            if nonce > 1000000:
                return False, nonce, 0
                
    def check_pow(self, block_header: Any) -> bool:
        """
        Verify proof of work for a block.
        
        Args:
            block_header: Header of block to verify
            
        Returns:
            True if valid, False otherwise
        """
        block_hash = block_header.get_hash()
        target = '0' * self.difficulty
        return block_hash.startswith(target)
        
    def adjust_difficulty(self) -> None:
        """
        Adjust mining difficulty based on block times.
        """
        self.blocks_since_adjustment += 1
        
        if self.blocks_since_adjustment >= self.adjustment_interval:
            current_time = time.time()
            time_passed = current_time - self.last_adjustment_time
            expected_time = self.target_block_time * self.adjustment_interval
            
            # Adjust difficulty up or down based on time ratio
            if time_passed < expected_time * 0.5:
                self.difficulty += 1
            elif time_passed > expected_time * 2:
                self.difficulty = max(1, self.difficulty - 1)
                
            self.last_adjustment_time = current_time
            self.blocks_since_adjustment = 0
            
    def _calculate_energy_usage(self, hashes: int, time_taken: float) -> float:
        """
        Estimate energy usage for mining.
        
        Args:
            hashes: Number of hash calculations performed
            time_taken: Time taken in seconds
            
        Returns:
            Estimated energy usage in joules
        """
        # Rough estimate based on typical CPU power consumption
        watts_per_hash = 0.001
        return hashes * watts_per_hash * time_taken