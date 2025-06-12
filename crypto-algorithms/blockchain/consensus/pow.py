"""
Proof of Work (PoW) Consensus Implementation

This module provides a detailed implementation of Proof of Work consensus
with difficulty adjustment and energy consumption estimation.
"""

from typing import Optional, Tuple
import time
import math
from ..block import Block, BlockHeader
from ...hashing.sha256 import SHA256

class ProofOfWork:
    """
    Proof of Work consensus implementation with difficulty adjustment
    and energy consumption estimation.
    """
    
    def __init__(self, 
                 initial_difficulty: int = 4,
                 target_block_time: int = 600,  # 10 minutes
                 difficulty_adjustment_interval: int = 2016,  # ~2 weeks
                 hash_power_per_watt: float = 50e6):  # 50 MH/s per watt (typical ASIC)
        """
        Initialize PoW consensus.
        
        Args:
            initial_difficulty: Initial mining difficulty (leading zeros)
            target_block_time: Target time between blocks in seconds
            difficulty_adjustment_interval: Blocks between difficulty adjustments
            hash_power_per_watt: Hash operations per watt of power
        """
        self.difficulty = initial_difficulty
        self.target_block_time = target_block_time
        self.adjustment_interval = difficulty_adjustment_interval
        self.hash_power_per_watt = hash_power_per_watt
        self.block_times = []  # Track block times for adjustment
    
    def get_target(self, difficulty: int) -> str:
        """Get target hash based on difficulty."""
        return '0' * difficulty + 'f' * (64 - difficulty)
    
    def check_pow(self, block_header: BlockHeader) -> bool:
        """
        Check if block meets proof of work requirement.
        
        Args:
            block_header: Block header to check
            
        Returns:
            True if proof of work is valid
        """
        block_hash = block_header.get_hash()
        target = self.get_target(block_header.difficulty_target)
        return int(block_hash, 16) <= int(target, 16)
    
    def mine_block(self, block: Block, max_nonce: int = 2**32) -> Tuple[bool, int, float]:
        """
        Mine a block using proof of work.
        
        Args:
            block: Block to mine
            max_nonce: Maximum nonce value to try
            
        Returns:
            Tuple of (success, hashes_tried, estimated_energy_kWh)
        """
        block.header.difficulty_target = self.difficulty
        target = self.get_target(self.difficulty)
        
        start_time = time.time()
        nonce = 0
        
        while nonce < max_nonce:
            block.header.nonce = nonce
            block_hash = block.get_hash()
            
            if int(block_hash, 16) <= int(target, 16):
                # Calculate statistics
                end_time = time.time()
                time_taken = end_time - start_time
                
                # Estimate energy consumption
                hashes_per_second = nonce / time_taken
                watts_used = hashes_per_second / self.hash_power_per_watt
                kwh_used = (watts_used * time_taken) / 3600000  # Convert to kWh
                
                # Update block times for difficulty adjustment
                self.block_times.append(end_time)
                if len(self.block_times) > self.adjustment_interval:
                    self.block_times.pop(0)
                
                return True, nonce, kwh_used
                
            nonce += 1
            
        return False, nonce, 0.0
    
    def adjust_difficulty(self) -> None:
        """
        Adjust mining difficulty based on recent block times.
        Uses a more sophisticated algorithm than basic blockchain implementation.
        """
        if len(self.block_times) < self.adjustment_interval:
            return
            
        # Calculate actual time taken for last interval
        time_taken = self.block_times[-1] - self.block_times[0]
        target_time = self.target_block_time * (self.adjustment_interval - 1)
        
        # Calculate ratio with dampening
        ratio = time_taken / target_time
        ratio = max(0.25, min(4.0, ratio))  # Limit adjustment to 4x in either direction
        
        # Adjust difficulty logarithmically
        self.difficulty = max(1, round(self.difficulty - math.log2(ratio)))
    
    def estimate_network_hashrate(self, recent_blocks: list) -> float:
        """
        Estimate total network hashrate based on recent blocks.
        
        Args:
            recent_blocks: List of recent blocks with timestamps
            
        Returns:
            Estimated network hashrate in hashes per second
        """
        if len(recent_blocks) < 2:
            return 0.0
            
        # Calculate average time between blocks
        times = [block.header.timestamp for block in recent_blocks]
        time_diffs = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
        avg_time = sum(time_diffs) / len(time_diffs)
        
        # Calculate average difficulty
        difficulties = [block.header.difficulty_target for block in recent_blocks]
        avg_difficulty = sum(difficulties) / len(difficulties)
        
        # Estimate hashrate
        # On average, need to try 2^difficulty hashes to find a block
        hashes_per_block = 2 ** avg_difficulty
        return hashes_per_block / avg_time
    
    def estimate_energy_consumption(self, hashrate: float) -> float:
        """
        Estimate network energy consumption in kWh per day.
        
        Args:
            hashrate: Network hashrate in hashes per second
            
        Returns:
            Estimated daily energy consumption in kWh
        """
        watts = hashrate / self.hash_power_per_watt
        kwh_per_day = (watts * 24) / 1000
        return kwh_per_day
    
    def get_mining_stats(self, recent_blocks: list) -> dict:
        """
        Get comprehensive mining statistics.
        
        Args:
            recent_blocks: List of recent blocks
            
        Returns:
            Dictionary of mining statistics
        """
        hashrate = self.estimate_network_hashrate(recent_blocks)
        energy = self.estimate_energy_consumption(hashrate)
        
        return {
            'difficulty': self.difficulty,
            'network_hashrate': hashrate,
            'daily_energy_kwh': energy,
            'avg_block_time': self.target_block_time,
            'next_adjustment_blocks': self.adjustment_interval - len(self.block_times),
            'estimated_yearly_energy_mwh': energy * 365 / 1000
        }

# Example usage and tests
if __name__ == "__main__":
    # Create PoW consensus with test parameters
    pow_consensus = ProofOfWork(
        initial_difficulty=4,
        target_block_time=10,  # 10 seconds for testing
        difficulty_adjustment_interval=10,
        hash_power_per_watt=1e6  # 1 MH/s per watt for testing
    )
    
    # Create and mine some blocks
    recent_blocks = []
    for i in range(5):
        block = Block(version=1)
        print(f"\nMining block {i+1}...")
        
        success, hashes, energy = pow_consensus.mine_block(block)
        if success:
            recent_blocks.append(block)
            print(f"Block mined with {hashes} hashes")
            print(f"Estimated energy used: {energy:.6f} kWh")
            print(f"Block hash: {block.get_hash()}")
            
            # Adjust difficulty periodically
            pow_consensus.adjust_difficulty()
    
    # Get mining statistics
    stats = pow_consensus.get_mining_stats(recent_blocks)
    print("\nMining Statistics:")
    for key, value in stats.items():
        if 'hashrate' in key:
            print(f"{key}: {value:.2e} H/s")
        elif 'energy' in key:
            print(f"{key}: {value:.2f} kWh")
        else:
            print(f"{key}: {value}")