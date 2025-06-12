"""
Proof of Work (PoW) Consensus Implementation

This module provides a detailed implementation of Proof of Work consensus
with difficulty adjustment and energy consumption estimation.
"""

from typing import Optional, Tuple, List, Dict, Any
import time
import math
import hashlib
import json
from datetime import datetime

class Block:
    """Simple block structure for PoW demonstration."""
    
    def __init__(self, index: int, previous_hash: str, timestamp: float = None, 
                 data: str = "", nonce: int = 0, difficulty: int = 0):
        """
        Initialize a block.
        
        Args:
            index: Block height
            previous_hash: Hash of the previous block
            timestamp: Block creation time (defaults to current time)
            data: Block data (could be transactions, etc.)
            nonce: Nonce value for mining
            difficulty: Mining difficulty (number of leading zeros)
        """
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.data = data
        self.nonce = nonce
        self.difficulty = difficulty
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate block hash."""
        block_string = f"{self.index}{self.previous_hash}{self.timestamp}{self.data}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary."""
        return {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "data": self.data,
            "nonce": self.nonce,
            "difficulty": self.difficulty,
            "hash": self.hash
        }
    
    def __str__(self) -> str:
        """String representation of block."""
        return (f"Block #{self.index}\n"
                f"Hash: {self.hash}\n"
                f"Previous: {self.previous_hash}\n"
                f"Time: {datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Difficulty: {self.difficulty}\n"
                f"Nonce: {self.nonce}\n"
                f"Data: {self.data[:50]}{'...' if len(self.data) > 50 else ''}")

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
        
        # For educational purposes
        self.mining_steps = []
        self.difficulty_adjustments = []
    
    def get_target(self, difficulty: int) -> str:
        """Get target hash based on difficulty."""
        return '0' * difficulty + 'f' * (64 - difficulty)
    
    def check_pow(self, block: Block) -> bool:
        """
        Check if block meets proof of work requirement.
        
        Args:
            block: Block to check
            
        Returns:
            True if proof of work is valid
        """
        block_hash = block.hash
        target = self.get_target(block.difficulty)
        return int(block_hash, 16) <= int(target, 16)
    
    def mine_block(self, block: Block, max_nonce: int = 2**32, 
                  track_steps: bool = False, step_interval: int = 100000) -> Tuple[bool, int, float, List[Dict]]:
        """
        Mine a block using proof of work.
        
        Args:
            block: Block to mine
            max_nonce: Maximum nonce value to try
            track_steps: Whether to track mining steps for visualization
            step_interval: How often to record steps
            
        Returns:
            Tuple of (success, hashes_tried, estimated_energy_kWh, steps)
        """
        block.difficulty = self.difficulty
        target = self.get_target(self.difficulty)
        
        # Reset mining steps
        self.mining_steps = []
        
        start_time = time.time()
        nonce = 0
        
        if track_steps:
            self.mining_steps.append({
                "time": 0,
                "nonce": nonce,
                "hash": block.hash,
                "target": target
            })
        
        while nonce < max_nonce:
            block.nonce = nonce
            block.hash = block.calculate_hash()
            
            if track_steps and nonce % step_interval == 0:
                self.mining_steps.append({
                    "time": time.time() - start_time,
                    "nonce": nonce,
                    "hash": block.hash,
                    "target": target
                })
            
            if int(block.hash, 16) <= int(target, 16):
                # Calculate statistics
                end_time = time.time()
                time_taken = end_time - start_time
                
                # Estimate energy consumption
                hashes_per_second = nonce / time_taken if time_taken > 0 else 0
                watts_used = hashes_per_second / self.hash_power_per_watt
                kwh_used = (watts_used * time_taken) / 3600000  # Convert to kWh
                
                # Update block times for difficulty adjustment
                self.block_times.append(end_time)
                if len(self.block_times) > self.adjustment_interval:
                    self.block_times.pop(0)
                
                # Record final step
                if track_steps:
                    self.mining_steps.append({
                        "time": time_taken,
                        "nonce": nonce,
                        "hash": block.hash,
                        "target": target,
                        "success": True
                    })
                
                return True, nonce, kwh_used, self.mining_steps
                
            nonce += 1
            
        # Mining failed
        end_time = time.time()
        time_taken = end_time - start_time
        
        if track_steps:
            self.mining_steps.append({
                "time": time_taken,
                "nonce": nonce,
                "hash": block.hash,
                "target": target,
                "success": False
            })
            
        return False, nonce, 0.0, self.mining_steps
    
    def adjust_difficulty(self) -> int:
        """
        Adjust mining difficulty based on recent block times.
        Uses a more sophisticated algorithm than basic blockchain implementation.
        
        Returns:
            New difficulty value
        """
        if len(self.block_times) < 2:
            return self.difficulty
            
        # Calculate actual time taken for last interval
        blocks_to_consider = min(len(self.block_times) - 1, self.adjustment_interval)
        time_taken = self.block_times[-1] - self.block_times[-1 - blocks_to_consider]
        target_time = self.target_block_time * blocks_to_consider
        
        # Calculate ratio with dampening
        ratio = time_taken / target_time if target_time > 0 else 1.0
        ratio = max(0.25, min(4.0, ratio))  # Limit adjustment to 4x in either direction
        
        # Adjust difficulty logarithmically
        old_difficulty = self.difficulty
        self.difficulty = max(1, round(self.difficulty - math.log2(ratio)))
        
        # Record adjustment for educational purposes
        self.difficulty_adjustments.append({
            "time": self.block_times[-1],
            "blocks_considered": blocks_to_consider,
            "time_taken": time_taken,
            "target_time": target_time,
            "ratio": ratio,
            "old_difficulty": old_difficulty,
            "new_difficulty": self.difficulty
        })
        
        return self.difficulty
    
    def estimate_network_hashrate(self, recent_blocks: List[Block]) -> float:
        """
        Estimate total network hashrate based on recent blocks.
        
        Args:
            recent_blocks: List of recent blocks
            
        Returns:
            Estimated network hashrate in hashes per second
        """
        if len(recent_blocks) < 2:
            return 0.0
            
        # Calculate average time between blocks
        times = [block.timestamp for block in recent_blocks]
        time_diffs = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
        avg_time = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        
        # Calculate average difficulty
        difficulties = [block.difficulty for block in recent_blocks]
        avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 0
        
        # Estimate hashrate
        # On average, need to try 2^difficulty hashes to find a block
        if avg_time > 0:
            hashes_per_block = 2 ** avg_difficulty
            return hashes_per_block / avg_time
        return 0.0
    
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
    
    def get_mining_stats(self, recent_blocks: List[Block]) -> Dict[str, Any]:
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
            'estimated_yearly_energy_mwh': energy * 365 / 1000,
            'target': self.get_target(self.difficulty)
        }
    
    def get_difficulty_history(self) -> List[Dict[str, Any]]:
        """
        Get difficulty adjustment history.
        
        Returns:
            List of difficulty adjustments
        """
        return self.difficulty_adjustments
    
    def export_mining_steps(self) -> str:
        """
        Export mining steps as JSON for visualization.
        
        Returns:
            JSON string of mining steps
        """
        return json.dumps(self.mining_steps, indent=2)

class Blockchain:
    """Simple blockchain implementation using PoW consensus."""
    
    def __init__(self, pow_consensus: ProofOfWork = None):
        """
        Initialize blockchain.
        
        Args:
            pow_consensus: PoW consensus mechanism (creates default if None)
        """
        self.chain = []
        self.pow = pow_consensus if pow_consensus else ProofOfWork()
        
        # Create genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self) -> Block:
        """
        Create the genesis block.
        
        Returns:
            Genesis block
        """
        genesis = Block(0, "0" * 64, data="Genesis Block")
        self.chain.append(genesis)
        return genesis
    
    def get_latest_block(self) -> Block:
        """
        Get the latest block in the chain.
        
        Returns:
            Latest block
        """
        return self.chain[-1]
    
    def add_block(self, data: str, track_mining: bool = False) -> Tuple[Block, Dict[str, Any]]:
        """
        Mine and add a new block to the chain.
        
        Args:
            data: Block data
            track_mining: Whether to track mining steps
            
        Returns:
            Tuple of (new block, mining stats)
        """
        latest_block = self.get_latest_block()
        new_block = Block(
            index=latest_block.index + 1,
            previous_hash=latest_block.hash,
            data=data
        )
        
        # Mine the block
        success, hashes, energy, steps = self.pow.mine_block(new_block, track_steps=track_mining)
        
        if success:
            self.chain.append(new_block)
            
            # Adjust difficulty periodically
            if new_block.index % self.pow.adjustment_interval == 0:
                self.pow.adjust_difficulty()
            
            mining_stats = {
                'hashes_tried': hashes,
                'energy_kwh': energy,
                'time_taken': steps[-1]['time'] if steps else 0,
                'difficulty': new_block.difficulty
            }
            
            return new_block, mining_stats
        
        raise Exception("Failed to mine block")
    
    def is_chain_valid(self) -> bool:
        """
        Validate the entire blockchain.
        
        Returns:
            True if chain is valid
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            # Check hash integrity
            if current.hash != current.calculate_hash():
                return False
            
            # Check chain continuity
            if current.previous_hash != previous.hash:
                return False
            
            # Check proof of work
            if not self.pow.check_pow(current):
                return False
        
        return True
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """
        Convert blockchain to list of dictionaries.
        
        Returns:
            List of block dictionaries
        """
        return [block.to_dict() for block in self.chain]
    
    def export_json(self) -> str:
        """
        Export blockchain as JSON.
        
        Returns:
            JSON string of blockchain
        """
        return json.dumps(self.to_dict(), indent=2)

# Example usage and tests
if __name__ == "__main__":
    # Create blockchain with test parameters
    pow_consensus = ProofOfWork(
        initial_difficulty=4,
        target_block_time=10,  # 10 seconds for testing
        difficulty_adjustment_interval=5,
        hash_power_per_watt=1e6  # 1 MH/s per watt for testing
    )
    
    blockchain = Blockchain(pow_consensus)
    
    # Mine some blocks
    for i in range(10):
        print(f"\nMining block {i+1}...")
        block, stats = blockchain.add_block(f"Block {i+1} Data", track_mining=(i==0))
        
        print(f"Block mined with {stats['hashes_tried']} hashes")
        print(f"Estimated energy used: {stats['energy_kwh']:.6f} kWh")
        print(f"Block hash: {block.hash}")
        
        # Show mining stats
        mining_stats = pow_consensus.get_mining_stats(blockchain.chain)
        print("\nMining Statistics:")
        print(f"Difficulty: {mining_stats['difficulty']}")
        print(f"Network hashrate: {mining_stats['network_hashrate']:.2e} H/s")
        print(f"Daily energy: {mining_stats['daily_energy_kwh']:.2f} kWh")
    
    # Validate chain
    print(f"\nBlockchain valid: {blockchain.is_chain_valid()}")
    
    # Export mining steps from first block for visualization
    with open("mining_steps.json", "w") as f:
        f.write(pow_consensus.export_mining_steps())
    
    # Export blockchain
    with open("blockchain.json", "w") as f:
        f.write(blockchain.export_json())