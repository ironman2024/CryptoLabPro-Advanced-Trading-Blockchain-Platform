"""
Proof of Stake (PoS) Consensus Implementation

This module provides a detailed implementation of Proof of Stake consensus
with validator selection, rewards, and slashing conditions.
"""

from typing import Dict, List, Optional, Tuple, Set
import time
import random
from dataclasses import dataclass
import math
from ..block import Block, Transaction
from ...hashing.sha256 import SHA256

@dataclass
class Validator:
    """Validator information for PoS."""
    address: str
    stake: float
    active_since: int
    last_proposed: int
    total_proposed: int
    total_rewards: float
    slashed: bool = False
    jailed_until: int = 0

class ProofOfStake:
    """
    Proof of Stake consensus implementation with validator management,
    rewards, and slashing conditions.
    """
    
    def __init__(self,
                 minimum_stake: float = 32.0,  # Minimum stake required (like ETH2)
                 block_time: int = 12,  # Target 12 seconds per block
                 epoch_length: int = 32,  # Blocks per epoch
                 reward_rate: float = 0.06):  # 6% annual return target
        """
        Initialize PoS consensus.
        
        Args:
            minimum_stake: Minimum stake required to become validator
            block_time: Target time between blocks in seconds
            epoch_length: Number of blocks in an epoch
            reward_rate: Annual reward rate for staking
        """
        self.minimum_stake = minimum_stake
        self.block_time = block_time
        self.epoch_length = epoch_length
        self.reward_rate = reward_rate
        
        self.validators: Dict[str, Validator] = {}
        self.active_validators: Set[str] = set()
        self.total_staked = 0.0
        self.current_epoch = 0
        
        # Slashing conditions
        self.double_sign_penalty = 0.5  # 50% of stake
        self.offline_penalty = 0.01  # 1% of stake
        self.min_attestations = 0.8  # Must participate in 80% of attestations
    
    def register_validator(self, address: str, stake: float) -> bool:
        """
        Register a new validator.
        
        Args:
            address: Validator's address
            stake: Amount to stake
            
        Returns:
            True if registration successful
        """
        if stake < self.minimum_stake:
            return False
            
        if address in self.validators:
            return False
            
        self.validators[address] = Validator(
            address=address,
            stake=stake,
            active_since=int(time.time()),
            last_proposed=0,
            total_proposed=0,
            total_rewards=0.0
        )
        
        self.total_staked += stake
        self.active_validators.add(address)
        return True
    
    def increase_stake(self, address: str, amount: float) -> bool:
        """
        Increase a validator's stake.
        
        Args:
            address: Validator's address
            amount: Amount to add to stake
            
        Returns:
            True if stake increased successfully
        """
        if address not in self.validators:
            return False
            
        validator = self.validators[address]
        validator.stake += amount
        self.total_staked += amount
        return True
    
    def decrease_stake(self, address: str, amount: float) -> bool:
        """
        Decrease a validator's stake.
        
        Args:
            address: Validator's address
            amount: Amount to remove from stake
            
        Returns:
            True if stake decreased successfully
        """
        if address not in self.validators:
            return False
            
        validator = self.validators[address]
        if validator.stake - amount < self.minimum_stake:
            return False
            
        validator.stake -= amount
        self.total_staked -= amount
        return True
    
    def select_proposer(self, seed: int) -> Optional[str]:
        """
        Select block proposer using weighted random selection.
        
        Args:
            seed: Random seed (usually previous block hash)
            
        Returns:
            Selected validator's address
        """
        if not self.active_validators:
            return None
            
        # Get active validators and their weights
        weights = []
        addresses = []
        
        for addr in self.active_validators:
            validator = self.validators[addr]
            if not validator.slashed and validator.jailed_until <= time.time():
                weights.append(validator.stake)
                addresses.append(addr)
                
        if not addresses:
            return None
            
        # Use seed for deterministic selection
        random.seed(seed)
        total_weight = sum(weights)
        selection = random.uniform(0, total_weight)
        
        cumulative = 0
        for addr, weight in zip(addresses, weights):
            cumulative += weight
            if selection <= cumulative:
                return addr
                
        return addresses[-1]
    
    def validate_block(self, block: Block, proposer: str) -> bool:
        """
        Validate a proposed block.
        
        Args:
            block: Proposed block
            proposer: Address of proposed validator
            
        Returns:
            True if block is valid
        """
        if proposer not in self.validators:
            return False
            
        validator = self.validators[proposer]
        
        # Check if validator is eligible
        if validator.slashed or validator.jailed_until > time.time():
            return False
            
        # Verify block timestamp
        if block.header.timestamp <= validator.last_proposed:
            return False
            
        # Basic block validation (would do more in practice)
        if not block.verify():
            return False
            
        return True
    
    def finalize_block(self, block: Block, proposer: str) -> None:
        """
        Finalize a block and distribute rewards.
        
        Args:
            block: Finalized block
            proposer: Address of proposer
        """
        validator = self.validators[proposer]
        
        # Update validator stats
        validator.last_proposed = block.header.timestamp
        validator.total_proposed += 1
        
        # Calculate and distribute rewards
        base_reward = self.calculate_base_reward(validator.stake)
        inclusion_reward = self.calculate_inclusion_reward(len(block.transactions))
        total_reward = base_reward + inclusion_reward
        
        validator.total_rewards += total_reward
        
        # Update epoch if needed
        block_number = len(block.transactions)  # Simplified
        if block_number % self.epoch_length == 0:
            self.current_epoch += 1
            self.process_epoch_transition()
    
    def calculate_base_reward(self, stake: float) -> float:
        """
        Calculate base reward for block proposal.
        
        Args:
            stake: Validator's stake
            
        Returns:
            Base reward amount
        """
        # Daily rate from annual rate
        daily_rate = self.reward_rate / 365
        
        # Blocks per day
        blocks_per_day = 86400 // self.block_time
        
        # Reward per block
        return (stake * daily_rate) / blocks_per_day
    
    def calculate_inclusion_reward(self, num_transactions: int) -> float:
        """
        Calculate additional reward based on included transactions.
        
        Args:
            num_transactions: Number of transactions in block
            
        Returns:
            Additional reward amount
        """
        # Simple linear reward based on transactions
        return 0.0001 * num_transactions
    
    def slash_validator(self, address: str, reason: str) -> bool:
        """
        Slash a validator for misbehavior.
        
        Args:
            address: Validator's address
            reason: Reason for slashing
            
        Returns:
            True if validator was slashed
        """
        if address not in self.validators:
            return False
            
        validator = self.validators[address]
        
        if reason == "double_sign":
            penalty = validator.stake * self.double_sign_penalty
            validator.slashed = True
        elif reason == "offline":
            penalty = validator.stake * self.offline_penalty
            validator.jailed_until = int(time.time()) + 86400  # Jail for 24 hours
        else:
            return False
            
        validator.stake -= penalty
        self.total_staked -= penalty
        
        if validator.stake < self.minimum_stake:
            self.active_validators.remove(address)
            
        return True
    
    def process_epoch_transition(self) -> None:
        """Process validator set changes at epoch transition."""
        current_time = int(time.time())
        
        # Check for inactive validators
        for addr in list(self.active_validators):
            validator = self.validators[addr]
            
            # Remove if jailed or slashed
            if validator.slashed or validator.jailed_until > current_time:
                self.active_validators.remove(addr)
                continue
                
            # Check minimum attestations (simplified)
            if validator.total_proposed > 0:
                participation_rate = (
                    validator.total_proposed /
                    (self.current_epoch * self.epoch_length)
                )
                if participation_rate < self.min_attestations:
                    self.slash_validator(addr, "offline")
    
    def get_validator_stats(self, address: str) -> Optional[dict]:
        """
        Get statistics for a validator.
        
        Args:
            address: Validator's address
            
        Returns:
            Dictionary of validator statistics
        """
        if address not in self.validators:
            return None
            
        validator = self.validators[address]
        current_time = int(time.time())
        
        return {
            'address': validator.address,
            'stake': validator.stake,
            'active_time': current_time - validator.active_since,
            'blocks_proposed': validator.total_proposed,
            'total_rewards': validator.total_rewards,
            'status': 'slashed' if validator.slashed else (
                'jailed' if validator.jailed_until > current_time else 'active'
            ),
            'effective_balance': validator.stake + validator.total_rewards,
            'annual_return': (
                validator.total_rewards * 365 * 86400 /
                (current_time - validator.active_since) /
                validator.stake
            ) if current_time > validator.active_since else 0.0
        }
    
    def get_network_stats(self) -> dict:
        """Get overall network statistics."""
        current_time = int(time.time())
        active_stake = sum(
            v.stake for v in self.validators.values()
            if not v.slashed and v.jailed_until <= current_time
        )
        
        return {
            'total_validators': len(self.validators),
            'active_validators': len(self.active_validators),
            'total_staked': self.total_staked,
            'active_stake': active_stake,
            'current_epoch': self.current_epoch,
            'participation_rate': len(self.active_validators) / len(self.validators) if self.validators else 0,
            'network_security': active_stake / self.total_staked if self.total_staked > 0 else 0
        }

# Example usage and tests
if __name__ == "__main__":
    # Create PoS consensus
    pos = ProofOfStake(
        minimum_stake=32.0,
        block_time=12,
        epoch_length=32,
        reward_rate=0.06
    )
    
    # Register some validators
    validators = [
        ("validator1", 100.0),
        ("validator2", 50.0),
        ("validator3", 75.0)
    ]
    
    print("Registering validators...")
    for addr, stake in validators:
        if pos.register_validator(addr, stake):
            print(f"Registered {addr} with {stake} stake")
    
    # Simulate some blocks
    print("\nSimulating block production...")
    for i in range(5):
        # Select proposer
        seed = i  # Would use previous block hash in practice
        proposer = pos.select_proposer(seed)
        
        if proposer:
            # Create and validate block
            block = Block(version=1)
            block.add_transaction(Transaction("user1", "user2", 1.0, 0.1, i))
            
            if pos.validate_block(block, proposer):
                pos.finalize_block(block, proposer)
                print(f"Block {i+1} proposed by {proposer}")
    
    # Print validator stats
    print("\nValidator Statistics:")
    for addr, _ in validators:
        stats = pos.get_validator_stats(addr)
        if stats:
            print(f"\n{addr}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    # Print network stats
    print("\nNetwork Statistics:")
    network_stats = pos.get_network_stats()
    for key, value in network_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")