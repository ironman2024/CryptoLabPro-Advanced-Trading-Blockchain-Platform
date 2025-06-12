"""
Proof of Stake (PoS) Consensus Implementation

This module provides a detailed implementation of Proof of Stake consensus
with delegation, slashing, and reward distribution mechanisms.
"""

from typing import Dict, List, Tuple, Any, Optional, Set
import hashlib
import random
import time
import json
from datetime import datetime

class Account:
    """Account in a PoS system with staking capabilities."""
    
    def __init__(self, address: str, balance: float = 0.0, stake: float = 0.0):
        """
        Initialize an account.
        
        Args:
            address: Account address
            balance: Initial balance
            stake: Initial staked amount
        """
        self.address = address
        self.balance = balance
        self.stake = stake
        self.delegated_to = None  # Address this account delegates to
        self.delegated_from = {}  # Address -> amount delegated from others
        self.last_stake_time = 0  # Time of last stake
        self.unstaking = []  # List of (amount, unlock_time) for unstaking funds
        self.slashed = 0.0  # Amount slashed for misbehavior
        self.rewards = 0.0  # Accumulated rewards
    
    def get_total_stake(self) -> float:
        """
        Get total stake including delegations.
        
        Returns:
            Total stake amount
        """
        delegated_total = sum(self.delegated_from.values())
        return self.stake + delegated_total
    
    def add_stake(self, amount: float) -> bool:
        """
        Add stake from balance.
        
        Args:
            amount: Amount to stake
            
        Returns:
            True if successful
        """
        if amount <= 0 or amount > self.balance:
            return False
        
        self.balance -= amount
        self.stake += amount
        self.last_stake_time = time.time()
        return True
    
    def remove_stake(self, amount: float, lock_period: int = 86400) -> bool:
        """
        Remove stake (with lock period).
        
        Args:
            amount: Amount to unstake
            lock_period: Time in seconds before funds are available
            
        Returns:
            True if successful
        """
        if amount <= 0 or amount > self.stake:
            return False
        
        self.stake -= amount
        unlock_time = time.time() + lock_period
        self.unstaking.append((amount, unlock_time))
        return True
    
    def process_unstaking(self) -> float:
        """
        Process unstaking and return unlocked funds.
        
        Returns:
            Amount unlocked
        """
        current_time = time.time()
        unlocked = 0.0
        remaining = []
        
        for amount, unlock_time in self.unstaking:
            if current_time >= unlock_time:
                unlocked += amount
                self.balance += amount
            else:
                remaining.append((amount, unlock_time))
        
        self.unstaking = remaining
        return unlocked
    
    def delegate(self, to_address: str, amount: float) -> bool:
        """
        Delegate stake to another validator.
        
        Args:
            to_address: Address to delegate to
            amount: Amount to delegate
            
        Returns:
            True if successful
        """
        if amount <= 0 or amount > self.stake or self.delegated_to is not None:
            return False
        
        self.stake -= amount
        self.delegated_to = to_address
        return True
    
    def receive_delegation(self, from_address: str, amount: float) -> bool:
        """
        Receive delegation from another account.
        
        Args:
            from_address: Address delegating
            amount: Amount delegated
            
        Returns:
            True if successful
        """
        if amount <= 0:
            return False
        
        if from_address in self.delegated_from:
            self.delegated_from[from_address] += amount
        else:
            self.delegated_from[from_address] = amount
        
        return True
    
    def slash(self, amount: float) -> float:
        """
        Slash stake for misbehavior.
        
        Args:
            amount: Amount to slash
            
        Returns:
            Actual amount slashed
        """
        slashable = min(amount, self.stake)
        self.stake -= slashable
        self.slashed += slashable
        
        # Also slash delegated stakes proportionally
        delegated_total = sum(self.delegated_from.values())
        if delegated_total > 0:
            slash_ratio = min(1.0, amount / (self.stake + delegated_total))
            for addr in self.delegated_from:
                slash_amount = self.delegated_from[addr] * slash_ratio
                self.delegated_from[addr] -= slash_amount
                slashable += slash_amount
        
        return slashable
    
    def add_reward(self, amount: float) -> None:
        """
        Add staking reward.
        
        Args:
            amount: Reward amount
        """
        self.rewards += amount
        self.balance += amount
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert account to dictionary.
        
        Returns:
            Account as dictionary
        """
        return {
            "address": self.address,
            "balance": self.balance,
            "stake": self.stake,
            "delegated_to": self.delegated_to,
            "delegated_from": self.delegated_from,
            "total_stake": self.get_total_stake(),
            "last_stake_time": self.last_stake_time,
            "unstaking": self.unstaking,
            "slashed": self.slashed,
            "rewards": self.rewards
        }

class Block:
    """Block in a PoS blockchain."""
    
    def __init__(self, index: int, previous_hash: str, validator: str, 
                 timestamp: float = None, data: str = ""):
        """
        Initialize a block.
        
        Args:
            index: Block height
            previous_hash: Hash of the previous block
            validator: Address of the validator who created this block
            timestamp: Block creation time (defaults to current time)
            data: Block data
        """
        self.index = index
        self.previous_hash = previous_hash
        self.validator = validator
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.data = data
        self.hash = self.calculate_hash()
        self.signature = ""  # To be signed by validator
    
    def calculate_hash(self) -> str:
        """Calculate block hash."""
        block_string = f"{self.index}{self.previous_hash}{self.validator}{self.timestamp}{self.data}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def sign(self, signature: str) -> None:
        """
        Sign the block.
        
        Args:
            signature: Validator's signature
        """
        self.signature = signature
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert block to dictionary.
        
        Returns:
            Block as dictionary
        """
        return {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "validator": self.validator,
            "timestamp": self.timestamp,
            "data": self.data,
            "hash": self.hash,
            "signature": self.signature
        }
    
    def __str__(self) -> str:
        """String representation of block."""
        return (f"Block #{self.index}\n"
                f"Hash: {self.hash}\n"
                f"Previous: {self.previous_hash}\n"
                f"Validator: {self.validator}\n"
                f"Time: {datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Data: {self.data[:50]}{'...' if len(self.data) > 50 else ''}")

class ProofOfStake:
    """
    Proof of Stake consensus implementation with delegation and slashing.
    """
    
    def __init__(self, 
                 minimum_stake: float = 1.0,
                 block_time: int = 5,  # seconds
                 reward_rate: float = 0.05,  # 5% annual
                 slash_rate: float = 0.1):  # 10% for violations
        """
        Initialize PoS consensus.
        
        Args:
            minimum_stake: Minimum stake to be a validator
            block_time: Target time between blocks in seconds
            reward_rate: Annual reward rate for staking
            slash_rate: Percentage of stake to slash for violations
        """
        self.minimum_stake = minimum_stake
        self.block_time = block_time
        self.reward_rate = reward_rate
        self.slash_rate = slash_rate
        
        self.accounts = {}  # address -> Account
        self.validators = set()  # Set of validator addresses
        
        # For educational purposes
        self.validator_selection_steps = []
        self.reward_distribution_steps = []]
        self.reward_distribution_steps = []
    
    def create_account(self, address: str, initial_balance: float = 0.0) -> Account:
        """
        Create a new account.
        
        Args:
            address: Account address
            initial_balance: Initial balance
            
        Returns:
            New account
        """
        account = Account(address, initial_balance)
        self.accounts[address] = account
        return account
    
    def add_stake(self, address: str, amount: float) -> bool:
        """
        Add stake for an account.
        
        Args:
            address: Account address
            amount: Amount to stake
            
        Returns:
            True if successful
        """
        if address not in self.accounts:
            return False
        
        success = self.accounts[address].add_stake(amount)
        if success and self.accounts[address].get_total_stake() >= self.minimum_stake:
            self.validators.add(address)
        
        return success
    
    def remove_stake(self, address: str, amount: float) -> bool:
        """
        Remove stake for an account.
        
        Args:
            address: Account address
            amount: Amount to unstake
            
        Returns:
            True if successful
        """
        if address not in self.accounts:
            return False
        
        success = self.accounts[address].remove_stake(amount)
        if success and self.accounts[address].get_total_stake() < self.minimum_stake:
            self.validators.discard(address)
        
        return success
    
    def delegate_stake(self, from_address: str, to_address: str, amount: float) -> bool:
        """
        Delegate stake from one account to another.
        
        Args:
            from_address: Delegator address
            to_address: Validator address
            amount: Amount to delegate
            
        Returns:
            True if successful
        """
        if from_address not in self.accounts or to_address not in self.accounts:
            return False
        
        delegator = self.accounts[from_address]
        validator = self.accounts[to_address]
        
        if delegator.delegate(to_address, amount):
            if validator.receive_delegation(from_address, amount):
                if validator.get_total_stake() >= self.minimum_stake:
                    self.validators.add(to_address)
                return True
        
        return False
    
    def slash_validator(self, address: str) -> float:
        """
        Slash a validator for misbehavior.
        
        Args:
            address: Validator address
            
        Returns:
            Amount slashed
        """
        if address not in self.accounts:
            return 0.0
        
        validator = self.accounts[address]
        slash_amount = validator.get_total_stake() * self.slash_rate
        slashed = validator.slash(slash_amount)
        
        if validator.get_total_stake() < self.minimum_stake:
            self.validators.discard(address)
        
        return slashed
    
    def select_validator(self, seed: str = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Select a validator for the next block based on stake.
        
        Args:
            seed: Random seed (e.g., previous block hash)
            
        Returns:
            Tuple of (validator address, selection steps)
        """
        if not self.validators:
            return None, []
        
        # Reset steps
        self.validator_selection_steps = []
        
        # Get total stake
        total_stake = sum(self.accounts[v].get_total_stake() for v in self.validators)
        self.validator_selection_steps.append({
            "step": "calculate_total_stake",
            "total_stake": total_stake,
            "validators": len(self.validators)
        })
        
        # Select validator proportional to stake
        if seed:
            random.seed(seed)
        
        r = random.uniform(0, total_stake)
        self.validator_selection_steps.append({
            "step": "random_selection",
            "random_value": r,
            "total_stake": total_stake
        })
        
        cumulative = 0
        selected = None
        
        validator_stakes = []
        for validator in self.validators:
            stake = self.accounts[validator].get_total_stake()
            validator_stakes.append({
                "validator": validator,
                "stake": stake,
                "probability": stake / total_stake if total_stake > 0 else 0
            })
        
        self.validator_selection_steps.append({
            "step": "validator_stakes",
            "validators": validator_stakes
        })
        
        for validator in self.validators:
            stake = self.accounts[validator].get_total_stake()
            cumulative += stake
            if r <= cumulative:
                selected = validator
                self.validator_selection_steps.append({
                    "step": "selected_validator",
                    "validator": validator,
                    "stake": stake,
                    "cumulative": cumulative,
                    "random_value": r
                })
                break
        
        return selected, self.validator_selection_steps
    
    def distribute_rewards(self, validator: str, block_reward: float) -> Dict[str, float]:
        """
        Distribute block rewards to validator and delegators.
        
        Args:
            validator: Validator address
            block_reward: Total block reward
            
        Returns:
            Dictionary of address -> reward amount
        """
        if validator not in self.accounts:
            return {}
        
        # Reset steps
        self.reward_distribution_steps = []
        
        validator_account = self.accounts[validator]
        total_stake = validator_account.get_total_stake()
        
        if total_stake == 0:
            return {}
        
        # Calculate validator's own reward
        validator_stake_ratio = validator_account.stake / total_stake
        validator_reward = block_reward * validator_stake_ratio
        
        self.reward_distribution_steps.append({
            "step": "calculate_validator_reward",
            "validator": validator,
            "validator_stake": validator_account.stake,
            "total_stake": total_stake,
            "stake_ratio": validator_stake_ratio,
            "reward": validator_reward
        })
        
        # Add reward to validator
        validator_account.add_reward(validator_reward)
        
        # Distribute to delegators
        delegator_rewards = {}
        for delegator, amount in validator_account.delegated_from.items():
            if delegator in self.accounts:
                stake_ratio = amount / total_stake
                reward = block_reward * stake_ratio
                
                self.reward_distribution_steps.append({
                    "step": "calculate_delegator_reward",
                    "delegator": delegator,
                    "delegated_stake": amount,
                    "total_stake": total_stake,
                    "stake_ratio": stake_ratio,
                    "reward": reward
                })
                
                self.accounts[delegator].add_reward(reward)
                delegator_rewards[delegator] = reward
        
        # Return all rewards
        all_rewards = {validator: validator_reward}
        all_rewards.update(delegator_rewards)
        
        self.reward_distribution_steps.append({
            "step": "final_rewards",
            "rewards": all_rewards
        })
        
        return all_rewards
    
    def calculate_annual_rewards(self, address: str) -> float:
        """
        Calculate estimated annual rewards for an account.
        
        Args:
            address: Account address
            
        Returns:
            Estimated annual reward
        """
        if address not in self.accounts:
            return 0.0
        
        account = self.accounts[address]
        
        # Direct staking rewards
        direct_reward = account.stake * self.reward_rate
        
        # Rewards from delegations to this validator
        delegated_reward = 0.0
        if address in self.validators:
            delegated_total = sum(account.delegated_from.values())
            delegated_reward = delegated_total * self.reward_rate
        
        return direct_reward + delegated_reward
    
    def get_validator_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all validators.
        
        Returns:
            List of validator statistics
        """
        stats = []
        for validator in self.validators:
            account = self.accounts[validator]
            stats.append({
                "address": validator,
                "own_stake": account.stake,
                "delegated_stake": sum(account.delegated_from.values()),
                "total_stake": account.get_total_stake(),
                "delegators": len(account.delegated_from),
                "annual_reward_rate": self.reward_rate,
                "estimated_annual_reward": self.calculate_annual_rewards(validator)
            })
        
        return stats
    
    def export_selection_steps(self) -> str:
        """
        Export validator selection steps as JSON.
        
        Returns:
            JSON string of selection steps
        """
        return json.dumps(self.validator_selection_steps, indent=2)
    
    def export_reward_steps(self) -> str:
        """
        Export reward distribution steps as JSON.
        
        Returns:
            JSON string of reward steps
        """
        return json.dumps(self.reward_distribution_steps, indent=2)

class PoSBlockchain:
    """Blockchain implementation using Proof of Stake consensus."""
    
    def __init__(self, pos_consensus: ProofOfStake = None):
        """
        Initialize blockchain.
        
        Args:
            pos_consensus: PoS consensus mechanism (creates default if None)
        """
        self.chain = []
        self.pos = pos_consensus if pos_consensus else ProofOfStake()
        
        # Create genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self) -> Block:
        """
        Create the genesis block.
        
        Returns:
            Genesis block
        """
        genesis = Block(0, "0" * 64, "genesis", data="Genesis Block")
        genesis.sign("genesis_signature")  # No real signature for genesis
        self.chain.append(genesis)
        return genesis
    
    def get_latest_block(self) -> Block:
        """
        Get the latest block in the chain.
        
        Returns:
            Latest block
        """
        return self.chain[-1]
    
    def create_block(self, data: str, validator: str) -> Block:
        """
        Create a new block.
        
        Args:
            data: Block data
            validator: Validator address
            
        Returns:
            New block
        """
        latest_block = self.get_latest_block()
        new_block = Block(
            index=latest_block.index + 1,
            previous_hash=latest_block.hash,
            validator=validator,
            data=data
        )
        
        # In a real system, the validator would sign this with their private key
        # Here we just use a simple string for demonstration
        signature = f"signed_by_{validator}_{new_block.hash[:8]}"
        new_block.sign(signature)
        
        return new_block
    
    def add_block(self, block: Block, block_reward: float = 1.0) -> Dict[str, float]:
        """
        Add a block to the chain and distribute rewards.
        
        Args:
            block: Block to add
            block_reward: Reward for this block
            
        Returns:
            Dictionary of rewards distributed
        """
        # Verify block
        if block.index != len(self.chain):
            raise ValueError("Invalid block index")
        
        if block.previous_hash != self.get_latest_block().hash:
            raise ValueError("Invalid previous hash")
        
        # In a real system, we would verify the signature here
        
        # Add block to chain
        self.chain.append(block)
        
        # Distribute rewards
        rewards = self.pos.distribute_rewards(block.validator, block_reward)
        
        return rewards
    
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
            
            # In a real system, we would verify signatures here
        
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
    # Create PoS consensus with test parameters
    pos_consensus = ProofOfStake(
        minimum_stake=10.0,
        block_time=5,
        reward_rate=0.05,
        slash_rate=0.1
    )
    
    # Create accounts
    pos_consensus.create_account("alice", 100.0)
    pos_consensus.create_account("bob", 200.0)
    pos_consensus.create_account("charlie", 50.0)
    pos_consensus.create_account("dave", 30.0)
    
    # Add stakes
    pos_consensus.add_stake("alice", 50.0)
    pos_consensus.add_stake("bob", 100.0)
    pos_consensus.add_stake("charlie", 20.0)
    
    # Delegate
    pos_consensus.delegate_stake("dave", "alice", 20.0)
    
    # Create blockchain
    blockchain = PoSBlockchain(pos_consensus)
    
    # Create and add blocks
    for i in range(5):
        print(f"\nCreating block {i+1}...")
        
        # Select validator
        seed = blockchain.get_latest_block().hash
        validator, steps = pos_consensus.select_validator(seed)
        
        if validator:
            print(f"Selected validator: {validator}")
            
            # Create and add block
            block = blockchain.create_block(f"Block {i+1} Data", validator)
            rewards = blockchain.add_block(block, block_reward=1.0)
            
            print(f"Block created by {validator}")
            print(f"Rewards: {rewards}")
            
            # Show validator stats
            stats = pos_consensus.get_validator_stats()
            print("\nValidator Statistics:")
            for stat in stats:
                print(f"{stat['address']}: {stat['total_stake']:.2f} staked, "
                      f"{stat['estimated_annual_reward']:.2f} annual reward")
    
    # Export selection steps from last block
    with open("validator_selection.json", "w") as f:
        f.write(pos_consensus.export_selection_steps())
    
    # Export reward steps from last block
    with open("reward_distribution.json", "w") as f:
        f.write(pos_consensus.export_reward_steps())
    
    # Export blockchain
    with open("pos_blockchain.json", "w") as f:
        f.write(blockchain.export_json())