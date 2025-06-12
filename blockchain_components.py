"""
Simplified blockchain components for the Streamlit app.
"""

import hashlib
import time
import random
from datetime import datetime

class PosAccount:
    def __init__(self, address, balance=0.0, stake=0.0):
        self.address = address
        self.balance = balance
        self.stake = stake
        self.rewards = 0.0
    
    def get_total_stake(self):
        return self.stake
    
    def add_stake(self, amount):
        if amount <= 0 or amount > self.balance:
            return False
        self.balance -= amount
        self.stake += amount
        return True
    
    def add_reward(self, amount):
        self.rewards += amount
        self.balance += amount
    
    def to_dict(self):
        return {
            "address": self.address,
            "balance": self.balance,
            "stake": self.stake,
            "rewards": self.rewards
        }

class PosBlock:
    def __init__(self, index, previous_hash, validator, timestamp=None, data=""):
        self.index = index
        self.previous_hash = previous_hash
        self.validator = validator
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.data = data
        self.hash = self.calculate_hash()
    
    def calculate_hash(self):
        block_string = f"{self.index}{self.previous_hash}{self.validator}{self.timestamp}{self.data}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self):
        return {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "validator": self.validator,
            "timestamp": datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            "data": self.data,
            "hash": self.hash
        }

class PowBlock:
    def __init__(self, index, previous_hash, timestamp=None, data="", nonce=0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.data = data
        self.nonce = nonce
        self.hash = self.calculate_hash()
    
    def calculate_hash(self):
        block_string = f"{self.index}{self.previous_hash}{self.timestamp}{self.data}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self):
        return {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "timestamp": datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            "data": self.data,
            "nonce": self.nonce,
            "hash": self.hash
        }

def mine_pow_block(previous_block, data, difficulty=3):
    """Mine a new PoW block with the given difficulty."""
    new_block = PowBlock(
        index=previous_block.index + 1,
        previous_hash=previous_block.hash,
        data=data
    )
    
    target = '0' * difficulty
    max_nonce = 100000
    
    for nonce in range(max_nonce):
        new_block.nonce = nonce
        new_block.hash = new_block.calculate_hash()
        
        if new_block.hash.startswith(target):
            return new_block, nonce
    
    return None, max_nonce

def create_pos_block(previous_block, validator, data):
    """Create a new PoS block."""
    new_block = PosBlock(
        index=previous_block.index + 1,
        previous_hash=previous_block.hash,
        validator=validator,
        data=data
    )
    
    return new_block

def select_validator(accounts):
    """Select a validator based on stake."""
    validators = list(accounts.keys())
    weights = [accounts[v].stake for v in validators]
    
    if sum(weights) > 0:
        # Normalize weights
        weights = [w/sum(weights) for w in weights]
        
        # Select validator
        selected_validator = random.choices(validators, weights=weights, k=1)[0]
        return selected_validator
    
    return None

# Initialize demo data
def initialize_blockchain_demo():
    """Initialize blockchain demo data."""
    pow_blocks = [PowBlock(0, "0"*64, data="Genesis Block")]
    pos_blocks = [PosBlock(0, "0"*64, validator="Genesis", data="Genesis Block")]
    
    pos_accounts = {
        "Alice": PosAccount("Alice", 1000, 100),
        "Bob": PosAccount("Bob", 800, 200),
        "Charlie": PosAccount("Charlie", 1200, 300)
    }
    
    # Add some pre-mined blocks for demonstration
    for i in range(1, 4):
        # Add PoW blocks
        pow_blocks.append(PowBlock(
            index=i,
            previous_hash=pow_blocks[i-1].hash,
            data=f"Pre-mined PoW Block {i}",
            nonce=random.randint(1000, 9999)
        ))
        
        # Add PoS blocks
        validator = random.choice(list(pos_accounts.keys()))
        pos_blocks.append(PosBlock(
            index=i,
            previous_hash=pos_blocks[i-1].hash,
            validator=validator,
            data=f"Pre-mined PoS Block {i}"
        ))
        
        # Add rewards to validators
        pos_accounts[validator].add_reward(1.0)
    
    return pow_blocks, pos_blocks, pos_accounts