"""
Tests for blockchain components.
"""

import pytest
import time
from crypto_algorithms.blockchain.block import Block, Transaction, MerkleTree
from crypto_algorithms.blockchain.chain import Blockchain
from crypto_algorithms.blockchain.consensus.pow import ProofOfWork
from crypto_algorithms.blockchain.consensus.pos import ProofOfStake

def test_transaction():
    """Test transaction functionality."""
    tx = Transaction(
        from_address="sender",
        to_address="recipient",
        amount=1.0,
        fee=0.1,
        nonce=1
    )
    
    # Test hash calculation
    tx_hash = tx.get_hash()
    assert len(tx_hash) == 64  # 32 bytes in hex
    
    # Test serialization
    tx_dict = tx.to_dict()
    tx2 = Transaction.from_dict(tx_dict)
    assert tx2.get_hash() == tx_hash

def test_merkle_tree():
    """Test Merkle tree implementation."""
    transactions = [
        Transaction("A", "B", 1.0, 0.1, 1),
        Transaction("B", "C", 0.5, 0.1, 1),
        Transaction("C", "D", 0.25, 0.1, 1)
    ]
    
    # Build tree
    tree = MerkleTree.build(transactions)
    assert len(tree) > 1  # Should have multiple levels
    
    # Get root
    root = MerkleTree.get_root(transactions)
    assert len(root) == 64  # 32 bytes in hex
    
    # Get and verify proof
    proof = MerkleTree.get_proof(transactions, 1)
    assert len(proof) > 0
    
    tx_hash = transactions[1].get_hash()
    assert MerkleTree.verify_proof(tx_hash, proof, root)
    
    # Test with invalid proof
    assert not MerkleTree.verify_proof(tx_hash, proof[:-1], root)

def test_block():
    """Test block functionality."""
    block = Block(version=1)
    
    # Add transactions
    tx = Transaction("A", "B", 1.0, 0.1, 1)
    block.add_transaction(tx)
    
    # Test Merkle root update
    assert block.header.merkle_root == MerkleTree.get_root([tx])
    
    # Test mining
    initial_hash = block.get_hash()
    block.mine(4)  # Require 4 leading zeros
    assert block.get_hash().startswith('0000')
    assert block.get_hash() != initial_hash
    
    # Test verification
    assert block.verify()
    
    # Test serialization
    block_dict = block.to_dict()
    block2 = Block.from_dict(block_dict)
    assert block2.get_hash() == block.get_hash()

def test_blockchain():
    """Test blockchain functionality."""
    chain = Blockchain()
    
    # Test genesis block
    assert len(chain.main_chain) == 1
    
    # Add transactions
    tx = Transaction("A", "B", 1.0, 0.1, 1)
    assert chain.add_transaction(tx)
    
    # Mine block
    block = chain.mine_block("miner")
    assert block is not None
    assert len(chain.main_chain) == 2
    
    # Test fork handling
    fork_block = Block(version=1, previous_hash=chain.main_chain[0])
    fork_block.add_transaction(Transaction("C", "D", 2.0, 0.2, 1))
    fork_block.mine(chain.difficulty)
    
    assert chain.add_block(fork_block)
    assert chain.get_block_height(fork_block.get_hash()) == 1
    
    # Test chain stats
    stats = chain.get_chain_stats()
    assert stats['height'] == len(chain.main_chain) - 1
    assert stats['fork_blocks'] > 0

def test_pow_consensus():
    """Test Proof of Work consensus."""
    pow_consensus = ProofOfWork(
        initial_difficulty=4,
        target_block_time=10,
        difficulty_adjustment_interval=10
    )
    
    # Create block
    block = Block(version=1)
    block.add_transaction(Transaction("A", "B", 1.0, 0.1, 1))
    
    # Mine block
    success, hashes, energy = pow_consensus.mine_block(block)
    assert success
    assert hashes > 0
    assert energy > 0
    
    # Verify PoW
    assert pow_consensus.check_pow(block.header)
    
    # Test difficulty adjustment
    pow_consensus.adjust_difficulty()
    assert pow_consensus.difficulty > 0

def test_pos_consensus():
    """Test Proof of Stake consensus."""
    pos = ProofOfStake(
        minimum_stake=100.0,
        block_time=12,
        epoch_length=10
    )
    
    # Register validators
    assert pos.register_validator("validator1", 100.0)
    assert pos.register_validator("validator2", 200.0)
    
    # Select proposer
    proposer = pos.select_proposer(123)  # Use seed 123
    assert proposer in ["validator1", "validator2"]
    
    # Create and validate block
    block = Block(version=1)
    block.add_transaction(Transaction("A", "B", 1.0, 0.1, 1))
    
    assert pos.validate_block(block, proposer)
    
    # Test slashing
    assert pos.slash_validator("validator1", "offline")
    stats = pos.get_validator_stats("validator1")
    assert stats['status'] == 'jailed'
    
    # Test network stats
    network_stats = pos.get_network_stats()
    assert network_stats['total_validators'] == 2
    assert network_stats['active_validators'] == 1

if __name__ == "__main__":
    pytest.main([__file__])