"""
Tests for visualization components.
"""

import pytest
import plotly.graph_objects as go
from crypto_algorithms.blockchain.block import Block, Transaction
from crypto_algorithms.blockchain.chain import Blockchain
from visualizations.blockchain_flow_chart import BlockchainVisualizer
from visualizations.hash_function_animator import HashVisualizer
from visualizations.merkle_tree_viewer import MerkleTreeVisualizer

def test_blockchain_visualizer():
    """Test blockchain visualization."""
    # Create blockchain with some data
    chain = Blockchain()
    for i in range(3):
        tx = Transaction("A", "B", 1.0, 0.1, i)
        chain.add_transaction(tx)
        chain.mine_block("miner")
    
    # Create visualizer
    viz = BlockchainVisualizer()
    viz.chain = chain
    
    # Test chain visualization
    chain_fig = viz.visualize_chain()
    assert isinstance(chain_fig, go.Figure)
    assert len(chain_fig.data) > 0
    
    # Test block creation visualization
    block = chain.get_latest_block()
    block_fig = viz.visualize_block_creation(block)
    assert isinstance(block_fig, go.Figure)
    assert len(block_fig.data) > 0
    
    # Test transaction flow visualization
    tx = block.transactions[0]
    tx_fig = viz.visualize_transaction_flow(tx)
    assert isinstance(tx_fig, go.Figure)
    assert len(tx_fig.data) > 0
    
    # Test network stats visualization
    stats_fig = viz.visualize_network_stats()
    assert isinstance(stats_fig, go.Figure)
    assert len(stats_fig.data) > 0

def test_hash_visualizer():
    """Test hash function visualization."""
    viz = HashVisualizer()
    
    # Test avalanche visualization
    avalanche_fig = viz.visualize_avalanche("test message")
    assert isinstance(avalanche_fig, go.Figure)
    assert len(avalanche_fig.data) > 0
    
    # Test collision search visualization
    collision_fig = viz.visualize_collision_search(prefix_bits=8)
    assert isinstance(collision_fig, go.Figure)
    assert len(collision_fig.data) > 0
    
    # Test hash internals visualization
    internals_fig = viz.visualize_hash_internals("test message")
    assert isinstance(internals_fig, go.Figure)
    assert len(internals_fig.data) > 0

def test_merkle_tree_visualizer():
    """Test Merkle tree visualization."""
    # Create some transactions
    transactions = [
        Transaction("A", "B", 1.0, 0.1, 1),
        Transaction("B", "C", 0.5, 0.1, 1),
        Transaction("C", "D", 0.25, 0.1, 1)
    ]
    
    viz = MerkleTreeVisualizer()
    
    # Test tree visualization
    tree_fig = viz.visualize_tree(transactions)
    assert isinstance(tree_fig, go.Figure)
    assert len(tree_fig.data) > 0
    
    # Test proof verification visualization
    proof_fig = viz.visualize_proof_verification(transactions, 1)
    assert isinstance(proof_fig, go.Figure)
    assert len(proof_fig.data) > 0
    
    # Test tree construction visualization
    construction_fig = viz.visualize_tree_construction(transactions)
    assert isinstance(construction_fig, go.Figure)
    assert len(construction_fig.data) > 0
    assert hasattr(construction_fig, 'frames')

def test_visualization_parameters():
    """Test visualization parameter handling."""
    viz_blockchain = BlockchainVisualizer()
    viz_hash = HashVisualizer()
    viz_merkle = MerkleTreeVisualizer()
    
    # Test custom dimensions
    width, height = 1000, 800
    
    # Blockchain visualizations
    chain_fig = viz_blockchain.visualize_chain(width=width, height=height)
    assert chain_fig.layout.width == width
    assert chain_fig.layout.height == height
    
    # Hash visualizations
    avalanche_fig = viz_hash.visualize_avalanche(
        "test",
        width=width,
        height=height
    )
    assert avalanche_fig.layout.width == width
    assert avalanche_fig.layout.height == height
    
    # Merkle tree visualizations
    transactions = [Transaction("A", "B", 1.0, 0.1, 1)]
    tree_fig = viz_merkle.visualize_tree(
        transactions,
        width=width,
        height=height
    )
    assert tree_fig.layout.width == width
    assert tree_fig.layout.height == height

def test_visualization_interactivity():
    """Test interactive elements of visualizations."""
    # Create test data
    chain = Blockchain()
    tx = Transaction("A", "B", 1.0, 0.1, 1)
    chain.add_transaction(tx)
    block = chain.mine_block("miner")
    
    # Test blockchain visualization
    viz_blockchain = BlockchainVisualizer()
    viz_blockchain.chain = chain
    chain_fig = viz_blockchain.visualize_chain()
    
    # Check for hover text
    for trace in chain_fig.data:
        if trace.hovertext is not None:
            assert len(trace.hovertext) > 0
    
    # Test hash visualization
    viz_hash = HashVisualizer()
    avalanche_fig = viz_hash.visualize_avalanche("test")
    
    # Check for animation frames
    assert avalanche_fig.layout.updatemenus is not None
    
    # Test Merkle tree visualization
    viz_merkle = MerkleTreeVisualizer()
    transactions = [Transaction("A", "B", 1.0, 0.1, 1)]
    tree_fig = viz_merkle.visualize_tree(transactions)
    
    # Check for node information
    for trace in tree_fig.data:
        if trace.text is not None:
            assert len(trace.text) > 0

if __name__ == "__main__":
    pytest.main([__file__])