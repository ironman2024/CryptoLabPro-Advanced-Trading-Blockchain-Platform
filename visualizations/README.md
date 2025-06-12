# Blockchain Visualizations

This directory contains interactive visualization tools for understanding blockchain concepts and cryptographic primitives.

## Components

### Blockchain Flow Chart
- Chain structure visualization
- Block creation process
- Transaction flow
- Network statistics

### Hash Function Animator
- Avalanche effect demonstration
- Collision search visualization
- Internal state animation
- Bit distribution analysis

### Merkle Tree Viewer
- Tree structure visualization
- Proof verification animation
- Tree construction process
- Interactive node exploration

## Usage Examples

### Visualizing Blockchain Structure
```python
from visualizations.blockchain_flow_chart import BlockchainVisualizer

viz = BlockchainVisualizer()

# Visualize chain
chain_fig = viz.visualize_chain()
chain_fig.show()

# Visualize block creation
block_fig = viz.visualize_block_creation(block)
block_fig.show()

# Visualize transaction flow
tx_fig = viz.visualize_transaction_flow(transaction)
tx_fig.show()
```

### Demonstrating Hash Functions
```python
from visualizations.hash_function_animator import HashVisualizer

viz = HashVisualizer()

# Show avalanche effect
avalanche_fig = viz.visualize_avalanche("Hello, World!")
avalanche_fig.show()

# Show collision search
collision_fig = viz.visualize_collision_search(prefix_bits=16)
collision_fig.show()

# Show hash internals
internals_fig = viz.visualize_hash_internals("Hello, World!")
internals_fig.show()
```

### Exploring Merkle Trees
```python
from visualizations.merkle_tree_viewer import MerkleTreeVisualizer

viz = MerkleTreeVisualizer()

# Visualize tree
tree_fig = viz.visualize_tree(transactions)
tree_fig.show()

# Show proof verification
proof_fig = viz.visualize_proof_verification(transactions, tx_index=1)
proof_fig.show()

# Animate tree construction
construction_fig = viz.visualize_tree_construction(transactions)
construction_fig.show()
```

## Educational Features

### Interactive Elements
- Zoom and pan support
- Hover information
- Animation controls
- Real-time updates

### Color Coding
- Block states and types
- Transaction status
- Hash distributions
- Proof verification paths

### Annotations
- Process explanations
- Technical details
- Performance metrics
- Security considerations

## Technical Details

### Dependencies
- Plotly for interactive plots
- NetworkX for graph layouts
- NumPy for calculations
- Core blockchain components

### Performance
Typical render times:
- Chain visualization: ~100ms
- Hash animations: ~200ms
- Merkle trees: ~150ms

### Browser Support
- Chrome/Firefox/Safari
- WebGL acceleration
- Mobile responsive

## Usage Notes

For best performance:
- Limit chain size to ~100 blocks
- Use compressed transaction lists
- Enable WebGL in browser
- Consider screen resolution

## References

- [Plotly Documentation](https://plotly.com/python/)
- [NetworkX Documentation](https://networkx.org/)
- [Bitcoin Block Explorer](https://blockchain.info/)
- [Ethereum Block Explorer](https://etherscan.io/)