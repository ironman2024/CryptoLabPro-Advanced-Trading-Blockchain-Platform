# Cryptocurrency Protocol Educational Repository

A comprehensive educational platform for understanding cryptocurrency and blockchain technology through interactive implementations and visualizations.

## 🌟 Features

### 🔐 Cryptographic Primitives
- SHA-256 and Keccak-256 implementations with visualization
- ECDSA and Schnorr signature schemes with step-by-step explanations
- Merkle tree construction and verification with interactive proofs
- Educational demonstrations of cryptographic principles

### ⛓️ Blockchain Components
- Complete block and chain implementation
- Proof of Work (PoW) consensus with difficulty adjustment
- Proof of Stake (PoS) consensus with delegation and slashing
- Transaction pool management
- Fork resolution and chain reorganization

### 📊 Interactive Simulations
- Mining process simulation with multiple miners
- Transaction flow visualization
- Block propagation across network nodes
- Consensus mechanism comparisons

### 🖥️ Web Visualizations
- React-based Merkle tree viewer
- Mining puzzle difficulty visualization
- Blockchain explorer components
- Network topology viewer

### 📓 Educational Notebooks
- Interactive Jupyter notebooks for hands-on learning
- Step-by-step algorithm walkthroughs
- Performance and security analysis
- Real-world applications and case studies

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 14+ (for web visualizations)
- Jupyter Notebook (for interactive notebooks)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/crypto-edu.git
cd crypto-edu

# Install Python dependencies
pip install -r requirements.txt

# Install web dependencies (optional)
cd web
npm install
```

### Quick Start

```python
# Example: Create and visualize a blockchain
from crypto_edu.blockchain.consensus.pow import Blockchain, ProofOfWork
from crypto_edu.visualizations.merkle_tree_viewer import MerkleTreeVisualizer

# Initialize components
pow_consensus = ProofOfWork(initial_difficulty=4)
blockchain = Blockchain(pow_consensus)

# Mine some blocks
for i in range(5):
    block, stats = blockchain.add_block(f"Block {i+1} Data")
    print(f"Block mined with {stats['hashes_tried']} hashes")
    print(f"Block hash: {block.hash}")

# Visualize Merkle tree
transactions = [...]  # Your transactions
viz = MerkleTreeVisualizer()
tree_fig = viz.visualize_tree(transactions)
tree_fig.show()
```

## 📚 Repository Structure

```
/crypto-edu/
  ├── /algorithms/              # Core cryptographic algorithms
  │   ├── /hash/                # Hash functions (SHA-256, Keccak, etc.)
  │   ├── /signatures/          # Digital signatures (ECDSA, Schnorr, etc.)
  │   ├── /symmetric/           # Symmetric encryption
  │   └── /asymmetric/          # Asymmetric encryption
  │
  ├── /blockchain/              # Blockchain components
  │   ├── /core/                # Block, chain, transaction structures
  │   ├── /consensus/           # Consensus algorithms (PoW, PoS)
  │   ├── /network/             # P2P networking simulation
  │   └── /wallets/             # Wallet implementations
  │
  ├── /simulations/             # Interactive simulations
  │   ├── /mining/              # Mining process simulation
  │   ├── /transactions/        # Transaction flow simulation
  │   ├── /consensus/           # Consensus mechanism simulation
  │   └── /attacks/             # Attack vector simulations
  │
  ├── /visualizations/          # Data visualizations
  │   ├── /components/          # Reusable visualization components
  │   ├── /merkle/              # Merkle tree visualizations
  │   ├── /blocks/              # Block and chain visualizations
  │   └── /network/             # Network visualizations
  │
  ├── /notebooks/               # Jupyter notebooks for education
  │   ├── /basics/              # Cryptography basics
  │   ├── /blockchain/          # Blockchain mechanics
  │   ├── /consensus/           # Consensus deep dives
  │   └── /applications/        # Real-world applications
  │
  ├── /docs/                    # Documentation
  │   ├── /whitepaper/          # Whitepaper summary
  │   ├── /glossary/            # Term glossary
  │   ├── /tutorials/           # Step-by-step tutorials
  │   └── /api/                 # API documentation
  │
  ├── /web/                     # Web-based visualizations
  │   ├── /react-components/    # React components
  │   ├── /dash-apps/           # Dash applications
  │   └── /static/              # Static assets
  │
  └── /tests/                   # Test suite
```

## 🎓 Educational Pathways

### Beginner Path
1. Start with the cryptography basics notebooks
2. Explore the hash function implementations
3. Try the interactive mining simulation
4. Learn about blockchain structure and blocks

### Intermediate Path
1. Study the consensus mechanisms (PoW and PoS)
2. Explore digital signatures and their applications
3. Understand Merkle trees and their proofs
4. Experiment with transaction flow simulations

### Advanced Path
1. Analyze security considerations and attack vectors
2. Study the network propagation simulations
3. Explore advanced cryptographic techniques
4. Build your own blockchain application

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution
- Additional cryptographic primitives
- New visualization components
- Performance improvements
- Documentation and tutorials
- Test coverage

## 📖 Documentation

- [API Reference](docs/api/README.md)
- [Tutorials](docs/tutorials/README.md)
- [Examples](docs/examples/README.md)
- [Security Considerations](docs/security.md)

## 🔒 Security Notes

This repository is for educational purposes. For production use:
- Use established cryptographic libraries
- Follow security best practices
- Keep private keys secure
- Stay updated on vulnerabilities

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Bitcoin and Ethereum developers
- Cryptography researchers
- Open source contributors
- Educational resources