# Blockchain Components

This directory contains core blockchain components with educational implementations of blocks, chains, and consensus mechanisms.

## Components

### Block Structure
- Complete block implementation supporting both Bitcoin-style and Ethereum-style blocks
- Merkle tree implementation for transaction verification
- Transaction pool management
- Block validation and verification

### Blockchain
- Chain management with fork resolution
- Block reorganization capabilities
- UTXO/Account state tracking
- Difficulty adjustment

### Consensus Mechanisms

#### Proof of Work (PoW)
- Complete mining implementation
- Difficulty adjustment algorithm
- Network hashrate estimation
- Energy consumption modeling

#### Proof of Stake (PoS)
- Validator management
- Stake-weighted selection
- Rewards and penalties
- Slashing conditions

## Usage Examples

### Creating and Mining Blocks
```python
from blockchain.block import Block, Transaction
from blockchain.consensus.pow import ProofOfWork

# Create PoW consensus
pow_consensus = ProofOfWork(initial_difficulty=4)

# Create block
block = Block(version=1)
block.add_transaction(Transaction("Alice", "Bob", 1.0, 0.1, 1))

# Mine block
success, hashes, energy = pow_consensus.mine_block(block)
if success:
    print(f"Block mined with {hashes} hashes")
    print(f"Estimated energy: {energy:.6f} kWh")
```

### Running a PoS Validator
```python
from blockchain.consensus.pos import ProofOfStake

# Create PoS consensus
pos = ProofOfStake(minimum_stake=32.0)

# Register validator
pos.register_validator("validator1", 100.0)

# Propose block
proposer = pos.select_proposer(seed=123)
if proposer == "validator1":
    block = Block(version=1)
    if pos.validate_block(block, proposer):
        pos.finalize_block(block, proposer)
```

### Managing the Chain
```python
from blockchain.chain import Blockchain

# Create blockchain
chain = Blockchain()

# Add transactions
chain.add_transaction(Transaction("Alice", "Bob", 1.0, 0.1, 1))

# Mine block
block = chain.mine_block("miner")

# Get chain stats
stats = chain.get_chain_stats()
print(f"Chain height: {stats['height']}")
```

## Educational Features

### Merkle Tree Visualization
The block implementation includes methods to visualize Merkle trees:
```python
from blockchain.block import Block, MerkleTree

block = Block()
# Add transactions...

# Get Merkle proof for transaction
proof = MerkleTree.get_proof(block.transactions, 0)
print("Merkle Proof:", proof)
```

### Mining Simulation
The PoW implementation includes detailed mining statistics:
```python
from blockchain.consensus.pow import ProofOfWork

pow = ProofOfWork()
stats = pow.get_mining_stats(recent_blocks)
print(f"Network hashrate: {stats['network_hashrate']:.2e} H/s")
print(f"Daily energy: {stats['daily_energy_kwh']:.2f} kWh")
```

### Validator Economics
The PoS implementation models validator economics:
```python
from blockchain.consensus.pos import ProofOfStake

pos = ProofOfStake()
# Register validators...

stats = pos.get_validator_stats("validator1")
print(f"Annual return: {stats['annual_return']:.2%}")
```

## Security Considerations

For production use:
- Use established blockchain platforms
- These implementations are for learning
- Real networks need additional security measures
- Proper key management is critical

## Performance Notes

Typical performance on modern hardware:

Block Processing:
- Block creation: ~1000 tx/s
- Merkle tree building: ~10000 tx/s
- Block validation: ~500 tx/s

Consensus:
- PoW mining: Hardware dependent
- PoS block production: ~100 blocks/s
- Chain reorganization: ~1000 blocks/s

## References

- [Bitcoin Whitepaper](https://bitcoin.org/bitcoin.pdf)
- [Ethereum Yellow Paper](https://ethereum.github.io/yellowpaper/paper.pdf)
- [PoS in Ethereum 2.0](https://ethereum.org/en/developers/docs/consensus-mechanisms/pos/)
- [Mining Energy Consumption](https://cbeci.org/)