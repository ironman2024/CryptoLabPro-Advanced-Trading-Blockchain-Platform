# Wallet Components

This directory contains implementations of cryptocurrency wallet functionality including HD wallets and key generation utilities.

## Components

### HD Wallets (BIP32/39/44)
- Complete BIP39 mnemonic implementation
- Hierarchical key derivation (BIP32)
- Standard derivation paths (BIP44)
- Multiple address type support

### Key Generation
- ECDSA keypair generation
- Schnorr keypair generation (BIP340)
- Multiple address formats
- Multisig address creation

## Usage Examples

### Creating HD Wallet
```python
from wallets.hd_wallets import Wallet

# Create new wallet with random mnemonic
wallet = Wallet()
print(f"Mnemonic: {wallet.mnemonic}")

# Create account (BIP44 path)
address = wallet.create_account(0)
print(f"Address: {address}")

# Sign transaction
tx = wallet.sign_transaction(
    from_address=address,
    to_address="recipient_address",
    amount=1.0,
    fee=0.1
)
```

### Generating Keys
```python
from wallets.keygen import KeyGenerator

# Create key generator
keygen = KeyGenerator(testnet=False)

# Generate ECDSA keypair
keypair = keygen.generate_keypair()
print(f"Private key (WIF): {keypair.wif}")
print(f"Address: {keypair.address}")

# Create multisig address
public_keys = [keygen.generate_keypair().public_key for _ in range(3)]
address, redeem_script = keygen.create_multisig_address(public_keys, 2)
print(f"2-of-3 multisig address: {address}")
```

## Address Types

The implementation supports multiple address types:

1. **P2PKH (Pay to Public Key Hash)**
   - Legacy Bitcoin addresses
   - Starting with '1'

2. **P2SH (Pay to Script Hash)**
   - Multisig and nested SegWit
   - Starting with '3'

3. **P2WPKH (Native SegWit)**
   - Bech32 addresses
   - Starting with 'bc1q'

4. **P2TR (Taproot)**
   - Schnorr signatures
   - Starting with 'bc1p'

## Security Features

### Key Generation
- Strong randomness (secrets module)
- Compressed public keys by default
- Optional testnet support

### HD Wallet
- BIP39 mnemonic validation
- Optional passphrase support
- Hardened derivation
- Key export controls

## Educational Components

### Mnemonic Generation
```python
from wallets.hd_wallets import HDWallet

hd = HDWallet()
mnemonic = hd.generate_mnemonic(strength=256)  # 24 words
seed = hd.mnemonic_to_seed(mnemonic, passphrase="optional")
```

### Address Generation
```python
from wallets.keygen import KeyGenerator

keygen = KeyGenerator()
keypair = keygen.generate_keypair()

# Different address formats
p2pkh = keygen.public_key_to_address(keypair.public_key, 'p2pkh')
p2sh = keygen.public_key_to_address(keypair.public_key, 'p2sh')
p2wpkh = keygen.public_key_to_address(keypair.public_key, 'p2wpkh')
p2tr = keygen.public_key_to_address(keypair.public_key, 'p2tr')
```

## Performance Notes

Typical performance on modern hardware:

Key Generation:
- ECDSA keypair: ~1000/s
- Schnorr keypair: ~2000/s
- HD derivation: ~500/s

Address Generation:
- P2PKH/P2SH: ~5000/s
- P2WPKH/P2TR: ~2000/s

## References

- [BIP32 - HD Wallets](https://github.com/bitcoin/bips/blob/master/bip-0032.mediawiki)
- [BIP39 - Mnemonic code](https://github.com/bitcoin/bips/blob/master/bip-0039.mediawiki)
- [BIP44 - HD Wallet structure](https://github.com/bitcoin/bips/blob/master/bip-0044.mediawiki)
- [BIP340 - Schnorr Signatures](https://github.com/bitcoin/bips/blob/master/bip-0340.mediawiki)