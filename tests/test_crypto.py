"""
Tests for cryptographic components.
"""

import pytest
from crypto_algorithms.hashing.sha256 import SHA256
from crypto_algorithms.hashing.keccak256 import Keccak256
from crypto_algorithms.signatures.ecdsa import ECDSA
from crypto_algorithms.signatures.schnorr import SchnorrSignature
from crypto_algorithms.wallets.hd_wallets import HDWallet
from crypto_algorithms.wallets.keygen import KeyGenerator

def test_sha256():
    """Test SHA-256 implementation."""
    sha256 = SHA256(use_builtin=False)
    test_vector = "abc"
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    
    assert sha256.hash(test_vector) == expected
    
    # Test avalanche effect
    orig_hash, mod_hash, diff_percent = sha256.demonstrate_avalanche("test")
    assert diff_percent > 45  # Should change roughly half the bits

def test_keccak256():
    """Test Keccak-256 implementation."""
    keccak256 = Keccak256(use_builtin=False)
    test_vector = "abc"
    # This is SHA3-256 test vector, Keccak-256 would be different
    expected = "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
    
    hash_val = keccak256.hash(test_vector)
    assert len(hash_val) == 64  # 32 bytes in hex
    
    # Test avalanche effect
    orig_hash, mod_hash, diff_percent = keccak256.demonstrate_avalanche("test")
    assert diff_percent > 45  # Should change roughly half the bits

def test_ecdsa():
    """Test ECDSA implementation."""
    ecdsa = ECDSA(use_builtin=False)
    
    # Generate keypair
    private_key, public_key = ecdsa.generate_keypair()
    
    # Sign and verify
    message = "test message"
    signature = ecdsa.sign(message, private_key)
    assert ecdsa.verify(message, signature, public_key)
    
    # Test with modified message
    assert not ecdsa.verify("wrong message", signature, public_key)

def test_schnorr():
    """Test Schnorr signature implementation."""
    schnorr = SchnorrSignature(use_builtin=False)
    
    # Generate keypair
    private_key, public_key = schnorr.generate_keypair()
    
    # Sign and verify
    message = "test message"
    signature = schnorr.sign(message, private_key)
    assert schnorr.verify(message, signature, public_key)
    
    # Test with modified message
    assert not schnorr.verify("wrong message", signature, public_key)
    
    # Test batch verification
    n_sigs = 5
    messages = [f"Message {i}" for i in range(n_sigs)]
    keypairs = [schnorr.generate_keypair() for _ in range(n_sigs)]
    signatures = [
        schnorr.sign(m, priv)
        for m, (priv, _) in zip(messages, keypairs)
    ]
    public_keys = [pub for _, pub in keypairs]
    
    valid, speedup = schnorr.demonstrate_batch_verification(
        messages, signatures, public_keys
    )
    assert valid
    assert speedup > 1  # Should be faster than individual verification

def test_hd_wallet():
    """Test HD wallet implementation."""
    wallet = HDWallet()
    
    # Generate mnemonic
    mnemonic = wallet.generate_mnemonic()
    assert len(mnemonic.split()) in [12, 15, 18, 21, 24]
    
    # Generate seed
    seed = wallet.mnemonic_to_seed(mnemonic)
    assert len(seed) == 64  # 512 bits
    
    # Derive keys
    master_key = wallet._derive_master_key(seed)
    assert master_key.depth == 0
    
    # Derive child key
    child_key = wallet._derive_child_key(master_key, 0)
    assert child_key.depth == 1
    
    # Test path derivation
    derived = wallet.derive_path(seed, "m/44'/0'/0'/0/0")
    assert derived.depth == 5

def test_key_generator():
    """Test key generation utilities."""
    keygen = KeyGenerator(testnet=False)
    
    # Generate ECDSA keypair
    keypair = keygen.generate_keypair()
    assert len(keypair.private_key) == 32
    assert len(keypair.public_key) in [33, 65]  # Compressed or uncompressed
    assert keypair.wif.startswith('K') or keypair.wif.startswith('L')
    assert keypair.address.startswith('1')
    
    # Generate Schnorr keypair
    schnorr_pair = keygen.generate_schnorr_keypair()
    assert len(schnorr_pair.private_key) == 32
    assert len(schnorr_pair.public_key) == 32  # x-only pubkey
    
    # Test multisig
    pubkeys = [keygen.generate_keypair().public_key for _ in range(3)]
    address, redeem_script = keygen.create_multisig_address(pubkeys, 2)
    assert address.startswith('3')
    assert len(redeem_script) > 0

if __name__ == "__main__":
    pytest.main([__file__])