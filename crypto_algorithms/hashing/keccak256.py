"""
Keccak-256 hash function implementation
"""

import hashlib
from typing import Union

class Keccak256:
    """Keccak-256 hash function wrapper."""
    
    def hash(self, data: Union[str, bytes]) -> str:
        """
        Calculate Keccak-256 hash of input data.
        
        Args:
            data: Input data as string or bytes
            
        Returns:
            Hex string of hash
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        # Use SHA3-256 which is based on Keccak
        return hashlib.sha3_256(data).hexdigest()