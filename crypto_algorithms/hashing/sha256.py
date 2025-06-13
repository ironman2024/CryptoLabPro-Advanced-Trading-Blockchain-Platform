"""
SHA-256 hash function implementation
"""

import hashlib
from typing import Union

class SHA256:
    """SHA-256 hash function wrapper."""
    
    def hash(self, data: Union[str, bytes]) -> str:
        """
        Calculate SHA-256 hash of input data.
        
        Args:
            data: Input data as string or bytes
            
        Returns:
            Hex string of hash
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        return hashlib.sha256(data).hexdigest()