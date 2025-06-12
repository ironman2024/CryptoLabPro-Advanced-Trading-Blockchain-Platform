"""
Hash Function Animation

This module provides interactive visualizations of hash function mechanics
including the avalanche effect and collision demonstrations.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Tuple, Dict
import time
from ..crypto_algorithms.hashing.sha256 import SHA256
from ..crypto_algorithms.hashing.keccak256 import Keccak256

class HashVisualizer:
    """
    Interactive hash function visualization tools.
    """
    
    def __init__(self):
        """Initialize visualizer."""
        self.sha256 = SHA256(use_builtin=False)  # Use educational implementation
        self.keccak256 = Keccak256(use_builtin=False)
        
    def _bytes_to_bits(self, data: bytes) -> List[int]:
        """Convert bytes to list of bits."""
        result = []
        for b in data:
            for i in range(8):
                result.append((b >> (7-i)) & 1)
        return result
    
    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """Calculate Hamming distance between two hex strings."""
        b1 = self._bytes_to_bits(bytes.fromhex(hash1))
        b2 = self._bytes_to_bits(bytes.fromhex(hash2))
        return sum(a != b for a, b in zip(b1, b2))
    
    def visualize_avalanche(self, message: str,
                          width: int = 1000,
                          height: int = 600) -> go.Figure:
        """
        Visualize avalanche effect by flipping bits.
        
        Args:
            message: Input message
            width: Plot width
            height: Plot height
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'SHA-256 Bit Changes',
                'Keccak-256 Bit Changes',
                'SHA-256 Bit Distribution',
                'Keccak-256 Bit Distribution'
            )
        )
        
        # Generate data
        message_bytes = message.encode()
        results = []
        
        for i in range(len(message_bytes) * 8):
            # Flip each bit
            mod_bytes = bytearray(message_bytes)
            mod_bytes[i // 8] ^= 1 << (7 - (i % 8))
            mod_message = bytes(mod_bytes)
            
            # Calculate hashes
            sha_orig = self.sha256.hash(message_bytes)
            sha_mod = self.sha256.hash(mod_message)
            keccak_orig = self.keccak256.hash(message_bytes)
            keccak_mod = self.keccak256.hash(mod_message)
            
            # Calculate differences
            sha_diff = self._hamming_distance(sha_orig, sha_mod)
            keccak_diff = self._hamming_distance(keccak_orig, keccak_mod)
            
            results.append((sha_diff, keccak_diff))
        
        # Plot bit changes
        x = list(range(len(results)))
        sha_diffs = [r[0] for r in results]
        keccak_diffs = [r[1] for r in results]
        
        fig.add_trace(go.Scatter(
            x=x,
            y=sha_diffs,
            mode='lines+markers',
            name='SHA-256'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=x,
            y=keccak_diffs,
            mode='lines+markers',
            name='Keccak-256'
        ), row=1, col=2)
        
        # Plot bit distributions
        fig.add_trace(go.Histogram(
            x=sha_diffs,
            nbinsx=20,
            name='SHA-256 Distribution'
        ), row=2, col=1)
        
        fig.add_trace(go.Histogram(
            x=keccak_diffs,
            nbinsx=20,
            name='Keccak-256 Distribution'
        ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title='Avalanche Effect Visualization',
            width=width,
            height=height,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Bit Position", row=1, col=1)
        fig.update_xaxes(title_text="Bit Position", row=1, col=2)
        fig.update_xaxes(title_text="Changed Bits", row=2, col=1)
        fig.update_xaxes(title_text="Changed Bits", row=2, col=2)
        fig.update_yaxes(title_text="Bits Changed", row=1, col=1)
        fig.update_yaxes(title_text="Bits Changed", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        return fig
    
    def visualize_collision_search(self, prefix_bits: int = 16,
                                 max_attempts: int = 1000,
                                 width: int = 800,
                                 height: int = 600) -> go.Figure:
        """
        Visualize collision search process.
        
        Args:
            prefix_bits: Number of bits to match
            max_attempts: Maximum attempts
            width: Plot width
            height: Plot height
            
        Returns:
            Plotly figure
        """
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                'Collision Search Progress',
                'Hash Distribution'
            )
        )
        
        # Search for collisions
        seen = {}
        attempts = []
        collisions = []
        prefix_mask = (1 << prefix_bits) - 1
        
        for i in range(max_attempts):
            message = f"test{i}".encode()
            hash_val = int(self.sha256.hash(message)[:prefix_bits//4], 16)
            
            if hash_val in seen and seen[hash_val] != message:
                collisions.append(i)
            
            seen[hash_val] = message
            attempts.append(hash_val & prefix_mask)
        
        # Plot search progress
        fig.add_trace(go.Scatter(
            x=list(range(len(attempts))),
            y=attempts,
            mode='markers',
            marker=dict(
                size=5,
                color=attempts,
                colorscale='Viridis',
                showscale=True
            ),
            name='Hash Values'
        ), row=1, col=1)
        
        # Mark collisions
        if collisions:
            fig.add_trace(go.Scatter(
                x=collisions,
                y=[attempts[i] for i in collisions],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='x'
                ),
                name='Collisions'
            ), row=1, col=1)
        
        # Plot hash distribution
        fig.add_trace(go.Histogram(
            x=attempts,
            nbinsx=32,
            name='Hash Distribution'
        ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'Collision Search ({prefix_bits} bits)',
            width=width,
            height=height,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Attempts", row=1, col=1)
        fig.update_xaxes(title_text="Hash Value", row=2, col=1)
        fig.update_yaxes(title_text="Hash Value", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        return fig
    
    def visualize_hash_internals(self, message: str,
                               width: int = 1000,
                               height: int = 800) -> go.Figure:
        """
        Visualize internal state of hash function.
        
        Args:
            message: Input message
            width: Plot width
            height: Plot height
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Message Schedule',
                'Compression Function',
                'Round Constants',
                'State Evolution'
            )
        )
        
        # Get SHA-256 state
        message_bytes = message.encode()
        padded = self.sha256._pad_message(message_bytes)
        block = padded[:64]  # First block only
        
        # Message schedule
        w = [0] * 64
        for i in range(16):
            w[i] = int.from_bytes(block[i*4:(i+1)*4], 'big')
        
        for i in range(16, 64):
            s0 = self.sha256.gamma0(w[i-15])
            s1 = self.sha256.gamma1(w[i-2])
            w[i] = (w[i-16] + s0 + w[i-7] + s1) & 0xffffffff
        
        # Plot message schedule
        fig.add_trace(go.Heatmap(
            z=[[w[i] >> (24-j*8) & 0xff for j in range(4)]
               for i in range(64)],
            colorscale='Viridis',
            name='Message Words'
        ), row=1, col=1)
        
        # Plot compression function
        h = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        ]
        states = []
        a, b, c, d, e, f, g, h_val = h
        
        for i in range(64):
            states.append([a, b, c, d, e, f, g, h_val])
            
            t1 = (h_val +
                  self.sha256.sigma1(e) +
                  self.sha256.ch(e, f, g) +
                  self.sha256.K[i] +
                  w[i]) & 0xffffffff
            
            t2 = (self.sha256.sigma0(a) +
                  self.sha256.maj(a, b, c)) & 0xffffffff
            
            h_val = g
            g = f
            f = e
            e = (d + t1) & 0xffffffff
            d = c
            c = b
            b = a
            a = (t1 + t2) & 0xffffffff
        
        fig.add_trace(go.Heatmap(
            z=[[s >> 24 & 0xff for s in state] for state in states],
            colorscale='Viridis',
            name='State Values'
        ), row=1, col=2)
        
        # Plot round constants
        fig.add_trace(go.Heatmap(
            z=[[k >> (24-j*8) & 0xff for j in range(4)]
               for k in self.sha256.K],
            colorscale='Viridis',
            name='Round Constants'
        ), row=2, col=1)
        
        # Plot state evolution
        fig.add_trace(go.Scatter(
            x=list(range(64)),
            y=[sum(s) & 0xffffffff for s in states],
            mode='lines',
            name='State Sum'
        ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title='SHA-256 Internal State',
            width=width,
            height=height,
            showlegend=True
        )
        
        return fig

# Example usage and tests
if __name__ == "__main__":
    # Create visualizer
    viz = HashVisualizer()
    
    # Visualize avalanche effect
    avalanche_fig = viz.visualize_avalanche("Hello, World!")
    avalanche_fig.show()
    
    # Visualize collision search
    collision_fig = viz.visualize_collision_search(prefix_bits=16)
    collision_fig.show()
    
    # Visualize hash internals
    internals_fig = viz.visualize_hash_internals("Hello, World!")
    internals_fig.show()