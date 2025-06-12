"""
Merkle Tree Visualization

This module provides interactive visualizations of Merkle trees
including proof verification and tree construction.
"""

import plotly.graph_objects as go
import networkx as nx
from typing import List, Dict, Tuple
import hashlib
from ..crypto_algorithms.blockchain.block import Transaction, MerkleTree

class MerkleTreeVisualizer:
    """
    Interactive Merkle tree visualization tools.
    """
    
    def __init__(self):
        """Initialize visualizer."""
        self.layout = dict(
            title_x=0.5,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    
    def create_tree_graph(self, transactions: List[Transaction]) -> Tuple[nx.DiGraph, dict]:
        """
        Create NetworkX graph from Merkle tree.
        
        Args:
            transactions: List of transactions
            
        Returns:
            Tuple of (graph, positions)
        """
        G = nx.DiGraph()
        
        # Build Merkle tree
        tree = MerkleTree.build(transactions)
        
        # Add nodes level by level
        for level, nodes in enumerate(tree):
            for i, node in enumerate(nodes):
                G.add_node(f"{level}_{i}", hash=node, level=level)
                
                if level > 0:
                    # Connect to parent
                    parent_idx = i // 2
                    G.add_edge(f"{level-1}_{parent_idx}", f"{level}_{i}")
        
        # Calculate positions
        pos = {}
        for node in G.nodes():
            level = G.nodes[node]['level']
            idx = int(node.split('_')[1])
            width = len(tree[level])
            pos[node] = (idx - width/2, -level)
        
        return G, pos
    
    def visualize_tree(self, transactions: List[Transaction],
                      highlight_path: List[int] = None,
                      width: int = 800,
                      height: int = 600) -> go.Figure:
        """
        Create interactive Merkle tree visualization.
        
        Args:
            transactions: List of transactions
            highlight_path: List of indices to highlight (for proof)
            width: Plot width
            height: Plot height
            
        Returns:
            Plotly figure
        """
        # Create graph
        G, pos = self.create_tree_graph(transactions)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Connections'
        ))
        
        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            level, idx = map(int, node.split('_'))
            hash_val = G.nodes[node]['hash']
            node_text.append(
                f"Level: {level}<br>"
                f"Index: {idx}<br>"
                f"Hash: {hash_val[:8]}..."
            )
            
            # Color nodes in proof path
            if highlight_path and idx in highlight_path:
                node_color.append('red')
            else:
                node_color.append('lightblue')
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=20,
                color=node_color,
                line_width=2
            ),
            name='Nodes'
        ))
        
        # Update layout
        fig.update_layout(
            title='Merkle Tree Structure',
            width=width,
            height=height,
            **self.layout
        )
        
        return fig
    
    def visualize_proof_verification(self, transactions: List[Transaction],
                                   tx_index: int,
                                   width: int = 1000,
                                   height: int = 800) -> go.Figure:
        """
        Visualize Merkle proof verification process.
        
        Args:
            transactions: List of transactions
            tx_index: Index of transaction to prove
            width: Plot width
            height: Plot height
            
        Returns:
            Plotly figure
        """
        # Get proof
        proof = MerkleTree.get_proof(transactions, tx_index)
        root = MerkleTree.get_root(transactions)
        tx_hash = transactions[tx_index].get_hash()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Merkle Tree',
                'Proof Steps',
                'Hash Concatenation',
                'Verification Result'
            )
        )
        
        # Add Merkle tree visualization
        tree_fig = self.visualize_tree(
            transactions,
            highlight_path=[tx_index]
        )
        
        for trace in tree_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        # Visualize proof steps
        step_x = []
        step_y = []
        step_text = []
        current_hash = tx_hash
        
        for i, step in enumerate(proof):
            step_x.append(i)
            step_y.append(0 if step['position'] == 'left' else 1)
            step_text.append(
                f"Step {i+1}:<br>"
                f"Position: {step['position']}<br>"
                f"Hash: {step['hash'][:8]}..."
            )
            
            # Calculate next hash
            if step['position'] == 'left':
                current_hash = MerkleTree.hash_pair(step['hash'], current_hash)
            else:
                current_hash = MerkleTree.hash_pair(current_hash, step['hash'])
        
        fig.add_trace(go.Scatter(
            x=step_x,
            y=step_y,
            mode='markers+lines',
            text=step_text,
            marker=dict(size=15),
            name='Proof Steps'
        ), row=1, col=2)
        
        # Visualize hash concatenation
        concat_fig = go.Figure()
        
        for i, step in enumerate(proof):
            # Show bytes being concatenated
            concat_fig.add_trace(go.Heatmap(
                z=[[int(b) for b in format(int(step['hash'], 16), '0256b')]],
                colorscale='Viridis',
                showscale=False,
                name=f'Step {i+1}'
            ))
        
        for trace in concat_fig.data:
            fig.add_trace(trace, row=2, col=1)
        
        # Show verification result
        verified = current_hash == root
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=100 if verified else 0,
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color="green" if verified else "red"),
                steps=[
                    dict(range=[0, 50], color="red"),
                    dict(range=[50, 100], color="green")
                ]
            ),
            title=dict(text="Proof Verification")
        ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title='Merkle Proof Verification',
            width=width,
            height=height,
            showlegend=True
        )
        
        return fig
    
    def visualize_tree_construction(self, transactions: List[Transaction],
                                  width: int = 800,
                                  height: int = 600) -> go.Figure:
        """
        Animate Merkle tree construction process.
        
        Args:
            transactions: List of transactions
            width: Plot width
            height: Plot height
            
        Returns:
            Plotly figure with animation
        """
        # Create base figure
        fig = go.Figure()
        
        # Build tree level by level
        tree = MerkleTree.build(transactions)
        frames = []
        
        for level in range(len(tree)):
            # Create graph up to current level
            G = nx.DiGraph()
            
            for l in range(level + 1):
                for i, node in enumerate(tree[l]):
                    G.add_node(f"{l}_{i}", hash=node, level=l)
                    
                    if l > 0:
                        parent_idx = i // 2
                        G.add_edge(f"{l-1}_{parent_idx}", f"{l}_{i}")
            
            # Calculate positions
            pos = {}
            for node in G.nodes():
                l = G.nodes[node]['level']
                idx = int(node.split('_')[1])
                width = len(tree[l])
                pos[node] = (idx - width/2, -l)
            
            # Create frame
            frame_data = []
            
            # Add edges
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
            frame_data.append(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines',
                name='Connections'
            ))
            
            # Add nodes
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                l, idx = map(int, node.split('_'))
                hash_val = G.nodes[node]['hash']
                node_text.append(
                    f"Level: {l}<br>"
                    f"Index: {idx}<br>"
                    f"Hash: {hash_val[:8]}..."
                )
                node_color.append('lightblue')
            
            frame_data.append(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    size=20,
                    color=node_color,
                    line_width=2
                ),
                name='Nodes'
            ))
            
            frames.append(go.Frame(
                data=frame_data,
                name=f"level_{level}"
            ))
        
        # Add frames to figure
        fig.frames = frames
        
        # Add play button
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 1000, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 500}
                    }]
                }]
            }]
        )
        
        # Update layout
        fig.update_layout(
            title='Merkle Tree Construction',
            width=width,
            height=height,
            **self.layout
        )
        
        return fig

# Example usage and tests
if __name__ == "__main__":
    # Create visualizer
    viz = MerkleTreeVisualizer()
    
    # Create some transactions
    transactions = [
        Transaction("Alice", "Bob", 1.0, 0.1, 1),
        Transaction("Bob", "Charlie", 0.5, 0.1, 1),
        Transaction("Charlie", "Dave", 0.25, 0.1, 1),
        Transaction("Dave", "Alice", 0.75, 0.1, 1)
    ]
    
    # Visualize tree
    tree_fig = viz.visualize_tree(transactions)
    tree_fig.show()
    
    # Visualize proof verification
    proof_fig = viz.visualize_proof_verification(transactions, 1)
    proof_fig.show()
    
    # Visualize tree construction
    construction_fig = viz.visualize_tree_construction(transactions)
    construction_fig.show()