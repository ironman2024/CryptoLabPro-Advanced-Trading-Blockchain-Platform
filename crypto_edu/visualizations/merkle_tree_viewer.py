"""
Merkle Tree Visualization

This module provides interactive visualizations of Merkle trees
including proof verification and tree construction.
"""

import hashlib
import json
from typing import List, Dict, Tuple, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

class Transaction:
    """Simple transaction class for Merkle tree demonstration."""
    
    def __init__(self, sender: str, receiver: str, amount: float, fee: float = 0.0, timestamp: int = 0):
        """
        Initialize a transaction.
        
        Args:
            sender: Sender address
            receiver: Receiver address
            amount: Transaction amount
            fee: Transaction fee
            timestamp: Transaction timestamp
        """
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.fee = fee
        self.timestamp = timestamp
    
    def get_hash(self) -> str:
        """Calculate transaction hash."""
        tx_string = f"{self.sender}{self.receiver}{self.amount}{self.fee}{self.timestamp}"
        return hashlib.sha256(tx_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary."""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "fee": self.fee,
            "timestamp": self.timestamp,
            "hash": self.get_hash()
        }

class MerkleTree:
    """Merkle tree implementation with educational components."""
    
    @staticmethod
    def hash_pair(left: str, right: str) -> str:
        """
        Hash a pair of values.
        
        Args:
            left: Left value
            right: Right value
            
        Returns:
            Combined hash
        """
        combined = left + right
        return hashlib.sha256(combined.encode()).hexdigest()
    
    @classmethod
    def build(cls, transactions: List[Transaction]) -> List[List[str]]:
        """
        Build a Merkle tree from transactions.
        
        Args:
            transactions: List of transactions
            
        Returns:
            List of levels in the tree (bottom-up)
        """
        if not transactions:
            return [[hashlib.sha256(b"").hexdigest()]]
        
        # Get transaction hashes (leaf nodes)
        leaves = [tx.get_hash() for tx in transactions]
        
        # If odd number of leaves, duplicate the last one
        if len(leaves) % 2 == 1:
            leaves.append(leaves[-1])
        
        # Build tree bottom-up
        tree = [leaves]
        
        while len(tree[0]) > 1:
            level = []
            nodes = tree[0]
            
            for i in range(0, len(nodes), 2):
                if i + 1 < len(nodes):
                    level.append(cls.hash_pair(nodes[i], nodes[i+1]))
                else:
                    level.append(nodes[i])  # Odd node, promote to next level
            
            # If odd number of nodes, duplicate the last one
            if len(level) % 2 == 1 and len(level) > 1:
                level.append(level[-1])
                
            tree.insert(0, level)
        
        return tree
    
    @classmethod
    def get_root(cls, transactions: List[Transaction]) -> str:
        """
        Get the Merkle root of transactions.
        
        Args:
            transactions: List of transactions
            
        Returns:
            Merkle root hash
        """
        tree = cls.build(transactions)
        return tree[0][0]
    
    @classmethod
    def get_proof(cls, transactions: List[Transaction], tx_index: int) -> List[Dict[str, str]]:
        """
        Get Merkle proof for a transaction.
        
        Args:
            transactions: List of transactions
            tx_index: Index of transaction to prove
            
        Returns:
            List of proof steps
        """
        if tx_index < 0 or tx_index >= len(transactions):
            raise ValueError("Transaction index out of range")
        
        # Build tree
        tree = cls.build(transactions)
        
        # Start with leaf node
        proof = []
        index = tx_index
        
        # Go up the tree
        for level in range(len(tree) - 1, 0, -1):
            is_right = index % 2 == 1
            pair_index = index - 1 if is_right else index + 1
            
            # Check if pair index exists
            if pair_index < len(tree[level]):
                proof.append({
                    "position": "left" if is_right else "right",
                    "hash": tree[level][pair_index]
                })
            
            # Move to parent
            index = index // 2
        
        return proof
    
    @classmethod
    def verify_proof(cls, tx_hash: str, proof: List[Dict[str, str]], root: str) -> bool:
        """
        Verify a Merkle proof.
        
        Args:
            tx_hash: Transaction hash
            proof: Merkle proof
            root: Expected Merkle root
            
        Returns:
            True if proof is valid
        """
        current = tx_hash
        
        for step in proof:
            if step["position"] == "left":
                current = cls.hash_pair(step["hash"], current)
            else:
                current = cls.hash_pair(current, step["hash"])
        
        return current == root

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
    
    def create_tree_graph(self, transactions: List[Transaction]) -> Tuple[nx.DiGraph, Dict]:
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
                    # Connect to children
                    child_level = level + 1
                    if child_level < len(tree):
                        child_idx_start = i * 2
                        if child_idx_start < len(tree[child_level]):
                            G.add_edge(f"{level}_{i}", f"{child_level}_{child_idx_start}")
                            
                            if child_idx_start + 1 < len(tree[child_level]):
                                G.add_edge(f"{level}_{i}", f"{child_level}_{child_idx_start+1}")
        
        # Calculate positions
        pos = {}
        for node in G.nodes():
            level, idx = map(int, node.split('_'))
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
        for i, step in enumerate(proof):
            # Show hash bytes visualization
            fig.add_trace(go.Heatmap(
                z=[[int(b) for b in format(int(step['hash'][:8], 16), '032b')]],
                colorscale='Viridis',
                showscale=False,
                name=f'Step {i+1}'
            ), row=2, col=1)
        
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
                        # Connect to children
                        child_level = l + 1
                        if child_level < len(tree):
                            child_idx_start = i * 2
                            if child_idx_start < len(tree[child_level]):
                                G.add_edge(f"{l}_{i}", f"{child_level}_{child_idx_start}")
                                
                                if child_idx_start + 1 < len(tree[child_level]):
                                    G.add_edge(f"{l}_{i}", f"{child_level}_{child_idx_start+1}")
            
            # Calculate positions
            pos = {}
            for node in G.nodes():
                l, idx = map(int, node.split('_'))
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
        
        # Add initial data (empty)
        fig.add_trace(go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Connections'
        ))
        
        fig.add_trace(go.Scatter(
            x=[], y=[],
            mode='markers',
            hoverinfo='text',
            text=[],
            marker=dict(
                size=20,
                color=[],
                line_width=2
            ),
            name='Nodes'
        ))
        
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
    
    def export_tree_data(self, transactions: List[Transaction]) -> str:
        """
        Export tree data as JSON for external visualization.
        
        Args:
            transactions: List of transactions
            
        Returns:
            JSON string of tree data
        """
        tree = MerkleTree.build(transactions)
        
        # Convert to more readable format
        tree_data = []
        for level, nodes in enumerate(tree):
            level_data = []
            for i, node in enumerate(nodes):
                # Find children
                children = []
                if level < len(tree) - 1:
                    child_idx_start = i * 2
                    if child_idx_start < len(tree[level + 1]):
                        children.append({
                            "level": level + 1,
                            "index": child_idx_start,
                            "hash": tree[level + 1][child_idx_start]
                        })
                        
                        if child_idx_start + 1 < len(tree[level + 1]):
                            children.append({
                                "level": level + 1,
                                "index": child_idx_start + 1,
                                "hash": tree[level + 1][child_idx_start + 1]
                            })
                
                level_data.append({
                    "hash": node,
                    "index": i,
                    "children": children
                })
            
            tree_data.append({
                "level": level,
                "nodes": level_data
            })
        
        # Add transaction data
        tx_data = [tx.to_dict() for tx in transactions]
        
        data = {
            "tree": tree_data,
            "transactions": tx_data,
            "root": tree[0][0] if tree else None
        }
        
        return json.dumps(data, indent=2)

# Example usage and tests
if __name__ == "__main__":
    # Create visualizer
    viz = MerkleTreeVisualizer()
    
    # Create some transactions
    transactions = [
        Transaction("Alice", "Bob", 1.0, 0.1, 1),
        Transaction("Bob", "Charlie", 0.5, 0.1, 2),
        Transaction("Charlie", "Dave", 0.25, 0.1, 3),
        Transaction("Dave", "Alice", 0.75, 0.1, 4)
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
    
    # Export tree data
    with open("merkle_tree_data.json", "w") as f:
        f.write(viz.export_tree_data(transactions))