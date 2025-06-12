"""
Blockchain Flow Chart Visualization

This module provides interactive visualizations of blockchain mechanics
using Plotly and NetworkX.
"""

import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Tuple
import time
from ..crypto_algorithms.blockchain.block import Block, Transaction
from ..crypto_algorithms.blockchain.chain import Blockchain

class BlockchainVisualizer:
    """
    Interactive blockchain visualization tools.
    """
    
    def __init__(self):
        """Initialize visualizer."""
        self.chain = Blockchain()
        self.layout = dict(
            title_x=0.5,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    
    def create_block_graph(self, blocks: List[Block]) -> Tuple[nx.DiGraph, dict]:
        """
        Create NetworkX graph from blocks.
        
        Args:
            blocks: List of blocks to visualize
            
        Returns:
            Tuple of (graph, positions)
        """
        G = nx.DiGraph()
        
        # Add nodes
        for i, block in enumerate(blocks):
            G.add_node(block.get_hash(),
                      height=i,
                      timestamp=block.header.timestamp,
                      transactions=len(block.transactions))
        
        # Add edges
        for block in blocks[1:]:
            G.add_edge(block.header.previous_hash, block.get_hash())
        
        # Calculate positions
        pos = nx.spring_layout(G)
        
        return G, pos
    
    def visualize_chain(self, width: int = 800, height: int = 600) -> go.Figure:
        """
        Create interactive chain visualization.
        
        Args:
            width: Plot width
            height: Plot height
            
        Returns:
            Plotly figure
        """
        # Get blocks in main chain
        blocks = [self.chain.get_block_by_height(i)
                 for i in range(len(self.chain.main_chain))]
        
        # Create graph
        G, pos = self.create_block_graph(blocks)
        
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
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            block_data = G.nodes[node]
            node_text.append(
                f"Height: {block_data['height']}<br>"
                f"Transactions: {block_data['transactions']}<br>"
                f"Time: {time.ctime(block_data['timestamp'])}"
            )
            node_size.append(block_data['transactions'] * 10 + 20)
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=node_size,
                color=[G.nodes[node]['height'] for node in G.nodes()],
                line_width=2
            ),
            name='Blocks'
        ))
        
        # Update layout
        fig.update_layout(
            title='Blockchain Structure',
            width=width,
            height=height,
            **self.layout
        )
        
        return fig
    
    def visualize_block_creation(self, block: Block,
                               width: int = 800,
                               height: int = 800) -> go.Figure:
        """
        Visualize block creation process.
        
        Args:
            block: Block to visualize
            width: Plot width
            height: Plot height
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Transaction Pool',
                'Block Structure',
                'Mining Process'
            ),
            vertical_spacing=0.1
        )
        
        # Transaction pool visualization
        tx_amounts = [tx.amount for tx in block.transactions]
        tx_fees = [tx.fee for tx in block.transactions]
        
        fig.add_trace(go.Bar(
            x=[f'Tx {i}' for i in range(len(tx_amounts))],
            y=tx_amounts,
            name='Amount',
            marker_color='blue'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=[f'Tx {i}' for i in range(len(tx_fees))],
            y=tx_fees,
            name='Fee',
            marker_color='green'
        ), row=1, col=1)
        
        # Block structure visualization
        components = [
            ('Version', 4),
            ('Previous Hash', 32),
            ('Merkle Root', 32),
            ('Timestamp', 4),
            ('Difficulty', 4),
            ('Nonce', 4)
        ]
        
        fig.add_trace(go.Bar(
            x=[c[0] for c in components],
            y=[c[1] for c in components],
            name='Header Fields',
            marker_color='red'
        ), row=2, col=1)
        
        # Mining visualization
        nonces = list(range(0, 1000, 100))
        hashes = [int(block.get_hash(), 16) % 1000 for _ in nonces]
        
        fig.add_trace(go.Scatter(
            x=nonces,
            y=hashes,
            mode='lines+markers',
            name='Hash Values',
            marker_color='purple'
        ), row=3, col=1)
        
        # Add target line
        target = int('0' * block.header.difficulty_target + 'f' * (64 - block.header.difficulty_target), 16) % 1000
        fig.add_trace(go.Scatter(
            x=[min(nonces), max(nonces)],
            y=[target, target],
            mode='lines',
            name='Target',
            line=dict(color='red', dash='dash')
        ), row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title='Block Creation Process',
            width=width,
            height=height,
            showlegend=True
        )
        
        return fig
    
    def visualize_transaction_flow(self, transaction: Transaction,
                                 width: int = 800,
                                 height: int = 400) -> go.Figure:
        """
        Visualize transaction flow.
        
        Args:
            transaction: Transaction to visualize
            width: Plot width
            height: Plot height
            
        Returns:
            Plotly figure
        """
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["Sender", "Network", "Recipient"],
                color=["blue", "gray", "green"]
            ),
            link=dict(
                source=[0, 0, 1],
                target=[1, 1, 2],
                value=[transaction.amount, transaction.fee, transaction.amount],
                color=["blue", "red", "green"]
            )
        )])
        
        # Update layout
        fig.update_layout(
            title='Transaction Flow',
            width=width,
            height=height
        )
        
        return fig
    
    def visualize_network_stats(self, width: int = 800,
                              height: int = 400) -> go.Figure:
        """
        Visualize network statistics.
        
        Args:
            width: Plot width
            height: Plot height
            
        Returns:
            Plotly figure
        """
        # Get network stats
        stats = self.chain.get_chain_stats()
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=list(range(stats['height'])),
                y=[block.header.difficulty_target
                   for block in [self.chain.get_block_by_height(i)
                               for i in range(stats['height'])]],
                name="Difficulty"
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(stats['height'])),
                y=[len(block.transactions)
                   for block in [self.chain.get_block_by_height(i)
                               for i in range(stats['height'])]],
                name="Transactions"
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title='Network Statistics',
            width=width,
            height=height,
            xaxis_title='Block Height',
            yaxis_title='Difficulty',
            yaxis2_title='Transactions per Block'
        )
        
        return fig

# Example usage and tests
if __name__ == "__main__":
    # Create visualizer
    viz = BlockchainVisualizer()
    
    # Add some blocks
    for i in range(5):
        # Create transactions
        for _ in range(3):
            tx = Transaction(
                from_address=f"user{i}",
                to_address=f"user{i+1}",
                amount=1.0,
                fee=0.1,
                nonce=i
            )
            viz.chain.add_transaction(tx)
        
        # Mine block
        viz.chain.mine_block("miner")
    
    # Create visualizations
    chain_fig = viz.visualize_chain()
    chain_fig.show()
    
    block_fig = viz.visualize_block_creation(viz.chain.get_latest_block())
    block_fig.show()
    
    tx = viz.chain.get_latest_block().transactions[0]
    tx_fig = viz.visualize_transaction_flow(tx)
    tx_fig.show()
    
    stats_fig = viz.visualize_network_stats()
    stats_fig.show()