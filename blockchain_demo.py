import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import hashlib
import random
import json
from datetime import datetime

# Set page config
st.set_page_config(page_title="Blockchain Demo", layout="wide")

# Define simplified blockchain classes
class PosAccount:
    def __init__(self, address, balance=0.0, stake=0.0):
        self.address = address
        self.balance = balance
        self.stake = stake
        self.rewards = 0.0
    
    def get_total_stake(self):
        return self.stake
    
    def add_stake(self, amount):
        if amount <= 0 or amount > self.balance:
            return False
        self.balance -= amount
        self.stake += amount
        return True
    
    def add_reward(self, amount):
        self.rewards += amount
        self.balance += amount
    
    def to_dict(self):
        return {
            "address": self.address,
            "balance": self.balance,
            "stake": self.stake,
            "rewards": self.rewards
        }

class PosBlock:
    def __init__(self, index, previous_hash, validator, timestamp=None, data=""):
        self.index = index
        self.previous_hash = previous_hash
        self.validator = validator
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.data = data
        self.hash = self.calculate_hash()
    
    def calculate_hash(self):
        block_string = f"{self.index}{self.previous_hash}{self.validator}{self.timestamp}{self.data}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self):
        return {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "validator": self.validator,
            "timestamp": self.timestamp,
            "data": self.data,
            "hash": self.hash
        }

class PowBlock:
    def __init__(self, index, previous_hash, timestamp=None, data="", nonce=0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.data = data
        self.nonce = nonce
        self.hash = self.calculate_hash()
    
    def calculate_hash(self):
        block_string = f"{self.index}{self.previous_hash}{self.timestamp}{self.data}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self):
        return {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "data": self.data,
            "nonce": self.nonce,
            "hash": self.hash
        }

# Initialize session state
if 'pow_blocks' not in st.session_state:
    st.session_state.pow_blocks = [PowBlock(0, "0"*64, data="Genesis Block")]

if 'pos_blocks' not in st.session_state:
    st.session_state.pos_blocks = [PosBlock(0, "0"*64, validator="Genesis", data="Genesis Block")]

if 'pos_accounts' not in st.session_state:
    st.session_state.pos_accounts = {
        "Alice": PosAccount("Alice", 1000, 100),
        "Bob": PosAccount("Bob", 800, 200),
        "Charlie": PosAccount("Charlie", 1200, 300)
    }

# Title
st.title("🔗 Interactive Blockchain Demo")
st.markdown("Explore how blockchain consensus mechanisms work with this interactive demo")

# Main tabs
tab1, tab2 = st.tabs(["Proof of Work (PoW)", "Proof of Stake (PoS)"])

# PoW Tab
with tab1:
    st.header("Proof of Work Consensus")
    
    st.markdown("""
    **Proof of Work** requires miners to solve complex puzzles by finding a nonce that produces a hash with specific properties.
    
    Key characteristics:
    - Energy intensive
    - Highly secure against attacks
    - Used by Bitcoin
    """)
    
    # Mining form
    with st.form("pow_mining"):
        col1, col2 = st.columns([3, 1])
        with col1:
            data = st.text_area("Block Data", "Enter transaction data here...")
        with col2:
            difficulty = st.slider("Difficulty", 1, 5, 3)
        
        submitted = st.form_submit_button("Mine Block")
        
        if submitted:
            latest_block = st.session_state.pow_blocks[-1]
            new_block = PowBlock(
                index=latest_block.index + 1,
                previous_hash=latest_block.hash,
                data=data
            )
            
            # Mining simulation with progress bar
            target = '0' * difficulty
            
            with st.spinner("Mining in progress..."):
                progress_bar = st.progress(0)
                start_time = time.time()
                max_nonce = 1000000
                
                for nonce in range(max_nonce):
                    new_block.nonce = nonce
                    new_block.hash = new_block.calculate_hash()
                    
                    # Update progress bar occasionally
                    if nonce % 1000 == 0:
                        progress = min(nonce / max_nonce, 0.99)
                        progress_bar.progress(progress)
                    
                    if new_block.hash.startswith(target):
                        st.session_state.pow_blocks.append(new_block)
                        mining_time = time.time() - start_time
                        st.success(f"Block mined successfully in {mining_time:.2f} seconds!")
                        st.balloons()
                        break
                else:
                    st.error("Mining failed. Try reducing the difficulty.")
    
    # Display blockchain
    st.subheader("Blockchain Explorer")
    
    for block in st.session_state.pow_blocks:
        with st.expander(f"Block #{block.index} - {block.hash[:10]}..."):
            st.json(block.to_dict())
    
    # Visualization
    if len(st.session_state.pow_blocks) > 1:
        st.subheader("Blockchain Visualization")
        
        fig = go.Figure()
        
        # Add blocks as nodes
        for i, block in enumerate(st.session_state.pow_blocks):
            fig.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode='markers+text',
                marker=dict(size=40, color='blue'),
                text=[f"Block {block.index}"],
                textposition="bottom center",
                name=f"Block {block.index}"
            ))
        
        # Add connections between blocks
        for i in range(len(st.session_state.pow_blocks) - 1):
            fig.add_trace(go.Scatter(
                x=[i, i+1],
                y=[0, 0],
                mode='lines',
                line=dict(width=2, color='black'),
                showlegend=False
            ))
        
        fig.update_layout(
            title="Blockchain Structure",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

# PoS Tab
with tab2:
    st.header("Proof of Stake Consensus")
    
    st.markdown("""
    **Proof of Stake** selects validators based on the amount of cryptocurrency they hold and are willing to "stake".
    
    Key characteristics:
    - Energy efficient
    - Validators selected based on stake amount
    - Used by Ethereum 2.0, Cardano, and others
    """)
    
    # Account management
    st.subheader("Account Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create account
        with st.form("create_account"):
            new_name = st.text_input("Account Name")
            new_balance = st.number_input("Initial Balance", min_value=0.0, value=100.0)
            create_submitted = st.form_submit_button("Create Account")
            
            if create_submitted and new_name and new_name not in st.session_state.pos_accounts:
                st.session_state.pos_accounts[new_name] = PosAccount(new_name, new_balance)
                st.success(f"Account {new_name} created with {new_balance} coins!")
    
    with col2:
        # Stake management
        with st.form("stake_management"):
            account_name = st.selectbox("Select Account", list(st.session_state.pos_accounts.keys()))
            amount = st.number_input("Amount to Stake", min_value=0.1, value=10.0)
            stake_submitted = st.form_submit_button("Stake Coins")
            
            if stake_submitted and account_name in st.session_state.pos_accounts:
                account = st.session_state.pos_accounts[account_name]
                if account.add_stake(amount):
                    st.success(f"{account_name} staked {amount} coins successfully!")
                else:
                    st.error(f"Failed to stake. Insufficient balance.")
    
    # Display accounts
    st.subheader("Accounts Overview")
    
    accounts_df = pd.DataFrame([account.to_dict() for account in st.session_state.pos_accounts.values()])
    if not accounts_df.empty:
        st.dataframe(accounts_df)
    
    # Create new block
    st.subheader("Create New Block")
    
    with st.form("create_block"):
        block_data = st.text_area("Block Data", "Enter transaction data here...")
        create_block_submitted = st.form_submit_button("Create Block")
        
        if create_block_submitted:
            # Select validator based on stake
            validators = list(st.session_state.pos_accounts.keys())
            weights = [st.session_state.pos_accounts[v].stake for v in validators]
            
            if sum(weights) > 0:
                # Normalize weights
                weights = [w/sum(weights) for w in weights]
                
                # Select validator
                selected_validator = random.choices(validators, weights=weights, k=1)[0]
                
                # Create block
                prev_hash = st.session_state.pos_blocks[-1].hash
                new_block = PosBlock(
                    index=len(st.session_state.pos_blocks),
                    previous_hash=prev_hash,
                    validator=selected_validator,
                    data=block_data
                )
                
                # Add block
                st.session_state.pos_blocks.append(new_block)
                
                # Distribute rewards
                reward = 1.0  # 1 coin reward
                validator_account = st.session_state.pos_accounts[selected_validator]
                validator_account.add_reward(reward)
                
                st.success(f"Block #{new_block.index} created by validator {selected_validator}!")
                st.balloons()
            else:
                st.error("No validators available. Stake some coins first!")
    
    # Display blocks
    st.subheader("Blockchain Explorer")
    
    for block in st.session_state.pos_blocks:
        with st.expander(f"Block #{block.index} - Validated by {block.validator}"):
            st.json(block.to_dict())
    
    # Visualization
    if len(st.session_state.pos_blocks) > 1:
        st.subheader("Blockchain Visualization")
        
        fig = go.Figure()
        
        # Add blocks as nodes
        for i, block in enumerate(st.session_state.pos_blocks):
            fig.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode='markers+text',
                marker=dict(size=40, color='green'),
                text=[f"Block {block.index}<br>{block.validator}"],
                textposition="bottom center",
                name=f"Block {block.index}"
            ))
        
        # Add connections between blocks
        for i in range(len(st.session_state.pos_blocks) - 1):
            fig.add_trace(go.Scatter(
                x=[i, i+1],
                y=[0, 0],
                mode='lines',
                line=dict(width=2, color='black'),
                showlegend=False
            ))
        
        fig.update_layout(
            title="Blockchain Structure",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("This interactive demo shows the basic principles of blockchain consensus mechanisms.")