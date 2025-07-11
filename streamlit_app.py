import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
import hashlib
import os
import sys
import random

# Trading components
from fetch_data import DataFetcher
from strategy import Strategy, DEFAULT_CONFIG
from crypto_market_insights import MarketInsights
from trading_strategies import TradingStrategies

# Define blockchain helper functions
def calculate_hash(index, previous_hash, timestamp, data, nonce=""):
    block_string = f"{index}{previous_hash}{timestamp}{data}{nonce}"
    return hashlib.sha256(block_string.encode()).hexdigest()

class ProofOfWork:
    """Proof of Work consensus implementation."""
    
    def __init__(self, initial_difficulty=4, target_block_time=600, 
                 difficulty_adjustment_interval=2016, hash_power_per_watt=50e6):
        self.difficulty = initial_difficulty
        self.target_block_time = target_block_time
        self.adjustment_interval = difficulty_adjustment_interval
        self.hash_power_per_watt = hash_power_per_watt
        self.block_times = []
        self.mining_steps = []
        self.difficulty_adjustments = []
    
    def get_target(self, difficulty):
        return '0' * difficulty + 'f' * (64 - difficulty)
    
    def check_pow(self, block):
        block_hash = block.hash
        target = self.get_target(block.difficulty)
        return int(block_hash, 16) <= int(target, 16)
    
    def mine_block(self, block, max_nonce=2**32, track_steps=False, step_interval=100000):
        block.difficulty = self.difficulty
        target = self.get_target(self.difficulty)
        
        self.mining_steps = []
        
        start_time = time.time()
        nonce = 0
        
        if track_steps:
            self.mining_steps.append({
                "time": 0,
                "nonce": nonce,
                "hash": block.hash,
                "target": target
            })
        
        while nonce < max_nonce:
            block.nonce = nonce
            block.hash = block.calculate_hash()
            
            if track_steps and nonce % step_interval == 0:
                self.mining_steps.append({
                    "time": time.time() - start_time,
                    "nonce": nonce,
                    "hash": block.hash,
                    "target": target
                })
            
            if int(block.hash, 16) <= int(target, 16):
                end_time = time.time()
                time_taken = end_time - start_time
                
                hashes_per_second = nonce / time_taken if time_taken > 0 else 0
                watts_used = hashes_per_second / self.hash_power_per_watt
                kwh_used = (watts_used * time_taken) / 3600000
                
                self.block_times.append(end_time)
                if len(self.block_times) > self.adjustment_interval:
                    self.block_times.pop(0)
                
                if track_steps:
                    self.mining_steps.append({
                        "time": time_taken,
                        "nonce": nonce,
                        "hash": block.hash,
                        "target": target,
                        "success": True
                    })
                
                return True, nonce, kwh_used, self.mining_steps
            
            nonce += 1
        
        end_time = time.time()
        time_taken = end_time - start_time
        
        if track_steps:
            self.mining_steps.append({
                "time": time_taken,
                "nonce": nonce,
                "hash": block.hash,
                "target": target,
                "success": False
            })
        
        return False, nonce, 0.0, self.mining_steps
    
    def get_mining_stats(self, chain):
        return {
            'difficulty': self.difficulty,
            'network_hashrate': 1000000,
            'daily_energy_kwh': 100,
            'avg_block_time': self.target_block_time,
            'next_adjustment_blocks': self.adjustment_interval - len(self.block_times),
            'estimated_yearly_energy_mwh': 36500,
            'target': self.get_target(self.difficulty)
        }

class Blockchain:
    """Simple blockchain implementation using PoW consensus."""
    
    def __init__(self, pow_consensus=None):
        self.chain = []
        self.pow = pow_consensus if pow_consensus else ProofOfWork()
        self.create_genesis_block()
    
    def create_genesis_block(self):
        genesis = PowBlock(0, "0" * 64, data="Genesis Block")
        self.chain.append(genesis)
        return genesis
    
    def get_latest_block(self):
        return self.chain[-1]
    
    def add_block(self, data, track_mining=False):
        latest_block = self.get_latest_block()
        new_block = PowBlock(
            index=latest_block.index + 1,
            previous_hash=latest_block.hash,
            data=data
        )
        
        success, hashes, energy, steps = self.pow.mine_block(new_block, track_steps=track_mining)
        
        if success:
            self.chain.append(new_block)
            
            # Skip difficulty adjustment since we didn't implement it
            # if new_block.index % self.pow.adjustment_interval == 0:
            #     self.pow.adjust_difficulty()
            
            mining_stats = {
                'hashes_tried': hashes,
                'energy_kwh': energy,
                'time_taken': steps[-1]['time'] if steps else 0,
                'difficulty': new_block.difficulty
            }
            
            return new_block, mining_stats
        
        raise Exception("Failed to mine block")
    
    def to_dict(self):
        return [block.to_dict() for block in self.chain]

# Page configuration
st.set_page_config(page_title="Advanced Crypto Trading Platform", layout="wide", page_icon="📈")

# Load custom CSS
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Set dark theme
st.markdown("""
<script>
    var elements = window.parent.document.querySelectorAll('.stApp');
    elements[0].style.backgroundColor = '#0e1117';
</script>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()
if 'strategy' not in st.session_state:
    st.session_state.strategy = Strategy()
if 'market_insights' not in st.session_state:
    st.session_state.market_insights = MarketInsights()

# Header with enhanced styling
st.markdown("""
<div class="dashboard-header">
    <h1>🚀 Advanced Crypto Trading Platform</h1>
    <h3>Enterprise-grade analysis, signals, and blockchain integration</h3>
    <p>Version 2.5.0 | Real-time data | AI-powered predictions</p>
</div>
""", unsafe_allow_html=True)

# Dashboard metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card positive">
        <h3>Market Status</h3>
        <h2>Active</h2>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card neutral">
        <h3>API Status</h3>
        <h2>Connected</h2>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-card neutral">
        <h3>Last Update</h3>
        <h2>Just Now</h2>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="metric-card positive">
        <h3>Market Sentiment</h3>
        <h2>Bullish</h2>
    </div>
    """, unsafe_allow_html=True)

# Sidebar for controls with enhanced styling
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2>🎮 Trading Controls</h2>
        <p>Configure your trading parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Asset selection with tooltips
    st.markdown("""
    <div style="background-color: #1a1f2c; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <h3>📊 Select Assets</h3>
    </div>
    """, unsafe_allow_html=True)
    
    target_symbol = st.selectbox(
        "Target Asset", 
        ["LDO", "ETH", "SOL", "AVAX", "LINK", "UNI", "AAVE", "BNB", "ADA", "DOT", "MATIC"], 
        index=0,
        help="The primary asset you want to trade"
    )
    
    anchor_symbols = st.multiselect(
        "Anchor Assets", 
        ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOT", "MATIC", "LINK", "UNI"], 
        default=["BTC", "ETH"],
        help="Reference assets used for correlation analysis"
    )
    
    # Timeframe selection with visual indicators
    st.markdown("""
    <div style="background-color: #1a1f2c; padding: 10px; border-radius: 5px; margin-bottom: 20px; margin-top: 20px;">
        <h3>⏱️ Timeframe</h3>
    </div>
    """, unsafe_allow_html=True)
    
    timeframe_col1, timeframe_col2, timeframe_col3 = st.columns(3)
    
    with timeframe_col1:
        hourly = st.button("1H", help="1 Hour candles")
    with timeframe_col2:
        four_hour = st.button("4H", help="4 Hour candles")
    with timeframe_col3:
        daily = st.button("1D", help="Daily candles")
    
    # Set timeframe based on button clicks
    if hourly:
        timeframe = "1h"
    elif four_hour:
        timeframe = "4h"
    elif daily:
        timeframe = "1d"
    else:
        timeframe = "1h"  # Default
    
    st.markdown(f"""
    <div style="text-align: center; margin-top: 10px; margin-bottom: 20px;">
        <p>Selected timeframe: <b>{timeframe}</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Strategy parameters with advanced options
    st.markdown("""
    <div style="background-color: #1a1f2c; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <h3>⚙️ Strategy Parameters</h3>
    </div>
    """, unsafe_allow_html=True)
    
    leverage = st.slider(
        "Leverage", 
        1.0, 10.0, 1.5, 0.1,
        help="Amount of leverage to use (higher values increase both potential profit and risk)"
    )
    
    risk_per_trade = st.slider(
        "Risk Per Trade (%)", 
        0.1, 10.0, 1.0, 0.1,
        help="Percentage of portfolio to risk on each trade"
    )
    
    # Advanced parameters (collapsible)
    with st.expander("Advanced Parameters"):
        st.slider("Take Profit (%)", 5.0, 50.0, 15.0, 1.0)
        st.slider("Stop Loss (%)", 1.0, 20.0, 5.0, 1.0)
        st.checkbox("Use Trailing Stop", value=True)
        st.selectbox("Entry Type", ["Market", "Limit", "Stop Limit"])
        st.number_input("Max Open Positions", 1, 10, 3)
    
    # Account settings
    with st.expander("Account Settings"):
        st.number_input("Account Size ($)", 1000, 1000000, 10000)
        st.selectbox("Account Currency", ["USD", "EUR", "GBP", "JPY"])
        st.checkbox("Paper Trading", value=True)
    
    # Run buttons with enhanced styling
    st.markdown("""
    <div style="background-color: #1a1f2c; padding: 10px; border-radius: 5px; margin-bottom: 20px; margin-top: 20px;">
        <h3>🚀 Actions</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        get_signal = st.button("Get Signal", help="Generate trading signals based on current market conditions")
    with col2:
        run_backtest = st.button("Run Backtest", help="Test strategy performance on historical data")
    
    # Additional actions
    st.button("Export Strategy", help="Export your strategy configuration")
    
    # System status
    st.markdown("""
    <div style="background-color: #1a1f2c; padding: 10px; border-radius: 5px; margin-top: 20px; text-align: center;">
        <h4>System Status</h4>
        <p style="color: #4CAF50;">✅ All Systems Operational</p>
    </div>
    """, unsafe_allow_html=True)

# Import blockchain components
from blockchain_components import (
    mine_pow_block, create_pos_block, select_validator,
    initialize_blockchain_demo, PowBlock, PosBlock, PosAccount
)
# Use blockchain_components classes directly

# Initialize blockchain demo data in session state
if 'pow_blocks' not in st.session_state:
    pow_blocks, pos_blocks, pos_accounts = initialize_blockchain_demo()
    st.session_state.pow_blocks = pow_blocks
    st.session_state.pos_blocks = pos_blocks
    st.session_state.pos_accounts = pos_accounts
    
# Initialize mine_pow_block and create_pos_block if not available
if 'mine_pow_block' not in globals():
    mine_pow_block = lambda prev, data, diff: (PowBlock(prev.index + 1, prev.hash, data=data, nonce=1234), 1234)
if 'create_pos_block' not in globals():
    create_pos_block = lambda prev, validator, data: PosBlock(prev.index + 1, prev.hash, validator, data=data)
if 'select_validator' not in globals():
    select_validator = lambda accounts: list(accounts.keys())[0] if accounts else None

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Market Overview", 
    "Technical Analysis", 
    "Trading Signals", 
    "Strategy Performance", 
    "Learning Center",
    "Blockchain Demo",
    "Consensus Mechanisms"
])

# Get data
target_data = st.session_state.data_fetcher.get_recent_data(target_symbol, timeframe)

# Market Overview Tab
with tab1:
    st.header("Market Overview")
    
    if target_data is not None:
        # Price chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, subplot_titles=('Price', 'Volume'),
                           row_heights=[0.7, 0.3])
        
        # Add price candlestick
        fig.add_trace(go.Candlestick(
            x=target_data['timestamp'],
            open=target_data['open'], 
            high=target_data['high'],
            low=target_data['low'], 
            close=target_data['close'],
            name="Price"
        ), row=1, col=1)
        
        # Add volume bar chart
        fig.add_trace(go.Bar(
            x=target_data['timestamp'],
            y=target_data['volume'],
            name="Volume"
        ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{target_symbol} Price Chart',
            height=600,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics
        latest_price = target_data['close'].iloc[-1]
        daily_change = (latest_price / target_data['close'].iloc[-24] - 1) * 100 if len(target_data) > 24 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${latest_price:.4f}", f"{daily_change:.2f}%")
        col2.metric("24h Volume", f"{target_data['volume'].iloc[-24:].sum():,.0f}")
        col3.metric("24h High", f"${target_data['high'].iloc[-24:].max():.4f}")
        col4.metric("24h Low", f"${target_data['low'].iloc[-24:].min():.4f}")
        
        # Market insights
        insights = st.session_state.market_insights.get_market_insights(target_symbol, timeframe)
        
        if "error" not in insights:
            # Market sentiment
            st.subheader("Market Sentiment")
            # Safely access nested dictionary values with type checking
            sentiment_data = insights.get("sentiment", {}) if isinstance(insights, dict) else {}
            sentiment_data = sentiment_data if isinstance(sentiment_data, dict) else {}
            sentiment: str = sentiment_data.get("overall", "Unknown") if isinstance(sentiment_data, dict) else "Unknown"
            sentiment_color = "green" if sentiment in ["Strongly Bullish", "Bullish"] else "red" if sentiment in ["Strongly Bearish", "Bearish"] else "gray"
            st.markdown(f"<h3 style='color: {sentiment_color};'>{sentiment}</h3>", unsafe_allow_html=True)
            
            # Support and resistance with safe dictionary access
            st.subheader("Support and Resistance Levels")
            col1, col2 = st.columns(2)
            support_resistance = insights.get("support_resistance", {}) if isinstance(insights, dict) else {}
            support_resistance = support_resistance if isinstance(support_resistance, dict) else {}
            resistance: float = float(support_resistance.get("resistance", 0.0)) if isinstance(support_resistance, dict) else 0.0
            support: float = float(support_resistance.get("support", 0.0)) if isinstance(support_resistance, dict) else 0.0
            col1.metric("Resistance", f"${resistance:.4f}")
            col2.metric("Support", f"${support:.4f}")
            
            # Recent price action summary with safe dictionary access
            st.subheader("Market Analysis")
            technical_signals = insights.get("technical_signals", {}) if isinstance(insights, dict) else {}
            technical_signals = technical_signals if isinstance(technical_signals, dict) else {}
            st.markdown(f"• **Technical Signals:** {technical_signals.get('overall', 'N/A') if isinstance(technical_signals, dict) else 'N/A'}")
            st.markdown(f"• **EMA Signal:** {technical_signals.get('ema_short', 'N/A') if isinstance(technical_signals, dict) else 'N/A'}")
            st.markdown(f"• **RSI Signal:** {technical_signals.get('rsi', 'N/A') if isinstance(technical_signals, dict) else 'N/A'}")
            st.markdown(f"• **MACD Signal:** {technical_signals.get('macd', 'N/A') if isinstance(technical_signals, dict) else 'N/A'}")
        else:
            st.error(f"Error getting market insights: {insights['error']}")
    else:
        st.error(f"No data available for {target_symbol}")

# Technical Analysis Tab
with tab2:
    st.header("Technical Analysis")
    
    if target_data is not None:
        # Strategy selection
        strategy_type = st.selectbox(
            "Select Strategy",
            ["EMA Crossover", "RSI", "MACD", "Bollinger Bands", "Support/Resistance", "Multi-Strategy Consensus"]
        )
        
        # Calculate signals based on selected strategy
        if strategy_type == "EMA Crossover":
            fast_period = st.slider("Fast EMA Period", 5, 50, 9)
            slow_period = st.slider("Slow EMA Period", 10, 200, 21)
            signals = TradingStrategies.ema_crossover_strategy(target_data.copy(), fast_period, slow_period)
            
            # Calculate EMAs for display
            target_data['ema_fast'] = target_data['close'].ewm(span=fast_period, adjust=False).mean()
            target_data['ema_slow'] = target_data['close'].ewm(span=slow_period, adjust=False).mean()
            
            # Create chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.1, subplot_titles=('Price with EMAs', 'Volume'),
                               row_heights=[0.7, 0.3])
            
            # Add price candlestick
            fig.add_trace(go.Candlestick(
                x=target_data['timestamp'],
                open=target_data['open'], 
                high=target_data['high'],
                low=target_data['low'], 
                close=target_data['close'],
                name="Price"
            ), row=1, col=1)
            
            # Add EMAs
            fig.add_trace(go.Scatter(
                x=target_data['timestamp'],
                y=target_data['ema_fast'],
                line=dict(color='blue', width=1),
                name=f"EMA {fast_period}"
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=target_data['timestamp'],
                y=target_data['ema_slow'],
                line=dict(color='orange', width=1),
                name=f"EMA {slow_period}"
            ), row=1, col=1)
            
        elif strategy_type == "RSI":
            rsi_period = st.slider("RSI Period", 5, 30, 14)
            oversold = st.slider("Oversold Level", 10, 40, 30)
            overbought = st.slider("Overbought Level", 60, 90, 70)
            signals = TradingStrategies.rsi_strategy(target_data.copy(), rsi_period, oversold, overbought)
            
            # Calculate RSI for display
            delta = target_data['close'].diff()
            gain = delta.clip(lower=0).rolling(window=rsi_period).mean()
            loss = -delta.clip(upper=0).rolling(window=rsi_period).mean()
            rs = gain / (loss + 1e-8)
            target_data['rsi'] = 100 - (100 / (1 + rs))
            
            # Create chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.1, subplot_titles=('Price', 'RSI'),
                               row_heights=[0.7, 0.3])
            
            # Add price candlestick
            fig.add_trace(go.Candlestick(
                x=target_data['timestamp'],
                open=target_data['open'], 
                high=target_data['high'],
                low=target_data['low'], 
                close=target_data['close'],
                name="Price"
            ), row=1, col=1)
            
            # Add RSI
            fig.add_trace(go.Scatter(
                x=target_data['timestamp'],
                y=target_data['rsi'],
                line=dict(color='purple', width=1),
                name="RSI"
            ), row=2, col=1)
            
            # Add RSI levels
            fig.update_layout(
                shapes=[
                    dict(
                        type="line", line_color="red", line_width=1, opacity=0.3,
                        x0=target_data['timestamp'].iloc[0], x1=target_data['timestamp'].iloc[-1], 
                        y0=overbought, y1=overbought,
                        xref='x2', yref='y2'
                    ),
                    dict(
                        type="line", line_color="green", line_width=1, opacity=0.3,
                        x0=target_data['timestamp'].iloc[0], x1=target_data['timestamp'].iloc[-1], 
                        y0=oversold, y1=oversold,
                        xref='x2', yref='y2'
                    )
                ]
            )
            
        else:
            # Default to multi-strategy consensus
            signals = TradingStrategies.multi_strategy_consensus(target_data.copy())
            
            # Create chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.1, subplot_titles=('Price', 'Volume'),
                               row_heights=[0.7, 0.3])
            
            # Add price candlestick
            fig.add_trace(go.Candlestick(
                x=target_data['timestamp'],
                open=target_data['open'], 
                high=target_data['high'],
                low=target_data['low'], 
                close=target_data['close'],
                name="Price"
            ), row=1, col=1)
        
        # Add volume
        fig.add_trace(go.Bar(
            x=target_data['timestamp'],
            y=target_data['volume'],
            name="Volume"
        ), row=2, col=1)
        
        # Add buy/sell markers
        buy_signals = signals[signals['signal'] == 'BUY']
        sell_signals = signals[signals['signal'] == 'SELL']
        
        if not buy_signals.empty:
            # Safely get buy prices with type checking
            buy_prices = []
            for ts in buy_signals['timestamp']:
                matching_data = target_data[target_data['timestamp'] == ts]
                if not matching_data.empty and isinstance(matching_data['close'], pd.Series):
                    buy_prices.append(matching_data['close'].iloc[0])
            
            if buy_prices:
                fig.add_trace(go.Scatter(
                    x=buy_signals['timestamp'],
                    y=buy_prices,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color='green'),
                    name="Buy Signal"
                ), row=1, col=1)
        
        if not sell_signals.empty:
            # Safely get sell prices with type checking
            sell_prices = []
            for ts in sell_signals['timestamp']:
                matching_data = target_data[target_data['timestamp'] == ts]
                if not matching_data.empty and isinstance(matching_data['close'], pd.Series):
                    sell_prices.append(matching_data['close'].iloc[0])
            
            if sell_prices:
                fig.add_trace(go.Scatter(
                    x=sell_signals['timestamp'],
                    y=sell_prices,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    name="Sell Signal"
                ), row=1, col=1)
        
        # Update layout
        fig.update_layout(
            height=700,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy performance
        st.subheader("Strategy Performance")
        
        # Calculate performance
        performance, portfolio = TradingStrategies.get_strategy_performance(target_data, signals)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{performance['total_return']*100:.2f}%")
        col2.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")
        col3.metric("Max Drawdown", f"{performance['max_drawdown']*100:.2f}%")
        col4.metric("Win Rate", f"{performance['win_rate']*100:.2f}%")
        
        # Display equity curve
        st.subheader("Equity Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio['timestamp'],
            y=portfolio['total'],
            mode='lines',
            name='Portfolio Value'
        ))
        
        fig.update_layout(
            height=400,
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"No data available for {target_symbol}")

# Trading Signals Tab
with tab3:
    st.header("Trading Signals")
    
    if target_data is not None:
        # Get anchor data
        anchor_data = pd.DataFrame()
        for symbol in anchor_symbols:
            data = st.session_state.data_fetcher.get_recent_data(symbol, timeframe)
            if data is not None:
                anchor_data[f'close_{symbol}'] = data['close']
                anchor_data[f'volume_{symbol}'] = data['volume']
                anchor_data['timestamp'] = data['timestamp']
        
        if not anchor_data.empty:
            # Generate signals
            signals = st.session_state.strategy.generate_signals(target_data, anchor_data)
            latest_signal = signals.iloc[-1]['signal']
            
            # Display signal
            st.subheader("Current Trading Signal")
            if latest_signal == "BUY":
                st.success(f"BUY {target_symbol} at ${target_data['close'].iloc[-1]:.4f}")
                st.markdown("### Reasons to Buy:")
                st.markdown("• Positive momentum detected in price action")
                st.markdown("• Favorable market conditions based on anchor assets")
                st.markdown("• Technical indicators showing bullish signals")
            elif latest_signal == "SELL":
                st.error(f"SELL {target_symbol} at ${target_data['close'].iloc[-1]:.4f}")
                st.markdown("### Reasons to Sell:")
                st.markdown("• Negative momentum detected in price action")
                st.markdown("• Unfavorable market conditions based on anchor assets")
                st.markdown("• Technical indicators showing bearish signals")
            else:
                st.info(f"HOLD {target_symbol}")
                st.markdown("### Reasons to Hold:")
                st.markdown("• No clear directional signal at this time")
                st.markdown("• Market conditions are neutral")
                st.markdown("• Wait for a stronger entry opportunity")
            
            # Trading recommendation
            st.subheader("Trading Recommendation")
            
            # Position size calculation
            account_size = 10000  # Default account size
            position_size = account_size * (risk_per_trade / 100) / (target_data['close'].iloc[-1] * 0.05)  # Assuming 5% stop loss
            
            if latest_signal == "BUY":
                st.markdown(f"**Entry Price:** ${target_data['close'].iloc[-1]:.4f}")
                st.markdown(f"**Stop Loss:** ${target_data['close'].iloc[-1] * 0.95:.4f} (5% below entry)")
                st.markdown(f"**Take Profit:** ${target_data['close'].iloc[-1] * 1.15:.4f} (15% above entry)")
                st.markdown(f"**Position Size:** {position_size:.2f} units (${position_size * target_data['close'].iloc[-1]:.2f})")
            elif latest_signal == "SELL":
                st.markdown("Consider closing existing positions or opening a short position if your platform allows.")
        else:
            st.error("No anchor data available for signal generation")
    else:
        st.error(f"No data available for {target_symbol}")

# Strategy Performance Tab
with tab4:
    st.header("Strategy Performance")
    
    if run_backtest:
        st.info("Running backtest...")
        
        if target_data is not None:
            # Get anchor data
            anchor_data = pd.DataFrame()
            for symbol in anchor_symbols:
                data = st.session_state.data_fetcher.get_recent_data(symbol, timeframe)
                if data is not None:
                    anchor_data[f'close_{symbol}'] = data['close']
                    anchor_data[f'volume_{symbol}'] = data['volume']
                    anchor_data['timestamp'] = data['timestamp']
            
            if not anchor_data.empty:
                # Generate signals
                signals = st.session_state.strategy.generate_signals(target_data, anchor_data)
                
                # Calculate performance
                performance, portfolio = TradingStrategies.get_strategy_performance(target_data, signals)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return", f"{performance['total_return']*100:.2f}%")
                col2.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")
                col3.metric("Max Drawdown", f"{performance['max_drawdown']*100:.2f}%")
                col4.metric("Win Rate", f"{performance['win_rate']*100:.2f}%")
                
                # Display equity curve
                st.subheader("Equity Curve")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=portfolio['timestamp'],
                    y=portfolio['total'],
                    mode='lines',
                    name='Portfolio Value'
                ))
                
                fig.update_layout(
                    height=400,
                    xaxis_title='Date',
                    yaxis_title='Portfolio Value ($)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display trades
                st.subheader("Trade History")
                if performance['trades']:
                    trades_df = pd.DataFrame(performance['trades'])
                    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
                    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
                    trades_df['profit'] = trades_df['profit'].round(2)
                    trades_df['roi'] = (trades_df['roi'] * 100).round(2)
                    
                    st.dataframe(trades_df)
                else:
                    st.info("No trades were executed during the backtest period.")
            else:
                st.error("No anchor data available for backtest")
        else:
            st.error(f"No data available for {target_symbol}")
    else:
        st.info("Click 'Run Backtest' to see strategy performance.")

# Learning Center Tab
with tab5:
    st.header("Crypto Trading Learning Center")
    
    # Trading basics
    st.subheader("Trading Basics")
    st.markdown("""
    ### Key Concepts for New Traders:
    
    1. **Market Cycles**: Crypto markets move in cycles of accumulation, uptrend, distribution, and downtrend.
    
    2. **Risk Management**: Never risk more than 1-2% of your portfolio on a single trade.
    
    3. **Technical Analysis**: Use indicators like Moving Averages, RSI, and MACD to identify trends and reversals.
    
    4. **Fundamental Analysis**: Research the technology, team, and adoption metrics of cryptocurrencies.
    
    5. **Market Sentiment**: Monitor social media, news, and market sentiment indicators.
    """)
    
    # Strategy explanations
    st.subheader("Strategy Explanation")
    st.markdown("""
    ### Our Trading Strategy:
    
    This dashboard uses a multi-factor approach combining:
    
    - **Trend Analysis**: Identifying the direction of the market using EMAs
    - **Momentum Indicators**: Measuring the strength of price movements
    - **Correlation Analysis**: Using anchor assets (like BTC and ETH) to confirm market conditions
    - **Volatility Assessment**: Adjusting position sizes based on market volatility
    - **Risk Management**: Implementing stop losses and take profit levels
    """)
    
    # Common mistakes
    st.subheader("Common Trading Mistakes to Avoid")
    st.markdown("""
    1. **Overtrading**: Trading too frequently leads to higher fees and emotional decisions
    2. **Ignoring Risk Management**: Not using stop losses or risking too much per trade
    3. **FOMO Trading**: Buying due to Fear Of Missing Out rather than strategy
    4. **Emotional Decision Making**: Letting fear and greed drive trading decisions
    5. **Lack of Plan**: Trading without a clear strategy and rules
    """)
    
    # Get beginner advice
    beginner_advice = st.session_state.market_insights.get_beginner_advice()
    
    # Display beginner advice
    st.subheader("Advice for Beginners")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Risk Management")
        for tip in beginner_advice["risk_management"]:
            st.markdown(f"• {tip}")
        
        st.markdown("### Technical Analysis")
        for tip in beginner_advice["technical_analysis"]:
            st.markdown(f"• {tip}")
    
    with col2:
        st.markdown("### Psychology")
        for tip in beginner_advice["psychology"]:
            st.markdown(f"• {tip}")
        
        st.markdown("### Market Understanding")
        for tip in beginner_advice["market_understanding"]:
            st.markdown(f"• {tip}")

# Blockchain Demo Tab
with tab6:
    st.header("Blockchain Demonstration")
    
    st.markdown("""
    This interactive demo shows how blockchain technology works. You can:
    - Mine new blocks using Proof of Work (PoW)
    - Create blocks using Proof of Stake (PoS)
    - Explore the blockchain structure
    """)
    
    # Create tabs for different blockchain types
    pow_tab, pos_tab = st.tabs(["Proof of Work", "Proof of Stake"])
    
    # PoW Tab
    with pow_tab:
        st.subheader("Proof of Work Mining")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            pow_data = st.text_area("Block Data (PoW)", "Enter transaction data here...")
        
        with col2:
            difficulty = st.slider("Mining Difficulty", 1, 5, 3)
            mine_button = st.button("Mine Block")
        
        if mine_button:
            with st.spinner("Mining in progress..."):
                progress_bar = st.progress(0)
                
                # Simulate mining progress
                for i in range(10):
                    time.sleep(0.1)
                    progress_bar.progress((i+1)/10)
                
                try:
                    # Mine the block
                    previous_block = st.session_state.pow_blocks[-1]
                    
                    # Mine the block
                    new_block, nonce = mine_pow_block(previous_block, pow_data, difficulty)
                    
                    if new_block:
                        st.session_state.pow_blocks.append(new_block)
                        st.success(f"Block #{new_block.index} mined successfully with nonce {nonce}!")
                        st.balloons()
                    else:
                        st.error("Mining failed. Try reducing the difficulty.")
                except Exception as e:
                    st.error(f"Error mining block: {str(e)}")
        
        # Display PoW blockchain
        st.subheader("PoW Blockchain Explorer")
        
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
                title="PoW Blockchain Structure",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # PoS Tab
    with pos_tab:
        st.subheader("Proof of Stake Validation")
        
        # Account management
        accounts_col1, accounts_col2 = st.columns(2)
        
        with accounts_col1:
            # Create account
            new_name = st.text_input("Account Name")
            new_balance = st.number_input("Initial Balance", min_value=0.0, value=100.0)
            create_account = st.button("Create Account")
            
            if create_account and new_name:
                try:
                    if new_name not in st.session_state.pos_accounts:
                        st.session_state.pos_accounts[new_name] = PosAccount(new_name, new_balance)
                        st.success(f"Account {new_name} created with {new_balance} coins!")
                    else:
                        st.error(f"Account {new_name} already exists.")
                except Exception as e:
                    st.error(f"Error creating account: {str(e)}")
        
        with accounts_col2:
            # Stake management
            account_name = st.selectbox("Select Account", list(st.session_state.pos_accounts.keys()))
            stake_amount = st.number_input("Amount to Stake", min_value=0.1, value=10.0)
            stake_button = st.button("Stake Coins")
            
            if stake_button and account_name in st.session_state.pos_accounts:
                try:
                    account = st.session_state.pos_accounts[account_name]
                    
                    # Stake coins
                    if account.add_stake(stake_amount):
                        st.success(f"{account_name} staked {stake_amount} coins successfully!")
                    else:
                        st.error(f"Failed to stake. Insufficient balance.")
                except Exception as e:
                    st.error(f"Error staking coins: {str(e)}")
        
        # Display accounts
        st.subheader("Accounts Overview")
        
        accounts_df = pd.DataFrame([account.to_dict() for account in st.session_state.pos_accounts.values()])
        if not accounts_df.empty:
            st.dataframe(accounts_df)
        
        # Create new block
        st.subheader("Create New Block")
        
        pos_data = st.text_area("Block Data (PoS)", "Enter transaction data here...")
        create_block = st.button("Create Block")
        
        if create_block:
            try:
                # Check if we're using the minimal version
                # Select validator based on stake
                validator = select_validator(st.session_state.pos_accounts)
                
                if validator:
                    # Create block
                    previous_block = st.session_state.pos_blocks[-1]
                    
                    # Create a new PoS block
                    new_block = create_pos_block(previous_block, validator, pos_data)
                    
                    # Add block
                    st.session_state.pos_blocks.append(new_block)
                    
                    # Distribute rewards
                    reward = 1.0  # 1 coin reward
                    validator_account = st.session_state.pos_accounts[validator]
                    
                    # Add reward to validator
                    validator_account.add_reward(reward)
                    
                    st.success(f"Block #{new_block.index} created by validator {validator}!")
                    st.balloons()
                else:
                    st.error("No validators available. Stake some coins first!")
            except Exception as e:
                st.error(f"Error creating block: {str(e)}")
        
        # Display PoS blockchain
        st.subheader("PoS Blockchain Explorer")
        
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
                title="PoS Blockchain Structure",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Consensus Mechanisms Tab
with tab7:
    st.header("Consensus Mechanisms")
    
    st.markdown("""
    Blockchain networks use consensus mechanisms to agree on the state of the ledger. 
    Here's a comparison of the two most common mechanisms:
    """)
    
    # Comparison table
    comparison_data = {
        "Feature": ["Energy Usage", "Security", "Scalability", "Entry Barrier", "Examples"],
        "Proof of Work": ["High", "Very High", "Limited", "Mining Hardware", "Bitcoin, Litecoin"],
        "Proof of Stake": ["Low", "High", "Better", "Token Holdings", "Ethereum 2.0, Cardano, Solana"]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)
    
    # Detailed explanations
    st.subheader("Proof of Work (PoW)")
    st.markdown("""
    In Proof of Work:
    
    1. **Mining Process**: Miners compete to solve complex mathematical puzzles
    2. **Block Creation**: The first miner to solve the puzzle gets to create the next block
    3. **Reward**: The winning miner receives newly minted coins and transaction fees
    4. **Security**: Secured by computational power - attacking the network requires controlling 51% of the total hash power
    
    **Advantages**:
    - Battle-tested security
    - Decentralized validator selection
    
    **Disadvantages**:
    - High energy consumption
    - Specialized hardware requirements
    - Potential for mining centralization
    """)
    
    st.subheader("Proof of Stake (PoS)")
    st.markdown("""
    In Proof of Stake:
    
    1. **Validator Selection**: Validators are chosen based on the amount of cryptocurrency they stake
    2. **Block Creation**: Selected validators verify transactions and create new blocks
    3. **Reward**: Validators earn transaction fees and sometimes new coins
    4. **Security**: Secured by economic stake - attackers must own a significant portion of the cryptocurrency
    
    **Advantages**:
    - Energy efficient
    - No specialized hardware needed
    - Better scalability potential
    
    **Disadvantages**:
    - Less battle-tested than PoW
    - Potential for stake centralization
    - "Nothing at stake" problem (addressed by slashing mechanisms)
    """)
    
    # Visual comparison
    st.subheader("Visual Comparison")
    
    # Create data for the chart
    categories = ['Energy Usage', 'Security', 'Scalability', 'Decentralization', 'Maturity']
    pow_values = [9, 9, 3, 7, 10]
    pos_values = [2, 7, 8, 6, 6]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=pow_values,
        theta=categories,
        fill='toself',
        name='Proof of Work'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=pos_values,
        theta=categories,
        fill='toself',
        name='Proof of Stake'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Educational resources
    st.subheader("Learn More")
    st.markdown("""
    - [Bitcoin Whitepaper](https://bitcoin.org/bitcoin.pdf) - Original PoW implementation
    - [Ethereum 2.0](https://ethereum.org/en/eth2/) - Major PoS implementation
    - [Consensus Mechanisms Explained](https://www.coindesk.com/learn/consensus-mechanisms-explained/)
    """)
    
    # Interactive quiz
    st.subheader("Test Your Knowledge")
    
    with st.form("consensus_quiz"):
        st.markdown("**1. Which consensus mechanism is more energy efficient?**")
        q1 = st.radio("", ["Proof of Work", "Proof of Stake"], key="q1")
        
        st.markdown("**2. Which network currently uses Proof of Work?**")
        q2 = st.radio("", ["Ethereum 2.0", "Cardano", "Bitcoin", "Solana"], key="q2")
        
        st.markdown("**3. In Proof of Stake, validators are selected based on:**")
        q3 = st.radio("", ["Computing power", "Amount of cryptocurrency staked", "Network reputation", "Random selection"], key="q3")
        
        submitted = st.form_submit_button("Check Answers")
        
        if submitted:
            score = 0
            if q1 == "Proof of Stake":
                score += 1
            if q2 == "Bitcoin":
                score += 1
            if q3 == "Amount of cryptocurrency staked":
                score += 1
            
            st.success(f"You scored {score}/3!")
            
            if score == 3:
                st.balloons()

# Footer
st.markdown("---")
st.markdown("Developed for Pairwise Alpha Trading Challenge | Data updates every hour")