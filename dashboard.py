import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
from strategy import Strategy, DEFAULT_CONFIG
from ensemble_strategy import EnsembleStrategy
from real_time_trader import RealTimeTrader
from fetch_data import DataFetcher

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'trader' not in st.session_state:
    st.session_state.trader = RealTimeTrader()

if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()

if 'strategy' not in st.session_state:
    st.session_state.strategy = Strategy()

if 'ensemble_strategy' not in st.session_state:
    st.session_state.ensemble_strategy = EnsembleStrategy()

# Sidebar
st.sidebar.title("Trading Controls")

# Strategy selection
strategy_type = st.sidebar.selectbox(
    "Select Strategy",
    ["Standard Strategy", "Ensemble Strategy"]
)

# Target and anchor selection
target_symbol = st.sidebar.selectbox(
    "Target Symbol",
    ["LDO", "AVAX", "SOL", "LINK", "UNI"]
)

anchor_symbols = st.sidebar.multiselect(
    "Anchor Symbols",
    ["BTC", "ETH", "SOL", "BNB", "XRP"],
    default=["BTC", "ETH", "SOL"]
)

# Trading controls
if st.sidebar.button("Start Trading"):
    st.session_state.trader.start(update_interval_minutes=5)
    st.sidebar.success("Trading started!")

if st.sidebar.button("Stop Trading"):
    st.session_state.trader.stop()
    st.sidebar.error("Trading stopped!")

# Main dashboard
st.title("Crypto Trading Dashboard")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Market Overview", "Strategy Performance", "Trade History", "Settings"])

with tab1:
    st.header("Market Overview")
    
    # Get latest data
    try:
        target_data = st.session_state.data_fetcher.get_recent_data(target_symbol, "1h")
        
        if target_data is not None:
            # Create price chart
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
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                height=600,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show latest price and stats
            latest_price = target_data['close'].iloc[-1]
            daily_change = (latest_price / target_data['close'].iloc[-24] - 1) * 100 if len(target_data) > 24 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${latest_price:.4f}", f"{daily_change:.2f}%")
            col2.metric("24h Volume", f"{target_data['volume'].iloc[-24:].sum():,.0f}")
            col3.metric("24h High", f"${target_data['high'].iloc[-24:].max():.4f}")
            
            # Generate signals
            if strategy_type == "Standard Strategy":
                strategy = st.session_state.strategy
            else:
                strategy = st.session_state.ensemble_strategy
                
            # Prepare anchor data
            anchor_data = pd.DataFrame()
            for symbol in anchor_symbols:
                data = st.session_state.data_fetcher.get_recent_data(symbol, "1h")
                if data is not None:
                    anchor_data[f'close_{symbol}'] = data['close']
                    anchor_data[f'volume_{symbol}'] = data['volume']
                    anchor_data['timestamp'] = data['timestamp']
            
            if not anchor_data.empty:
                signals = strategy.generate_signals(target_data, anchor_data)
                latest_signal = signals.iloc[-1]['signal']
                
                # Display signal
                st.subheader("Trading Signal")
                if latest_signal == "BUY":
                    st.success(f"BUY {target_symbol} at ${latest_price:.4f}")
                elif latest_signal == "SELL":
                    st.error(f"SELL {target_symbol} at ${latest_price:.4f}")
                else:
                    st.info(f"HOLD {target_symbol}")
        else:
            st.error(f"No data available for {target_symbol}")
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")

with tab2:
    st.header("Strategy Performance")
    
    # Load performance data
    try:
        if os.path.exists("results/performance_report.json"):
            with open("results/performance_report.json", "r") as f:
                performance = json.load(f)
                
            # Display performance metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{performance['total_return']*100:.2f}%")
            col2.metric("Win Rate", f"{performance['win_rate']*100:.2f}%")
            col3.metric("Avg Profit", f"${performance['avg_profit']:.2f}")
            col4.metric("Max Drawdown", f"{performance['max_drawdown']*100:.2f}%")
            
            # Display equity curve
            if os.path.exists("results/equity_curve.json"):
                with open("results/equity_curve.json", "r") as f:
                    equity_data = json.load(f)
                
                if equity_data:
                    equity_df = pd.DataFrame(equity_data)
                    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity_df['timestamp'],
                        y=equity_df['equity'],
                        mode='lines',
                        name='Equity'
                    ))
                    
                    fig.update_layout(
                        title='Equity Curve',
                        xaxis_title='Date',
                        yaxis_title='Equity ($)',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available yet. Start trading to generate performance metrics.")
    except Exception as e:
        st.error(f"Error loading performance data: {str(e)}")

with tab3:
    st.header("Trade History")
    
    # Load trade history
    try:
        if os.path.exists("results/trade_history.json"):
            with open("results/trade_history.json", "r") as f:
                trades = json.load(f)
                
            if trades:
                # Convert to DataFrame
                trades_df = pd.DataFrame(trades)
                
                # Display trades
                st.dataframe(trades_df)
                
                # Calculate statistics
                buy_trades = trades_df[trades_df['type'] == 'BUY']
                sell_trades = trades_df[trades_df['type'] == 'SELL']
                
                if not sell_trades.empty:
                    total_profit = sell_trades['profit'].sum() if 'profit' in sell_trades else 0
                    avg_roi = sell_trades['roi'].mean() * 100 if 'roi' in sell_trades else 0
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Trades", len(sell_trades))
                    col2.metric("Total Profit", f"${total_profit:.2f}")
                    col3.metric("Average ROI", f"{avg_roi:.2f}%")
            else:
                st.info("No trades executed yet.")
        else:
            st.info("No trade history available yet.")
    except Exception as e:
        st.error(f"Error loading trade history: {str(e)}")

with tab4:
    st.header("Settings")
    
    # Strategy parameters
    st.subheader("Strategy Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.01) / 100
        leverage = st.slider("Leverage", 1.0, 3.0, 1.5, 0.1)
    
    with col2:
        risk_per_trade = st.slider("Risk Per Trade (%)", 0.5, 5.0, 1.0, 0.1)
        max_trades = st.slider("Max Concurrent Trades", 1, 5, 1)
    
    # Save settings
    if st.button("Save Settings"):
        # Update settings in trader
        st.session_state.trader.transaction_cost = transaction_cost
        
        st.success("Settings saved successfully!")
    
    # Train ML model
    st.subheader("Machine Learning Model")
    
    if st.button("Train Ensemble Model"):
        with st.spinner("Training model..."):
            try:
                # Get historical data
                target_data = st.session_state.data_fetcher.get_recent_data(target_symbol, "1h")
                
                # Prepare anchor data
                anchor_data = pd.DataFrame()
                for symbol in anchor_symbols:
                    data = st.session_state.data_fetcher.get_recent_data(symbol, "1h")
                    if data is not None:
                        anchor_data[f'close_{symbol}'] = data['close']
                        anchor_data[f'volume_{symbol}'] = data['volume']
                        anchor_data['timestamp'] = data['timestamp']
                
                if target_data is not None and not anchor_data.empty:
                    # Train model
                    st.session_state.ensemble_strategy.train_model((target_data, anchor_data))
                    st.success("Model trained successfully!")
                else:
                    st.error("Insufficient data for training")
            except Exception as e:
                st.error(f"Error training model: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Developed for Pairwise Alpha Trading Challenge")

# Run the app with: streamlit run dashboard.py