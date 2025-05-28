import streamlit as st
import pandas as pd
import json
from textwrap import indent
from fetch_data import DataFetcher, save_data
import threading
import time
from strategy import Strategy
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config with a modern theme
st.set_page_config(
    page_title="Lunor AI: Strategy Builder 🚀",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton button {
        width: 100%;
        height: 3em;
        font-size: 1.1em;
        margin: 1em 0;
        background-color: #ff4b4b;
        color: white;
    }
    .info-box {
        padding: 1em;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 1em 0;
    }
    .highlight {
        color: #ff4b4b;
        font-weight: bold;
    }
    h1 {
        color: #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize data fetcher
data_fetcher = DataFetcher(update_interval_minutes=60)

# Add this near the top after page config
if 'data_fetcher_running' not in st.session_state:
    st.session_state.data_fetcher_running = False

def start_data_updates():
    """Start background data updates if not already running"""
    if not st.session_state.data_fetcher_running:
        data_fetcher.start_background_updates()
        st.session_state.data_fetcher_running = True

# Add this at the start of the main content
start_data_updates()

# Sidebar navigation
with st.sidebar:
    st.image("lunor-full.png", width=160)
    st.title("Navigation")
    page = st.radio(
        "Choose a section:",
        ["Welcome", "Strategy Builder", "Help & Resources"]
    )
    
    st.divider()
    st.subheader("Data Status")
    if st.session_state.data_fetcher_running:
        st.success("🔄 Auto-updates running")
        if data_fetcher.last_update:
            st.info(f"Last update: {data_fetcher.last_update.strftime('%H:%M:%S')}")
    else:
        st.error("❌ Updates not running")
        if st.button("Start Updates"):
            start_data_updates()

if page == "Welcome":
    st.title("🚀 Welcome to PairWise Alpha Strategy Builder")
    
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
        ### What is PairWise Alpha?
        This is your friendly guide to building profitable crypto trading strategies! 
        We'll help you:
        - 📈 Track relationships between different crypto coins
        - ⏰ Use price movements to predict future changes
        - 💰 Create smart trading rules for better returns
        """)
        
        with st.expander("👉 How does it work?"):
            st.markdown("""
            1. Pick a coin you want to trade
            2. Choose which big coins (BTC/ETH/SOL) to watch
            3. Set up simple "if-then" rules
            4. Get a ready-to-use trading strategy!
            """)
    
    with col2:
        st.info("💡 **Quick Tip**\nStart with simple rules like:\n'Buy when BTC goes up 2%'\n'Sell when ETH drops 3%'")

elif page == "Strategy Builder":
    st.title("🔧 Build Your Strategy")
    
    # Step 1: Target Selection with visual helper
    st.header("Step 1: Choose Your Trading Target", divider="red")
    col1, col2 = st.columns([1,1])
    with col1:
        target_symbol = st.text_input(
            "Which coin do you want to trade?",
            value="LDO",
            help="Popular choices: LDO, BONK, RAY"
        )
    with col2:
        target_timeframe = st.select_slider(
            "How often do you want to trade?",
            options=["1H", "4H", "1D"],
            value="1H",
            help="1H=Hourly, 4H=Every 4 hours, 1D=Daily"
        )

    # Step 2: Anchor Selection with interactive cards
    st.header("Step 2: Select Your Signal Sources", divider="red")
    num_anchors = st.slider("How many coins do you want to watch?", 1, 3, 2)
    
    cols = st.columns(num_anchors)
    anchors = []
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"### Signal Source #{i+1}")
            symbol = st.selectbox(
                "Choose coin:",
                ["BTC", "ETH", "SOL"],
                key=f"a_sym_{i}"
            )
            tf = st.select_slider(
                "Timeframe:",
                ["1H", "4H", "1D"],
                key=f"a_tf_{i}"
            )
            lag = st.number_input(
                "Look back how many candles?",
                0, 48, 4,
                key=f"a_lag_{i}",
                help="Higher = More lag, but might catch trends earlier"
            )
            anchors.append({"symbol": symbol, "timeframe": tf, "lag": lag})

    # Step 3: Trading Rules with visual feedback
    st.header("Step 3: Set Your Trading Rules", divider="red")
    
    tab1, tab2 = st.tabs(["📈 Buy Rules", "📉 Sell Rules"])
    
    with tab1:
        num_buy = st.slider("Number of buy conditions", 1, 3, 2)
        buy_rules = []
        for i in range(num_buy):
            st.markdown(f"### Buy Rule #{i+1}")
            col1, col2 = st.columns([1,1])
            with col1:
                symbol = st.selectbox("When this coin:", ["BTC", "ETH", "SOL"], key=f"b_sym_{i}")
                direction = st.select_slider("Moves:", ["up", "down"], key=f"b_dir_{i}")
            with col2:
                pct = st.slider("By this much %:", -10.0, 10.0, 2.0, key=f"b_pct_{i}")
                tf = st.select_slider("Over this timeframe:", ["1H", "4H", "1D"], key=f"b_tf_{i}")
            buy_rules.append({"symbol": symbol, "timeframe": tf, "change_pct": pct, "direction": direction})

    with tab2:
        num_sell = st.slider("Number of sell conditions", 1, 3, 1)
        sell_rules = []
        for i in range(num_sell):
            st.markdown(f"### Sell Rule #{i+1}")
            col1, col2 = st.columns([1,1])
            with col1:
                symbol = st.selectbox("When this coin:", ["BTC", "ETH", "SOL"], key=f"s_sym_{i}")
                direction = st.select_slider("Moves:", ["up", "down"], key=f"s_dir_{i}")
            with col2:
                pct = st.slider("By this much %:", -10.0, 10.0, -3.0, key=f"s_pct_{i}")
                tf = st.select_slider("Over this timeframe:", ["1H", "4H", "1D"], key=f"s_tf_{i}")
            sell_rules.append({"symbol": symbol, "timeframe": tf, "change_pct": pct, "direction": direction})

elif page == "Help & Resources":
    st.title("📚 Help & Resources")
    
    st.header("Common Questions", divider="red")
    with st.expander("What timeframes can I use?"):
        st.markdown("""
        You can use:
        - 1H (hourly)
        - 4H (4-hour)
        - 1D (daily)
        """)
        
    with st.expander("Which coins can I use as signals?"):
        st.markdown("""
        You can only use:
        - BTC (Bitcoin)
        - ETH (Ethereum) 
        - SOL (Solana)
        
        These are called "anchor" coins because they tend to lead market movements.
        """)
        
    with st.expander("How do I know if my strategy is good?"):
        st.markdown("""
        A good strategy should:
        1. Have clear entry (BUY) and exit (SELL) rules
        2. Use multiple timeframes or conditions
        3. Consider both upward and downward moves
        4. Not be too sensitive to small price changes
        """)

    st.header("Quick Tips", divider="red")
    st.info("""
    💡 Start Simple
    - Use 1-2 anchor coins first
    - Test with hourly (1H) timeframes
    - Look for clear trends (>2% moves)
    """)
    
    st.warning("""
    ⚠️ Common Mistakes
    - Using too many conditions
    - Very small % changes (<1%)
    - Not having clear exit rules
    """)

# Add to session state initialization
if 'strategy' not in st.session_state:
    st.session_state.strategy = Strategy()
    
# Replace the "Generate strategy.py" button section with:
if st.button("🎯 Calculate Signals"):
    try:
        # Create config dictionary
        config = {
            "target": {"symbol": target_symbol, "timeframe": target_timeframe},
            "anchors": anchors,
            "buy_rules": buy_rules,
            "sell_rules": sell_rules
        }
        
        # Initialize strategy with config
        strategy = Strategy(config)
        
        # Load data
        btc_data = pd.read_csv('data/BTC_1H.csv')
        target_data = pd.read_csv(f'data/{target_symbol}_1H.csv')
        eth_data = pd.read_csv('data/ETH_1H.csv') 
        sol_data = pd.read_csv('data/SOL_1H.csv')

        # Prepare anchor data
        anchor_data = pd.DataFrame({'timestamp': btc_data['timestamp']})
        for symbol in ['BTC', 'ETH', 'SOL']:
            data = eval(f"{symbol.lower()}_data")
            anchor_data[f'close_{symbol}'] = data['close']
            anchor_data[f'volume_{symbol}'] = data['volume']

        # Generate signals
        signals = st.session_state.strategy.generate_signals(target_data, anchor_data)
        
        # Calculate statistics
        total = len(signals)
        buys = (signals['signal'] == 'BUY').sum()
        sells = (signals['signal'] == 'SELL').sum()
        holds = (signals['signal'] == 'HOLD').sum()

        # Display results
        col1, col2, col3 = st.columns(3)
        col1.metric("Buy Signals", f"{buys} ({buys/total*100:.1f}%)")
        col2.metric("Sell Signals", f"{sells} ({sells/total*100:.1f}%)")
        col3.metric("Hold Signals", f"{holds} ({holds/total*100:.1f}%)")

        # Plot signals timeline
        signals['timestamp'] = pd.to_datetime(signals['timestamp'])
        
        fig = go.Figure()
        for signal_type in ['BUY', 'SELL', 'HOLD']:
            mask = signals['signal'] == signal_type
            fig.add_trace(go.Scatter(
                x=signals[mask]['timestamp'],
                y=[signal_type] * mask.sum(),
                name=signal_type,
                mode='markers',
                marker=dict(
                    size=10,
                    symbol='circle',
                    color={'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}[signal_type]
                )
            ))
            
        fig.update_layout(
            title="Signal Timeline",
            xaxis_title="Date",
            yaxis_title="Signal Type",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show recent signals
        st.subheader("Recent Signals")
        recent = signals.tail(10).copy()
        recent['timestamp'] = recent['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(recent, hide_index=True)

    except Exception as e:
        st.error(f"Error calculating signals: {str(e)}")

# --- Generate Python ---
def format_list(name, items):
    lines = json.dumps(items, indent=4).replace('true', 'True').replace('false', 'False')
    return f"{name} = {lines}\n"

if st.button("🚀 Generate strategy.py"):
    code = f"""import pandas as pd

# === CONFIGURATION ===
TARGET_COIN = \"{target_symbol.upper()}\"
TIMEFRAME = \"{target_timeframe}\"

ANCHORS = {json.dumps(anchors, indent=4).replace('true', 'True')}

BUY_RULES = {json.dumps(buy_rules, indent=4).replace('true', 'True')}

SELL_RULES = {json.dumps(sell_rules, indent=4).replace('true', 'False')}

# === STRATEGY ENGINE ===
def generate_signals(candles_target: pd.DataFrame, candles_anchor: pd.DataFrame) -> pd.DataFrame:
    try:
        df = candles_target[['timestamp']].copy()
        for anchor in ANCHORS:
            col = f\"close_{{anchor['symbol']}}_{{anchor['timeframe']}}\"
            if col not in candles_anchor.columns:
                raise ValueError(f\"Missing column: {{col}}\")
            df[col] = candles_anchor[col].values

        signals = []
        for i in range(len(df)):
            buy_pass = True
            sell_pass = False

            for rule in BUY_RULES:
                col = f\"close_{{rule['symbol']}}_{{rule['timeframe']}}\"
                if col not in df.columns or pd.isna(df[col].iloc[i]):
                    buy_pass = False
                    break
                change = df[col].pct_change().shift(rule['lag']).iloc[i]
                if pd.isna(change):
                    buy_pass = False
                    break
                if rule['direction'] == 'up' and change <= rule['change_pct'] / 100:
                    buy_pass = False
                    break
                if rule['direction'] == 'down' and change >= rule['change_pct'] / 100:
                    buy_pass = False
                    break

            for rule in SELL_RULES:
                col = f\"close_{{rule['symbol']}}_{{rule['timeframe']}}\"
                if col not in df.columns or pd.isna(df[col].iloc[i]):
                    continue
                change = df[col].pct_change().shift(rule['lag']).iloc[i]
                if pd.isna(change):
                    continue
                if rule['direction'] == 'down' and change <= rule['change_pct'] / 100:
                    sell_pass = True
                if rule['direction'] == 'up' and change >= rule['change_pct'] / 100:
                    sell_pass = True

            if buy_pass:
                signals.append("BUY")
            elif sell_pass:
                signals.append("SELL")
            else:
                signals.append("HOLD")

        df['signal'] = signals
        return df[['timestamp', 'signal']]

    except Exception as e:
        raise RuntimeError(f\"Strategy failed: {{e}}\")
        
def get_coin_metadata() -> dict:
    return {{
        "target": {{"symbol": TARGET_COIN, "timeframe": TIMEFRAME}},
        "anchors": [{{"symbol": a["symbol"], "timeframe": a["timeframe"]}} for a in ANCHORS]
    }}
"""
    st.code(code, language="python")
    st.download_button("📥 Download strategy.py", data=code, file_name="strategy.py", mime="text/x-python")
