import pandas as pd
import numpy as np
import os

# Trading parameters and configuration
TIMEFRAME_MAP = {
    "1H": "1H",
    "4H": "4H", 
    "1D": "1D"
}

DEFAULT_CONFIG = {
    "target": {"symbol": "LDO", "timeframe": "1H"},
    "anchors": [
        {"symbol": "BTC", "timeframe": "1H"},
        {"symbol": "ETH", "timeframe": "1H"}
    ],
    "params": {
        "lookback": 500,
        "min_trades": 15,
        "win_rate_target": 0.50,  # Increased target
        "profit_target": 0.002,   # Increased from 0.0015
        "max_drawdown": 0.04      # Tightened from 0.05
    }
}

class DataFetcher:
    def __init__(self):
        pass
        
    def get_recent_data(self, symbol, interval):
        try:
            df = pd.read_csv(f'data/{symbol}_{interval}.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
            df = df.sort_values('timestamp', ascending=True).tail(500)
            return df
        except Exception as e:
            print(f"Error reading data for {symbol}: {str(e)}")
            return None

class Strategy:
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.target_symbol = self.config['target']['symbol']
        self.data_fetcher = DataFetcher()
        self.anchor_symbols = [anchor['symbol'] for anchor in self.config['anchors']]
        self.timeframe = self.config['target']['timeframe']

    def get_live_data(self):
        """Fetch live data for target and anchor assets."""
        try:
            target_data = self.data_fetcher.get_recent_data(self.target_symbol, self.timeframe)
            if target_data is None:
                return None, None

            anchor_data = pd.DataFrame()
            for symbol in self.anchor_symbols:
                data = self.data_fetcher.get_recent_data(symbol, self.timeframe)
                if data is not None:
                    anchor_data[f'close_{symbol}'] = data['close']
                    anchor_data[f'volume_{symbol}'] = data['volume']
                    anchor_data['timestamp'] = data['timestamp']

            if anchor_data.empty:
                return None, None

            return target_data, anchor_data
        except Exception as e:
            print(f"Error fetching live data: {str(e)}")
            return None, None

    def calculate_indicators(self, df):
        """Calculate technical indicators with improved parameters."""
        df = df.copy()
        df[['close', 'volume', 'high', 'low']] = df[['close', 'volume', 'high', 'low']].astype(float)
        
        # Multiple EMA periods for better trend detection
        for period in [8, 13, 21, 34, 55]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # EMA slope for trend strength
        df['ema_13_slope'] = df['ema_13'].diff(3) / df['ema_13'].shift(3)
        df['ema_21_slope'] = df['ema_21'].diff(5) / df['ema_21'].shift(5)
        
        # RSI with multiple timeframes and divergence detection
        def calculate_rsi(prices, period):
            delta = prices.diff()
            gain = delta.clip(lower=0).rolling(window=period).mean()
            loss = -delta.clip(upper=0).rolling(window=period).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        df['rsi_14'] = calculate_rsi(df['close'], 14)
        df['rsi_21'] = calculate_rsi(df['close'], 21)
        
        # RSI divergence signals
        df['price_higher_high'] = (df['close'] > df['close'].shift(10)) & (df['close'].shift(5) < df['close'].shift(15))
        df['rsi_lower_high'] = (df['rsi_14'] < df['rsi_14'].shift(10)) & (df['rsi_14'].shift(5) > df['rsi_14'].shift(15))
        df['bearish_divergence'] = df['price_higher_high'] & df['rsi_lower_high']
        
        # MACD with histogram analysis
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_hist_slope'] = df['macd_hist'].diff(2)
        
        # Volume analysis
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_sma_30'] = df['volume'].rolling(window=30).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_30']
        df['volume_trend'] = df['volume_sma_10'] / df['volume_sma_30']
        
        # Price momentum
        df['momentum_3'] = df['close'] / df['close'].shift(3) - 1
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # ATR for volatility and position sizing
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(window=14).mean()
        df['atr_pct'] = df['atr'] / df['close']
        
        # Bollinger Bands with squeeze detection
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = df['bb_std'] < df['bb_std'].rolling(window=20).mean() * 0.8
        
        # Support/Resistance levels
        df['recent_high'] = df['high'].rolling(window=20).max()
        df['recent_low'] = df['low'].rolling(window=20).min()
        df['near_resistance'] = df['close'] > (df['recent_high'] * 0.98)
        df['near_support'] = df['close'] < (df['recent_low'] * 1.02)
        
        return df

    def calculate_anchor_signals(self, df):
        """Enhanced anchor asset analysis for market regime detection."""
        df = df.copy()
        
        # Calculate momentum and correlation for anchor assets
        for symbol in ['BTC', 'ETH']:
            # Multiple momentum timeframes
            df[f'{symbol.lower()}_momentum_3'] = (df[f'close_{symbol}'] / df[f'close_{symbol}'].shift(3) - 1)
            df[f'{symbol.lower()}_momentum_5'] = (df[f'close_{symbol}'] / df[f'close_{symbol}'].shift(5) - 1)
            df[f'{symbol.lower()}_momentum_10'] = (df[f'close_{symbol}'] / df[f'close_{symbol}'].shift(10) - 1)
            
            # RSI for anchor assets
            delta = df[f'close_{symbol}'].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = -delta.clip(upper=0).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)
            df[f'{symbol.lower()}_rsi'] = 100 - (100 / (1 + rs))
            
            # Volume analysis for anchors
            df[f'{symbol.lower()}_volume_ratio'] = df[f'volume_{symbol}'] / df[f'volume_{symbol}'].rolling(window=20).mean()
        
        # Correlation analysis between LDO and anchors
        df['ldo_btc_corr'] = df['close'].rolling(window=20).corr(df['close_BTC'])
        df['ldo_eth_corr'] = df['close'].rolling(window=20).corr(df['close_ETH'])
        
        # Market regime classification
        strong_bull = (
            (df['btc_momentum_5'] > 0.02) & 
            (df['eth_momentum_5'] > 0.02) &
            (df['btc_rsi'] > 55) & 
            (df['eth_rsi'] > 55)
        )
        
        mild_bull = (
            (df['btc_momentum_10'] > 0) | 
            (df['eth_momentum_10'] > 0) |
            (df['btc_rsi'] > 50) | 
            (df['eth_rsi'] > 50)
        )
        
        df['market_regime'] = 0  # Bearish
        df.loc[mild_bull, 'market_regime'] = 1  # Mildly bullish  
        df.loc[strong_bull, 'market_regime'] = 2  # Strongly bullish
        
        # Composite market strength score
        df['market_strength'] = (
            df['btc_momentum_5'] * 0.3 +
            df['eth_momentum_5'] * 0.3 +
            (df['btc_rsi'] - 50) / 50 * 0.2 +
            (df['eth_rsi'] - 50) / 50 * 0.2
        )
        
        return df

    def generate_signals(self, candles_target: pd.DataFrame, candles_anchor: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals with enhanced logic for higher profitability and debug why no trades are made."""
        try:
            candles_target = self.calculate_indicators(candles_target)
            candles_anchor = candles_anchor.copy()
            anchor_cols = [col for col in candles_anchor.columns if col.startswith('close_')]
            for col in anchor_cols:
                candles_anchor[col] = candles_anchor[col].astype(float)
            df = pd.merge(candles_target, candles_anchor, on='timestamp', how='inner')
            df = df.dropna().reset_index(drop=True)
            df = self.calculate_anchor_signals(df)
            df['signal'] = 'HOLD'
            # --- Diagnostics: Check how many bars pass each filter ---
            strong_trend = (
                (df['ema_8'] > df['ema_13']) &
                (df['ema_13'] > df['ema_21']) &
                (df['ema_21'] > df['ema_34']) &
                (df['ema_13_slope'] > 0.001) &
                (df['ema_21_slope'] > 0.0005)
            )
            print('Strong trend bars:', strong_trend.sum())
            momentum_strong = (
                (df['rsi_14'] > 45) &  # Relaxed from 50
                (df['rsi_14'] < 78) &
                (df['momentum_3'] > 0.002) &  # Relaxed from 0.005
                (df['macd_hist'] > 0)
            )
            print('Momentum strong bars:', momentum_strong.sum())
            market_favorable = (
                (df['market_regime'] >= 0) &  # Allow neutral regime
                (df['market_strength'] > -0.02)  # Allow slightly negative
            )
            print('Market favorable bars:', market_favorable.sum())
            volume_strong = (
                (df['volume_ratio'] > 0.7) &  # Relaxed from 1.2
                (df['volume_trend'] > 0.95)   # Relaxed from 1.05
            )
            print('Volume strong bars:', volume_strong.sum())
            price_position_good = (
                (df['bb_position'] > 0.15) &  # Relaxed from 0.3
                (df['bb_position'] < 0.85) &  # Relaxed from 0.7
                (~df['near_resistance']) &
                (df['atr_pct'] < 0.12)        # Relaxed from 0.08
            )
            print('Price position good bars:', price_position_good.sum())
            # Remove divergence and correlation filters for now
            can_enter_mask = (strong_trend & momentum_strong & market_favorable & volume_strong & price_position_good)
            print('Bars passing all entry filters:', can_enter_mask.sum())
            print(df[['ema_8','ema_13','ema_21','ema_34','ema_13_slope','ema_21_slope','rsi_14','momentum_3','macd_hist','market_regime','market_strength','volume_ratio','volume_trend','bb_position','near_resistance','atr_pct']].tail(10))
            # --- Main signal loop ---
            position_active = False
            entry_price = 0
            bars_in_trade = 0
            cooldown_remaining = 0
            cooldown_period = 2
            stop_loss = 0
            take_profit = 0
            trailing_stop = 0
            best_price = 0
            for i in range(50, len(df)):
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                    continue
                current_price = df.at[df.index[i], 'close']
                current_atr = df.at[df.index[i], 'atr']
                if pd.isna(current_atr) or current_atr <= 0:
                    continue
                # Use relaxed can_enter logic
                can_enter = (
                    strong_trend.iloc[i] &
                    momentum_strong.iloc[i] &
                    market_favorable.iloc[i] &
                    volume_strong.iloc[i] &
                    price_position_good.iloc[i]
                )
                if not position_active and can_enter:
                    atr_multiplier = min(max(1.5, 3.0 / df.at[df.index[i], 'atr_pct']), 2.5)
                    stop_loss = current_price - (atr_multiplier * current_atr)
                    take_profit = current_price + (3.5 * atr_multiplier * current_atr)
                    trailing_stop = stop_loss
                    risk = current_price - stop_loss
                    reward = take_profit - current_price
                    rrr = reward / risk if risk > 0 else 0
                    if rrr >= 1.5:  # Slightly relaxed
                        df.at[df.index[i], 'signal'] = 'BUY'
                        position_active = True
                        entry_price = current_price
                        best_price = current_price
                        bars_in_trade = 0
                elif position_active:
                    bars_in_trade += 1
                    if current_price > best_price:
                        best_price = current_price
                        profit_pct = (best_price - entry_price) / entry_price
                        if profit_pct > 0.02:
                            trailing_stop = max(trailing_stop, best_price - (1.0 * current_atr))
                        elif profit_pct > 0.01:
                            trailing_stop = max(trailing_stop, best_price - (1.3 * current_atr))
                    exit_triggered = False
                    if current_price <= stop_loss:
                        exit_triggered = True
                    elif current_price >= take_profit:
                        exit_triggered = True
                    elif current_price <= trailing_stop:
                        exit_triggered = True
                    elif bars_in_trade >= 20:
                        exit_triggered = True
                    elif bars_in_trade >= 3:
                        trend_weakening = (
                            (df.at[df.index[i], 'ema_8'] < df.at[df.index[i], 'ema_13']) |
                            (df.at[df.index[i], 'ema_13_slope'] < -0.0005) |
                            (df.at[df.index[i], 'macd_hist'] < 0)
                        )
                        if trend_weakening:
                            exit_triggered = True
                    elif df.at[df.index[i], 'rsi_14'] > 80:
                        exit_triggered = True
                    elif (df.at[df.index[i], 'market_regime'] == 0) & (bars_in_trade >= 2):
                        exit_triggered = True
                    if exit_triggered:
                        df.at[df.index[i], 'signal'] = 'SELL'
                        position_active = False
                        cooldown_remaining = cooldown_period
                        entry_price = 0
                        bars_in_trade = 0
                        best_price = 0
            print("Enhanced Signals value counts:", df['signal'].value_counts())
            return df[['timestamp', 'signal']].set_index('timestamp').reindex(
                candles_target['timestamp'],
                fill_value='HOLD'
            ).reset_index()
        except Exception as e:
            error_info = f"Signal generation error: {str(e)}"
            print(error_info)
            return pd.DataFrame({'timestamp': candles_target['timestamp'], 'signal': 'HOLD'})

    def get_live_signal(self):
        try:
            target_data, anchor_data = self.get_live_data()
            if target_data is None or anchor_data is None:
                return "HOLD", "No data available"
            signals = self.generate_signals(target_data, anchor_data)
            latest_signal = signals.iloc[-1]['signal']
            latest_price = target_data.iloc[-1]['close']
            return latest_signal, f"Price: ${latest_price:.4f}"
        except Exception as e:
            error_info = f"Live signal error: {str(e)}"
            print(error_info)
            return "HOLD", error_info

def generate_signals(candles_target: pd.DataFrame, candles_anchor: pd.DataFrame) -> pd.DataFrame:
    """Generate trading signals using the strategy's logic."""
    strategy = Strategy()
    return strategy.generate_signals(candles_target, candles_anchor)

def get_coin_metadata():
    """Return metadata about the coin/strategy."""
    return {
        "name": "LDO Enhanced Profitability Strategy",
        "description": "An enhanced pairwise trading strategy for LDO with improved entry/exit conditions and risk management.",
        "target": DEFAULT_CONFIG["target"],
        "anchors": DEFAULT_CONFIG["anchors"]
    }

def backtest_strategy():
    print("\n=== Running Enhanced Profitability Backtest for LDO Strategy ===\n")
    
    # Load data
    data_dir = "data"
    target_file = os.path.join(data_dir, "LDO_1H.csv")
    btc_file = os.path.join(data_dir, "BTC_1H.csv")
    eth_file = os.path.join(data_dir, "ETH_1H.csv")

    # Read CSVs
    ldo = pd.read_csv(target_file)
    btc = pd.read_csv(btc_file)
    eth = pd.read_csv(eth_file)

    # Parse timestamps
    ldo['timestamp'] = pd.to_datetime(ldo['timestamp'])
    btc['timestamp'] = pd.to_datetime(btc['timestamp'])
    eth['timestamp'] = pd.to_datetime(eth['timestamp'])

    # Prepare anchor DataFrame
    anchor = pd.DataFrame({
        'timestamp': btc['timestamp'],
        'close_BTC': btc['close'],
        'volume_BTC': btc['volume']
    })
    anchor = pd.merge(anchor, pd.DataFrame({
        'timestamp': eth['timestamp'],
        'close_ETH': eth['close'],
        'volume_ETH': eth['volume']
    }), on='timestamp', how='inner')

    # Align target and anchor data
    ldo = ldo.sort_values('timestamp')
    anchor = anchor.sort_values('timestamp')
    ldo = ldo[ldo['timestamp'].isin(anchor['timestamp'])].reset_index(drop=True)
    anchor = anchor[anchor['timestamp'].isin(ldo['timestamp'])].reset_index(drop=True)

    # Generate signals
    strategy = Strategy()
    signals = strategy.generate_signals(ldo, anchor)
    ldo = ldo.merge(signals, on='timestamp', how='left')

    # Enhanced simulation with transaction costs
    initial_capital = 10000
    position = 0
    entry_price = 0
    equity_curve = [initial_capital]
    trade_returns = []
    max_equity = initial_capital
    drawdowns = []
    trades = []
    transaction_cost = 0.001  # 0.1% per trade
    
    for i, row in ldo.iterrows():
        price = row['close']
        signal = row['signal']
        
        if position == 0 and signal == 'BUY':
            position = 1
            entry_price = price * (1 + transaction_cost)  # Include buy cost
            entry_time = row['timestamp']
        elif position == 1 and signal == 'SELL':
            exit_price = price * (1 - transaction_cost)  # Include sell cost
            ret = (exit_price - entry_price) / entry_price
            trade_returns.append(ret)
            initial_capital *= (1 + ret)
            trades.append({
                'entry_time': entry_time,
                'exit_time': row['timestamp'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return': ret
            })
            position = 0
            entry_price = 0
        
        current_equity = initial_capital if position == 0 else initial_capital * ((price * (1 - transaction_cost)) / entry_price)
        equity_curve.append(current_equity)
        max_equity = max(max_equity, current_equity)
        drawdowns.append((max_equity - current_equity) / max_equity)

    # Enhanced metrics
    if len(trade_returns) == 0:
        print("No trades were made. Strategy may be too conservative.")
        return
    
    total_return = (equity_curve[-1] / equity_curve[0]) - 1
    profitability_score = min(max(total_return * 45, 0), 45)
    
    # Improved Sharpe calculation
    if len(trade_returns) > 1:
        sharpe = np.mean(trade_returns) / (np.std(trade_returns) + 1e-8) * np.sqrt(252*24)
    else:
        sharpe = 0
    sharpe_score = min(max(sharpe / 2 * 35, 0), 35)
    
    max_drawdown = max(drawdowns) if drawdowns else 0
    mdd_score = min(max((1 - max_drawdown) * 20, 0), 20)
    
    total_score = profitability_score + sharpe_score + mdd_score

    # Additional metrics
    winning_trades = [r for r in trade_returns if r > 0]
    losing_trades = [r for r in trade_returns if r <= 0]
    win_rate = len(winning_trades) / len(trade_returns) if trade_returns else 0
    
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades else float('inf')

    print(f"=== PERFORMANCE METRICS ===")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Profitability Score: {profitability_score:.2f} / 45")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Sharpe Score: {sharpe_score:.2f} / 35")
    print(f"Max Drawdown: {max_drawdown*100:.2f}%")
    print(f"Max Drawdown Score: {mdd_score:.2f} / 20")
    print(f"Total Score: {total_score:.2f} / 100\n")
    
    print(f"=== TRADE ANALYSIS ===")
    print(f"Number of Trades: {len(trade_returns)}")
    print(f"Win Rate: {win_rate*100:.2f}%")
    print(f"Average Trade Return: {np.mean(trade_returns)*100:.2f}%")
    print(f"Average Win: {avg_win*100:.2f}%")
    print(f"Average Loss: {avg_loss*100:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Best Trade: {max(trade_returns)*100:.2f}%")
    print(f"Worst Trade: {min(trade_returns)*100:.2f}%\n")
    
    print("=== CHALLENGE REQUIREMENTS ===")
    print(f"Profitability: {'✅' if profitability_score >= 15 else '❌'} (>= 15, current: {profitability_score:.2f})")
    print(f"Sharpe Ratio: {'✅' if sharpe_score >= 10 else '❌'} (>= 10, current: {sharpe_score:.2f})")
    print(f"Max Drawdown: {'✅' if mdd_score >= 5 else '❌'} (>= 5, current: {mdd_score:.2f})")
    print(f"Total Score: {'✅' if total_score >= 60 else '❌'} (>= 60, current: {total_score:.2f})")
    print("\n=== End of Enhanced Backtest ===\n")

if __name__ == "__main__":
    print("Enhanced Strategy Metadata:", get_coin_metadata())
    
    strategy = Strategy()
    signal, info = strategy.get_live_signal()
    print("Live Signal:", signal)
    print("Info:", info)
    
    backtest_strategy()