import pandas as pd
import numpy as np

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
        "win_rate_target": 0.45,
        "profit_target": 0.0015,
        "max_drawdown": 0.05
    }
}

def generate_signals(candles_target: pd.DataFrame, candles_anchor: pd.DataFrame) -> pd.DataFrame:
    """Generate trading signals using the strategy's logic.
    
    Args:
        candles_target (pd.DataFrame): Historical price data for the target asset
        candles_anchor (pd.DataFrame): Historical price data for the anchor assets (BTC and ETH)
        
    Returns:
        pd.DataFrame: DataFrame with timestamp and signal columns
    """
    strategy = Strategy()
    return strategy.generate_signals(candles_target, candles_anchor)

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

    def calculate_enhanced_indicators(self, df):
        df = df.copy()
        df[['close', 'volume', 'high', 'low']] = df[['close', 'volume', 'high', 'low']].astype(float)
        
        # Enhanced Multi-timeframe trend analysis
        for period in [8, 13, 21, 34, 55, 89]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Enhanced RSI with multiple timeframes
        def calculate_rsi(prices, period):
            delta = prices.diff()
            gain = delta.clip(lower=0).rolling(window=period).mean()
            loss = -delta.clip(upper=0).rolling(window=period).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        for period in [14, 21, 34]:
            df[f'rsi_{period}'] = calculate_rsi(df['close'], period)
        
        # RSI trend alignment
        df['rsi_trend_aligned'] = ((df['rsi_14'] > df['rsi_14'].shift(1)) & (df['close'] > df['close'].shift(1)))
        
        # Enhanced MACD analysis
        for (fast, slow, signal) in [(12, 26, 9), (5, 35, 5)]:
            ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal, adjust=False).mean()
            
            prefix = f'macd_{fast}_{slow}'
            df[f'{prefix}'] = macd
            df[f'{prefix}_signal'] = macd_signal
            df[f'{prefix}_hist'] = macd - macd_signal
            df[f'{prefix}_momentum'] = macd.diff()
        
        # Enhanced Volume and Trend Analysis
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_trend'] = df['volume_sma'] > df['volume_sma'].shift(5)
        df['volume_consistency'] = (df['volume'] > df['volume'].shift(1)).rolling(window=5).mean()

        # Volatility Metrics
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(window=14).mean()
        df['atr_ratio'] = df['atr'] / df['close']
        df['atr_ratio_ma'] = df['atr_ratio'].rolling(window=20).mean()

        # Trend Strength
        df['trend_strength'] = abs(df['close'] - df['sma_200']) / df['sma_200']
        df['trend_consistency'] = ((df['close'] > df['ema_21']).rolling(window=5).mean() > 0.8) | \
                                ((df['close'] < df['ema_21']).rolling(window=5).mean() > 0.8)
        
        return df

    def calculate_anchor_signals(self, df):
        """Calculate anchor asset signals for market regime detection."""
        df = df.copy()
        
        # Cross-market correlations
        for window in [10, 20, 50]:
            for symbol in ['BTC', 'ETH']:
                df[f'corr_{symbol}_{window}'] = df['close'].rolling(window).corr(df[f'close_{symbol}'])
            df[f'corr_btc_eth_{window}'] = df['close_BTC'].rolling(window).corr(df['close_ETH'])
        
        # Market regime scoring
        df['regime_score'] = (
            df['corr_BTC_20'].fillna(0) * 0.4 +
            df['corr_ETH_20'].fillna(0) * 0.4 +
            df['corr_btc_eth_20'].fillna(0) * 0.2
        )
        
        df['market_regime'] = pd.qcut(
            df['regime_score'],
            q=5,
            labels=['very_bearish', 'bearish', 'neutral', 'bullish', 'very_bullish']
        )
        
        # Market strength
        for symbol in ['BTC', 'ETH']:
            for period in [5, 10, 21]:
                momentum = (df[f'close_{symbol}'] - df[f'close_{symbol}'].shift(period)) / df[f'close_{symbol}'].shift(period)
                df[f'{symbol.lower()}_norm_mom_{period}'] = (momentum - momentum.mean()) / momentum.std()
        
        df['market_strength'] = (
            df['btc_norm_mom_10'] * 0.4 +
            df['eth_norm_mom_10'] * 0.4 +
            df['regime_score'] * 0.2
        )
        
        return df

    def generate_signals(self, candles_target: pd.DataFrame, candles_anchor: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals with enhanced machine learning-inspired features."""
        try:
            # Calculate indicators
            candles_target = self.calculate_enhanced_indicators(candles_target)
            
            # Prepare anchor data
            candles_anchor = candles_anchor.copy()
            anchor_cols = [col for col in candles_anchor.columns if col.startswith('close_')]
            for col in anchor_cols:
                candles_anchor[col] = candles_anchor[col].astype(float)
            
            df = pd.merge(candles_target, candles_anchor, on='timestamp', how='inner')
            df = df.dropna().reset_index(drop=True)
            df = self.calculate_anchor_signals(df)
            df['signal'] = 'HOLD'
            
            # Entry conditions and scoring
            df['trend_quality'] = (
                (df['close'] > df['vwap']) * 1.2 +
                (df['ema_8'] > df['ema_13']) * 1.0 +
                (df['ema_21'] > df['ema_55']) * 1.1 +
                (df['close'] > df['sma_200']) * 1.3 +
                df['trend_consistency'] * 1.4
            )
            
            df['regime_multiplier'] = df['market_regime'].map({
                'very_bearish': 0.5,
                'bearish': 0.7,
                'neutral': 1.0,
                'bullish': 1.2,
                'very_bullish': 1.3
            })
            
            df['momentum_quality'] = (
                (df['macd_12_26_momentum'] > 0) * 1.2 +
                (df['rsi_trend_aligned']) * 1.3 +
                (df['trend_strength'] > df['trend_strength'].rolling(10).mean()) * 1.1
            )
            
            df['volume_quality'] = (
                (df['volume_ratio'] > 1.2) * 1.3 +
                df['volume_trend'] * 1.1 +
                (df['volume_consistency'] > 0.6) * 1.2
            )
            
            df['entry_score'] = (
                df['trend_quality'] * 0.35 +
                df['momentum_quality'] * 0.25 +
                df['volume_quality'] * 0.20
            ) * df['regime_multiplier']
            
            # Position management variables
            position_active = False
            entry_price = 0
            stop_loss = 0
            take_profit = 0
            bars_in_trade = 0
            cooldown_remaining = 0
            
            # Risk parameters
            base_atr_multiplier_stop = 1.2
            base_atr_multiplier_target = 3.8
            cooldown_period = 4
            
            # Signal generation
            for i in range(50, len(df)):
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                    continue
                
                current_price = df.at[df.index[i], 'close']
                current_atr = df.at[df.index[i], 'atr']
                
                if not position_active:
                    entry_threshold = 4.0 * df.at[df.index[i], 'regime_multiplier']
                    
                    can_enter = (
                        df.at[df.index[i], 'entry_score'] >= entry_threshold and
                        cooldown_remaining == 0 and
                        not pd.isna(current_atr) and
                        current_atr > 0
                    )
                    
                    if can_enter:
                        atr_multiplier_stop = base_atr_multiplier_stop * df.at[df.index[i], 'regime_multiplier']
                        atr_multiplier_target = base_atr_multiplier_target * df.at[df.index[i], 'regime_multiplier']
                        
                        stop_loss = current_price - (atr_multiplier_stop * current_atr)
                        take_profit = current_price + (atr_multiplier_target * current_atr)
                        
                        risk = current_price - stop_loss
                        reward = take_profit - current_price
                        rrr = reward / risk if risk > 0 else 0
                        
                        if rrr >= 2.5:
                            df.at[df.index[i], 'signal'] = 'BUY'
                            position_active = True
                            entry_price = current_price
                            bars_in_trade = 0
                
                else:
                    bars_in_trade += 1
                    exit_triggered = False
                    
                    # Exit conditions
                    if current_price <= stop_loss:
                        exit_triggered = True
                    elif current_price >= take_profit:
                        exit_triggered = True
                    elif bars_in_trade >= 20:
                        exit_triggered = True
                    elif df.at[df.index[i], 'market_regime'] in ['very_bearish', 'bearish']:
                        exit_triggered = True
                    
                    if exit_triggered:
                        df.at[df.index[i], 'signal'] = 'SELL'
                        position_active = False
                        cooldown_remaining = cooldown_period
                        entry_price = 0
                        stop_loss = 0
                        take_profit = 0
                        bars_in_trade = 0
            
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
