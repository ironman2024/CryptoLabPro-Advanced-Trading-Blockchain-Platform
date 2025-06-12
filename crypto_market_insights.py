import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json
import os

class MarketInsights:
    """Provides market insights and recommendations for crypto trading"""
    
    def __init__(self):
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def get_market_sentiment(self, price_data, lookback=14):
        """Calculate market sentiment based on price action"""
        if len(price_data) < lookback:
            return "Neutral", 0
            
        # Calculate price change
        price_change = (price_data['close'].iloc[-1] / price_data['close'].iloc[-lookback] - 1) * 100
        
        # Calculate volume change
        volume_change = (price_data['volume'].iloc[-1] / price_data['volume'].iloc[-lookback:].mean() - 1) * 100
        
        # Calculate volatility
        returns = price_data['close'].pct_change().iloc[-lookback:]
        volatility = returns.std() * 100
        
        # Determine sentiment
        sentiment_score = 0
        
        # Price contribution
        if price_change > 5:
            sentiment_score += 2
        elif price_change > 2:
            sentiment_score += 1
        elif price_change < -5:
            sentiment_score -= 2
        elif price_change < -2:
            sentiment_score -= 1
            
        # Volume contribution
        if volume_change > 50:
            sentiment_score += 1
        elif volume_change < -50:
            sentiment_score -= 1
            
        # Volatility contribution
        if volatility > 5:
            sentiment_score -= 1  # High volatility often indicates uncertainty
            
        # Determine sentiment category
        if sentiment_score >= 2:
            sentiment = "Strongly Bullish"
        elif sentiment_score == 1:
            sentiment = "Bullish"
        elif sentiment_score == 0:
            sentiment = "Neutral"
        elif sentiment_score == -1:
            sentiment = "Bearish"
        else:
            sentiment = "Strongly Bearish"
            
        return sentiment, sentiment_score
    
    def get_technical_signals(self, price_data):
        """Get technical analysis signals"""
        signals = {}
        
        # Calculate EMAs
        price_data['ema_9'] = price_data['close'].ewm(span=9, adjust=False).mean()
        price_data['ema_21'] = price_data['close'].ewm(span=21, adjust=False).mean()
        price_data['ema_50'] = price_data['close'].ewm(span=50, adjust=False).mean()
        
        # Calculate RSI
        delta = price_data['close'].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = -delta.clip(upper=0).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        price_data['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        price_data['ema_12'] = price_data['close'].ewm(span=12, adjust=False).mean()
        price_data['ema_26'] = price_data['close'].ewm(span=26, adjust=False).mean()
        price_data['macd'] = price_data['ema_12'] - price_data['ema_26']
        price_data['macd_signal'] = price_data['macd'].ewm(span=9, adjust=False).mean()
        price_data['macd_hist'] = price_data['macd'] - price_data['macd_signal']
        
        # EMA signals
        if price_data['ema_9'].iloc[-1] > price_data['ema_21'].iloc[-1]:
            signals['ema_short'] = "Bullish"
        else:
            signals['ema_short'] = "Bearish"
            
        if price_data['ema_21'].iloc[-1] > price_data['ema_50'].iloc[-1]:
            signals['ema_long'] = "Bullish"
        else:
            signals['ema_long'] = "Bearish"
            
        # RSI signals
        rsi_value = price_data['rsi'].iloc[-1]
        if rsi_value > 70:
            signals['rsi'] = "Overbought"
        elif rsi_value < 30:
            signals['rsi'] = "Oversold"
        else:
            signals['rsi'] = "Neutral"
            
        # MACD signals
        if price_data['macd'].iloc[-1] > price_data['macd_signal'].iloc[-1]:
            signals['macd'] = "Bullish"
        else:
            signals['macd'] = "Bearish"
            
        # Overall signal
        bullish_count = sum(1 for signal in signals.values() if signal in ["Bullish", "Oversold"])
        bearish_count = sum(1 for signal in signals.values() if signal in ["Bearish", "Overbought"])
        
        if bullish_count > bearish_count:
            signals['overall'] = "Bullish"
        elif bearish_count > bullish_count:
            signals['overall'] = "Bearish"
        else:
            signals['overall'] = "Neutral"
            
        return signals
    
    def get_support_resistance(self, price_data, lookback=30):
        """Calculate support and resistance levels"""
        if len(price_data) < lookback:
            return None, None
            
        # Get recent price action
        recent_data = price_data.iloc[-lookback:]
        
        # Find local maxima and minima
        highs = []
        lows = []
        
        for i in range(2, len(recent_data) - 2):
            # Local high
            if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and 
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i-2] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+2]):
                highs.append(recent_data['high'].iloc[i])
                
            # Local low
            if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and 
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i-2] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+2]):
                lows.append(recent_data['low'].iloc[i])
        
        # If not enough points, use simple high/low
        if len(highs) < 2:
            highs = [recent_data['high'].max()]
        if len(lows) < 2:
            lows = [recent_data['low'].min()]
            
        # Calculate levels
        resistance = np.mean(highs)
        support = np.mean(lows)
        
        return support, resistance
    
    def get_trading_recommendation(self, price_data, sentiment, signals, risk_level="medium"):
        """Generate trading recommendation based on analysis"""
        current_price = price_data['close'].iloc[-1]
        
        # Determine if it's a good time to enter
        enter_trade = False
        if sentiment in ["Strongly Bullish", "Bullish"] and signals['overall'] == "Bullish":
            enter_trade = True
        
        # Risk level determines position size
        if risk_level == "low":
            position_size_pct = 0.02  # 2% of portfolio
        elif risk_level == "medium":
            position_size_pct = 0.05  # 5% of portfolio
        else:  # high
            position_size_pct = 0.10  # 10% of portfolio
            
        # Calculate stop loss and take profit
        atr = self._calculate_atr(price_data)
        
        if risk_level == "low":
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 4)
        elif risk_level == "medium":
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 3)
        else:  # high
            stop_loss = current_price - (atr * 1)
            take_profit = current_price + (atr * 2)
            
        recommendation = {
            "enter_trade": enter_trade,
            "position_size_pct": position_size_pct,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": (take_profit - current_price) / (current_price - stop_loss) if stop_loss < current_price else 0
        }
        
        return recommendation
    
    def _calculate_atr(self, price_data, period=14):
        """Calculate Average True Range"""
        high_low = price_data['high'] - price_data['low']
        high_close = abs(price_data['high'] - price_data['close'].shift())
        low_close = abs(price_data['low'] - price_data['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def get_market_insights(self, symbol, timeframe="1h"):
        """Get comprehensive market insights for a symbol"""
        try:
            # Get price data
            file_path = f"{self.data_dir}/{symbol}_{timeframe}.csv"
            if os.path.exists(file_path):
                price_data = pd.read_csv(file_path)
                price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
            else:
                return {"error": f"No data available for {symbol}_{timeframe}"}
                
            # Calculate sentiment
            sentiment, sentiment_score = self.get_market_sentiment(price_data)
            
            # Get technical signals
            signals = self.get_technical_signals(price_data)
            
            # Get support/resistance
            support, resistance = self.get_support_resistance(price_data)
            
            # Get trading recommendation
            recommendation = self.get_trading_recommendation(price_data, sentiment, signals)
            
            # Compile insights
            insights = {
                "symbol": symbol,
                "timeframe": timeframe,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "current_price": price_data['close'].iloc[-1],
                "24h_change": (price_data['close'].iloc[-1] / price_data['close'].iloc[-24] - 1) * 100 if len(price_data) > 24 else 0,
                "sentiment": {
                    "overall": sentiment,
                    "score": sentiment_score
                },
                "technical_signals": signals,
                "support_resistance": {
                    "support": support,
                    "resistance": resistance
                },
                "recommendation": recommendation
            }
            
            return insights
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_beginner_advice(self):
        """Get advice for beginner crypto traders"""
        advice = {
            "risk_management": [
                "Never risk more than 1-2% of your portfolio on a single trade",
                "Always use stop losses to protect your capital",
                "Don't chase pumps or FOMO into trades",
                "Start with small position sizes until you gain experience"
            ],
            "technical_analysis": [
                "Learn to identify key support and resistance levels",
                "Use multiple timeframes for confirmation",
                "Don't rely on a single indicator for trading decisions",
                "Look for confluence of multiple signals"
            ],
            "psychology": [
                "Develop and stick to a trading plan",
                "Keep a trading journal to track and improve your performance",
                "Control your emotions - both fear and greed can lead to poor decisions",
                "Be patient and wait for high-probability setups"
            ],
            "market_understanding": [
                "Understand that crypto markets are highly volatile",
                "Be aware of market cycles and overall trends",
                "Pay attention to Bitcoin's movements as it influences the entire market",
                "Stay informed about regulatory developments"
            ]
        }
        
        return advice

# Example usage
if __name__ == "__main__":
    insights = MarketInsights()
    btc_insights = insights.get_market_insights("BTC")
    print(json.dumps(btc_insights, indent=2))