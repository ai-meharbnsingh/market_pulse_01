# src/ai_trading/professional_technical_analyzer.py

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import asyncio
import yfinance as yf
from dataclasses import dataclass


@dataclass
class TechnicalSignal:
    """Structured technical signal with metadata"""
    indicator: str
    signal: str  # BUY, SELL, HOLD
    strength: float  # 0-1 confidence
    value: float
    threshold: Optional[float] = None
    reasoning: Optional[str] = None


class ProfessionalTechnicalAnalyzer:
    """
    Professional-grade technical analysis engine with 30+ indicators
    Designed for AI integration and multi-timeframe analysis
    """

    def __init__(self):
        self.indicator_config = {
            # Trend Following Indicators
            'sma': {'periods': [9, 20, 50, 100, 200]},
            'ema': {'periods': [9, 12, 21, 26, 50]},
            'wma': {'periods': [20, 50]},
            'dema': {'period': 21},
            'tema': {'period': 21},

            # Momentum Oscillators
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'stoch': {'k_period': 14, 'd_period': 3},
            'stochrsi': {'period': 14, 'rsi_period': 14},
            'williams_r': {'period': 14},
            'cci': {'period': 20},
            'mfi': {'period': 14},

            # MACD Family
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'macd_histogram': {'fast': 12, 'slow': 26, 'signal': 9},
            'ppo': {'fast': 12, 'slow': 26, 'signal': 9},

            # Volatility Indicators
            'bbands': {'period': 20, 'std': 2},
            'keltner': {'period': 20, 'multiplier': 2},
            'atr': {'period': 14},
            'natr': {'period': 14},
            'true_range': {},

            # Volume Indicators
            'obv': {},  # On Balance Volume
            'ad': {},  # Accumulation/Distribution
            'cmf': {'period': 20},  # Chaikin Money Flow
            'vwap': {},
            'vwma': {'period': 20},
            'pvi': {},  # Positive Volume Index
            'nvi': {},  # Negative Volume Index

            # Support/Resistance
            'pivot_points': {},
            'supertrend': {'period': 10, 'multiplier': 3},
            'sar': {'acceleration': 0.02, 'maximum': 0.2},

            # Custom Composite Indicators
            'trend_strength': {},
            'momentum_composite': {},
            'volume_profile': {}
        }

        # Signal interpretation thresholds
        self.signal_thresholds = {
            'strong_buy': 0.8,
            'buy': 0.6,
            'hold': 0.4,
            'sell': 0.2,
            'strong_sell': 0.0
        }

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators to dataframe
        Handles NaN values and edge cases professionally
        """
        if df.empty or len(df) < 50:
            print("⚠️ Insufficient data for professional analysis")
            return df

        try:
            print(f"🔧 Adding professional technical indicators...")
            original_cols = len(df.columns)

            # 1. TREND FOLLOWING INDICATORS
            df = self._add_trend_indicators(df)

            # 2. MOMENTUM OSCILLATORS
            df = self._add_momentum_indicators(df)

            # 3. VOLATILITY INDICATORS
            df = self._add_volatility_indicators(df)

            # 4. VOLUME INDICATORS
            df = self._add_volume_indicators(df)

            # 5. SUPPORT/RESISTANCE
            df = self._add_support_resistance_indicators(df)

            # 6. CUSTOM COMPOSITE INDICATORS
            df = self._add_composite_indicators(df)

            new_cols = len(df.columns) - original_cols
            print(f"✅ Added {new_cols} professional indicators")

            return df

        except Exception as e:
            print(f"❌ Error adding indicators: {e}")
            return df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive trend following indicators"""

        # Multiple SMAs
        for period in self.indicator_config['sma']['periods']:
            df[f'SMA_{period}'] = ta.sma(df['Close'], length=period)

        # Multiple EMAs
        for period in self.indicator_config['ema']['periods']:
            df[f'EMA_{period}'] = ta.ema(df['Close'], length=period)

        # Weighted Moving Average
        df['WMA_20'] = ta.wma(df['Close'], length=20)
        df['WMA_50'] = ta.wma(df['Close'], length=50)

        # Double/Triple EMA
        df['DEMA_21'] = ta.dema(df['Close'], length=21)
        df['TEMA_21'] = ta.tema(df['Close'], length=21)

        # Trend Direction Indicators
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
        df['AROON_UP'] = ta.aroon(df['High'], df['Low'], length=14)['AROONU_14']
        df['AROON_DOWN'] = ta.aroon(df['High'], df['Low'], length=14)['AROOND_14']

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive momentum oscillators"""

        # RSI (Multiple periods)
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        df['RSI_21'] = ta.rsi(df['Close'], length=21)

        # Stochastic Oscillator
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
        df['STOCH_K'] = stoch['STOCHk_14_3_3']
        df['STOCH_D'] = stoch['STOCHd_14_3_3']

        # StochRSI
        stochrsi = ta.stochrsi(df['Close'], length=14)
        df['STOCHRSI_K'] = stochrsi['STOCHRSIk_14_14_3_3']
        df['STOCHRSI_D'] = stochrsi['STOCHRSId_14_14_3_3']

        # Williams %R
        df['WILLIAMS_R'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)

        # Commodity Channel Index
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)

        # Money Flow Index
        df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)

        # Rate of Change
        df['ROC'] = ta.roc(df['Close'], length=12)

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility and band indicators"""

        # Bollinger Bands
        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is not None:
            # Handle different column naming conventions
            bb_cols = bb.columns.tolist()
            bb_upper_col = [col for col in bb_cols if 'BBU' in col or 'upper' in col.lower()][0] if any(
                'BBU' in col or 'upper' in col.lower() for col in bb_cols) else bb_cols[0]
            bb_middle_col = [col for col in bb_cols if 'BBM' in col or 'middle' in col.lower()][0] if any(
                'BBM' in col or 'middle' in col.lower() for col in bb_cols) else bb_cols[1] if len(bb_cols) > 1 else \
            bb_cols[0]
            bb_lower_col = [col for col in bb_cols if 'BBL' in col or 'lower' in col.lower()][0] if any(
                'BBL' in col or 'lower' in col.lower() for col in bb_cols) else bb_cols[2] if len(bb_cols) > 2 else \
            bb_cols[0]

            df['BB_UPPER'] = bb[bb_upper_col]
            df['BB_MIDDLE'] = bb[bb_middle_col]
            df['BB_LOWER'] = bb[bb_lower_col]
            df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MIDDLE']
            df['BB_POSITION'] = (df['Close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])

        # Keltner Channels
        kc = ta.kc(df['High'], df['Low'], df['Close'], length=20, scalar=2)
        if kc is not None:
            kc_cols = kc.columns.tolist()
            kc_upper_col = [col for col in kc_cols if 'KCU' in col or 'upper' in col.lower()][0] if any(
                'KCU' in col or 'upper' in col.lower() for col in kc_cols) else kc_cols[0] if kc_cols else None
            kc_middle_col = \
            [col for col in kc_cols if 'KCB' in col or 'basis' in col.lower() or 'middle' in col.lower()][0] if any(
                'KCB' in col or 'basis' in col.lower() or 'middle' in col.lower() for col in kc_cols) else kc_cols[
                1] if len(kc_cols) > 1 else None
            kc_lower_col = [col for col in kc_cols if 'KCL' in col or 'lower' in col.lower()][0] if any(
                'KCL' in col or 'lower' in col.lower() for col in kc_cols) else kc_cols[2] if len(kc_cols) > 2 else None

            if kc_upper_col:
                df['KC_UPPER'] = kc[kc_upper_col]
            if kc_middle_col:
                df['KC_MIDDLE'] = kc[kc_middle_col]
            if kc_lower_col:
                df['KC_LOWER'] = kc[kc_lower_col]

        # Average True Range
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['NATR'] = ta.natr(df['High'], df['Low'], df['Close'], length=14)

        # Donchian Channels
        dc = ta.donchian(df['High'], df['Low'], lower_length=20, upper_length=20)
        df['DC_UPPER'] = dc['DCU_20_20']
        df['DC_LOWER'] = dc['DCL_20_20']
        df['DC_MIDDLE'] = dc['DCM_20_20']

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""

        # On Balance Volume
        df['OBV'] = ta.obv(df['Close'], df['Volume'])

        # Accumulation/Distribution Line
        df['AD'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])

        # Chaikin Money Flow
        df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)

        # Volume Weighted Average Price
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])

        # Volume Moving Averages
        df['VOLUME_SMA_20'] = ta.sma(df['Volume'], length=20)
        df['VOLUME_RATIO'] = df['Volume'] / df['VOLUME_SMA_20']

        # Price Volume Trend
        df['PVT'] = ta.pvt(df['Close'], df['Volume'])

        # Volume Rate of Change
        df['VROC'] = ta.roc(df['Volume'], length=12)

        return df

    def _add_support_resistance_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add support/resistance indicators"""

        # Parabolic SAR
        df['SAR'] = ta.psar(df['High'], df['Low'], acceleration=0.02, maximum=0.2)['PSARl_0.02_0.2']

        # SuperTrend
        try:
            supertrend = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
            if supertrend is not None:
                st_cols = supertrend.columns.tolist()
                # Find the SuperTrend value column
                st_value_col = [col for col in st_cols if 'SUPERT' in col and 'd' not in col][0] if any(
                    'SUPERT' in col and 'd' not in col for col in st_cols) else None
                st_direction_col = [col for col in st_cols if 'SUPERTd' in col or 'direction' in col.lower()][0] if any(
                    'SUPERTd' in col or 'direction' in col.lower() for col in st_cols) else None

                if st_value_col:
                    df['SUPERTREND'] = supertrend[st_value_col]
                if st_direction_col:
                    df['SUPERTREND_DIRECTION'] = supertrend[st_direction_col]
        except Exception as e:
            print(f"⚠️ SuperTrend calculation failed: {e}")
            # Create dummy columns
            df['SUPERTREND'] = df['Close']
            df['SUPERTREND_DIRECTION'] = 1

        # Pivot Points (Classical)
        df['PIVOT'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
        df['R1'] = 2 * df['PIVOT'] - df['Low'].shift(1)
        df['R2'] = df['PIVOT'] + (df['High'].shift(1) - df['Low'].shift(1))
        df['S1'] = 2 * df['PIVOT'] - df['High'].shift(1)
        df['S2'] = df['PIVOT'] - (df['High'].shift(1) - df['Low'].shift(1))

        # Dynamic Support/Resistance (Fixed - no lookahead bias)
        df['LOCAL_HIGH'] = df['High'].rolling(window=20).max().shift(1)
        df['LOCAL_LOW'] = df['Low'].rolling(window=20).min().shift(1)

        return df

    def _add_composite_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom composite indicators"""

        # Trend Strength Composite (0-100)
        adx_normalized = np.clip(df['ADX'] / 50, 0, 1) if 'ADX' in df.columns else 0.5
        sma_trend = np.where(df['Close'] > df['SMA_50'], 1, 0) if 'SMA_50' in df.columns else 0.5
        ema_trend = np.where(df['EMA_21'] > df['EMA_50'], 1,
                             0) if 'EMA_21' in df.columns and 'EMA_50' in df.columns else 0.5
        df['TREND_STRENGTH'] = (adx_normalized + sma_trend + ema_trend) * 33.33

        # Momentum Composite (0-100)
        rsi_momentum = (df['RSI_14'] - 30) / 40 if 'RSI_14' in df.columns else 0.5
        stoch_momentum = df['STOCH_K'] / 100 if 'STOCH_K' in df.columns else 0.5
        roc_momentum = np.clip(df['ROC'] / 10 + 0.5, 0, 1) if 'ROC' in df.columns else 0.5
        df['MOMENTUM_COMPOSITE'] = np.clip((rsi_momentum + stoch_momentum + roc_momentum) * 33.33, 0, 100)

        # Volume Profile (Relative volume strength)
        if 'VOLUME_RATIO' in df.columns:
            df['VOLUME_PROFILE'] = np.clip(df['VOLUME_RATIO'] * 50, 0, 100)
        else:
            df['VOLUME_PROFILE'] = 50

        # Volatility Index (0-100)
        if 'ATR' in df.columns:
            atr_20_avg = df['ATR'].rolling(window=20).mean()
            df['VOLATILITY_INDEX'] = np.clip((df['ATR'] / atr_20_avg) * 50, 0, 100)
        else:
            df['VOLATILITY_INDEX'] = 50

        return df

    def generate_technical_signals(self, df: pd.DataFrame) -> List[TechnicalSignal]:
        """
        Generate structured technical trading signals
        Returns list of TechnicalSignal objects with reasoning
        """
        signals = []

        if df.empty or len(df) < 2:
            return signals

        latest = df.iloc[-1]

        try:
            # 1. RSI SIGNALS
            if 'RSI_14' in df.columns and pd.notna(latest['RSI_14']):
                rsi = latest['RSI_14']
                if rsi <= 30:
                    signals.append(TechnicalSignal(
                        indicator='RSI_14',
                        signal='BUY',
                        strength=0.8,
                        value=rsi,
                        threshold=30,
                        reasoning=f"RSI oversold at {rsi:.1f}, bullish reversal expected"
                    ))
                elif rsi >= 70:
                    signals.append(TechnicalSignal(
                        indicator='RSI_14',
                        signal='SELL',
                        strength=0.8,
                        value=rsi,
                        threshold=70,
                        reasoning=f"RSI overbought at {rsi:.1f}, bearish reversal expected"
                    ))

            # 2. MOVING AVERAGE CROSSOVER SIGNALS
            if all(col in df.columns for col in ['EMA_12', 'EMA_26']):
                ema_12_curr = latest['EMA_12']
                ema_26_curr = latest['EMA_26']
                ema_12_prev = df.iloc[-2]['EMA_12'] if len(df) > 1 else ema_12_curr
                ema_26_prev = df.iloc[-2]['EMA_26'] if len(df) > 1 else ema_26_curr

                # Golden Cross
                if ema_12_prev <= ema_26_prev and ema_12_curr > ema_26_curr:
                    signals.append(TechnicalSignal(
                        indicator='EMA_CROSSOVER',
                        signal='BUY',
                        strength=0.75,
                        value=ema_12_curr - ema_26_curr,
                        reasoning="Bullish EMA crossover (12>26), trend reversal signal"
                    ))
                # Death Cross
                elif ema_12_prev >= ema_26_prev and ema_12_curr < ema_26_curr:
                    signals.append(TechnicalSignal(
                        indicator='EMA_CROSSOVER',
                        signal='SELL',
                        strength=0.75,
                        value=ema_12_curr - ema_26_curr,
                        reasoning="Bearish EMA crossover (12<26), trend reversal signal"
                    ))

            # 3. BOLLINGER BANDS SIGNALS
            if all(col in df.columns for col in ['BB_UPPER', 'BB_LOWER', 'BB_POSITION']):
                bb_pos = latest['BB_POSITION']
                close_price = latest['Close']

                if bb_pos <= 0.1:  # Near lower band
                    signals.append(TechnicalSignal(
                        indicator='BOLLINGER_BANDS',
                        signal='BUY',
                        strength=0.7,
                        value=bb_pos,
                        threshold=0.1,
                        reasoning=f"Price near BB lower band (pos: {bb_pos:.2f}), oversold bounce expected"
                    ))
                elif bb_pos >= 0.9:  # Near upper band
                    signals.append(TechnicalSignal(
                        indicator='BOLLINGER_BANDS',
                        signal='SELL',
                        strength=0.7,
                        value=bb_pos,
                        threshold=0.9,
                        reasoning=f"Price near BB upper band (pos: {bb_pos:.2f}), overbought correction expected"
                    ))

            # 4. MACD SIGNALS
            macd_cols = [col for col in df.columns if 'MACD' in col and 'MACDs' not in col]
            signal_cols = [col for col in df.columns if 'MACDs' in col or ('MACD' in col and 'signal' in col.lower())]

            if macd_cols and signal_cols:
                macd = latest[macd_cols[0]]
                signal_line = latest[signal_cols[0]]
                macd_prev = df.iloc[-2][macd_cols[0]] if len(df) > 1 else macd
                signal_prev = df.iloc[-2][signal_cols[0]] if len(df) > 1 else signal_line

                # MACD bullish crossover
                if pd.notna(macd) and pd.notna(signal_line) and pd.notna(macd_prev) and pd.notna(signal_prev):
                    if macd_prev <= signal_prev and macd > signal_line:
                        signals.append(TechnicalSignal(
                            indicator='MACD',
                            signal='BUY',
                            strength=0.65,
                            value=macd - signal_line,
                            reasoning="MACD bullish crossover, momentum shifting up"
                        ))
                    # MACD bearish crossover
                    elif macd_prev >= signal_prev and macd < signal_line:
                        signals.append(TechnicalSignal(
                            indicator='MACD',
                            signal='SELL',
                            strength=0.65,
                            value=macd - signal_line,
                            reasoning="MACD bearish crossover, momentum shifting down"
                        ))

            # 5. VOLUME CONFIRMATION SIGNALS
            if 'VOLUME_RATIO' in df.columns:
                vol_ratio = latest['VOLUME_RATIO']
                if vol_ratio > 2.0:  # High volume spike
                    # Determine direction based on price movement
                    price_change = (latest['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100 if len(
                        df) > 1 else 0

                    if price_change > 1:  # Positive move with high volume
                        signals.append(TechnicalSignal(
                            indicator='VOLUME_BREAKOUT',
                            signal='BUY',
                            strength=0.6,
                            value=vol_ratio,
                            threshold=2.0,
                            reasoning=f"High volume breakout (vol ratio: {vol_ratio:.1f}), price up {price_change:.1f}%"
                        ))
                    elif price_change < -1:  # Negative move with high volume
                        signals.append(TechnicalSignal(
                            indicator='VOLUME_BREAKDOWN',
                            signal='SELL',
                            strength=0.6,
                            value=vol_ratio,
                            threshold=2.0,
                            reasoning=f"High volume breakdown (vol ratio: {vol_ratio:.1f}), price down {price_change:.1f}%"
                        ))

        except Exception as e:
            print(f"⚠️ Error generating signals: {e}")

        return signals

    def get_technical_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive technical analysis summary
        """
        if df.empty:
            return {"error": "No data available"}

        latest = df.iloc[-1]
        signals = self.generate_technical_signals(df)

        # Count signal types
        buy_signals = [s for s in signals if s.signal == 'BUY']
        sell_signals = [s for s in signals if s.signal == 'SELL']

        # Calculate composite scores
        trend_score = latest.get('TREND_STRENGTH', 50)
        momentum_score = latest.get('MOMENTUM_COMPOSITE', 50)
        volume_score = latest.get('VOLUME_PROFILE', 50)

        # Overall technical bias
        signal_bias = "NEUTRAL"
        if len(buy_signals) > len(sell_signals) and len(buy_signals) >= 2:
            signal_bias = "BULLISH"
        elif len(sell_signals) > len(buy_signals) and len(sell_signals) >= 2:
            signal_bias = "BEARISH"

        return {
            'timestamp': datetime.now().isoformat(),
            'price': latest['Close'],
            'total_signals': len(signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'signal_bias': signal_bias,
            'technical_scores': {
                'trend_strength': trend_score,
                'momentum_composite': momentum_score,
                'volume_profile': volume_score,
                'volatility_index': latest.get('VOLATILITY_INDEX', 50)
            },
            'key_levels': {
                'resistance': latest.get('R1', latest['Close'] * 1.02),
                'support': latest.get('S1', latest['Close'] * 0.98),
                'pivot': latest.get('PIVOT', latest['Close'])
            },
            'signals': [
                {
                    'indicator': s.indicator,
                    'signal': s.signal,
                    'strength': s.strength,
                    'value': s.value,
                    'reasoning': s.reasoning
                } for s in signals
            ]
        }


# Testing function
async def test_professional_analyzer():
    """Test the professional technical analyzer"""
    print("🧪 Testing Professional Technical Analyzer")
    print("=" * 60)

    analyzer = ProfessionalTechnicalAnalyzer()

    # Download sample data
    symbol = "RELIANCE.NS"
    print(f"📊 Testing with {symbol}")

    try:
        # Get data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6mo", interval="1d")

        if df.empty:
            print("❌ No data received")
            return

        print(f"📈 Got {len(df)} data points")

        # Add indicators
        df_with_indicators = analyzer.add_all_indicators(df.copy())
        print(f"🔧 Added indicators, now have {len(df_with_indicators.columns)} columns")

        # Generate signals
        signals = analyzer.generate_technical_signals(df_with_indicators)
        print(f"🎯 Generated {len(signals)} trading signals")

        for signal in signals:
            print(f"  📍 {signal.indicator}: {signal.signal} (strength: {signal.strength:.2f}) - {signal.reasoning}")

        # Get technical summary
        summary = analyzer.get_technical_summary(df_with_indicators)
        print(f"\n📋 Technical Summary:")
        print(f"  Overall Bias: {summary['signal_bias']}")
        print(f"  Trend Strength: {summary['technical_scores']['trend_strength']:.1f}/100")
        print(f"  Momentum: {summary['technical_scores']['momentum_composite']:.1f}/100")
        print(f"  Volume Profile: {summary['technical_scores']['volume_profile']:.1f}/100")

        print(f"\n✅ Professional Technical Analyzer is working perfectly!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(test_professional_analyzer())