from .ml_signal_enhancer import MLSignalEnhancer

# Placeholder mock models for testing
class MockAlphaModel:
    def predict_profitability(self, symbol, market_data):
        import numpy as np
        return {
            'signal': 'BUY' if np.random.random() > 0.5 else 'SELL',
            'confidence': np.random.uniform(0.5, 0.9),
            'factors': {'momentum': np.random.uniform(-1, 1)},
            'alpha_score': np.random.uniform(0, 1)
        }

class MockLSTMModel:
    def predict_profitability(self, symbol, market_data):
        import numpy as np
        return {
            'signal': 'BUY' if np.random.random() > 0.5 else 'SELL',
            'confidence': np.random.uniform(0.5, 0.9),
            'predicted_price': market_data['close'].iloc[-1] * np.random.uniform(0.9, 1.1),
            'price_change_pct': np.random.uniform(-0.05, 0.05),
            'horizon': 5
        }

__all__ = ['MLSignalEnhancer', 'MockAlphaModel', 'MockLSTMModel']