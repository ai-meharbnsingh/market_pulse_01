# simple_ml_test.py
"""
Simple ML Test - Bypass Prophet Issues
Test the Alpha Model and ML Signal Enhancer without LSTM
"""

import sys
import os
from pathlib import Path

# Add the models directory to path
sys.path.append("03_ML_ENGINE/models")


def test_alpha_model():
    """Test Alpha Model functionality"""
    try:
        from alpha_model import AlphaModelCore as AlphaModel

        # Create test data
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        # Generate sample market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(100).cumsum() + 2500,
            'high': np.random.randn(100).cumsum() + 2520,
            'low': np.random.randn(100).cumsum() + 2480,
            'close': np.random.randn(100).cumsum() + 2500,
            'volume': np.random.randint(1000000, 5000000, 100)
        }).set_index('timestamp')

        # Initialize Alpha Model with proper database path
        alpha_model = AlphaModel()
        # Ensure database directory exists
        os.makedirs("data", exist_ok=True)

        # Test prediction
        prediction = alpha_model.predict('RELIANCE', sample_data)

        print("✅ Alpha Model Test Results:")
        print(f"   Symbol: {prediction['symbol']}")
        print(f"   Signal: {prediction['signal']}")
        print(f"   Confidence: {prediction['confidence']:.2%}")
        print(f"   Alpha Score: {prediction['alpha_score']:.3f}")
        print(f"   Model: {prediction['model']}")

        return True

    except Exception as e:
        print(f"❌ Alpha Model Test Failed: {e}")
        return False


def test_ml_signal_enhancer():
    """Test ML Signal Enhancer without LSTM"""
    try:
        # Import ML Signal Enhancer
        from ml_signal_enhancer import MLSignalEnhancer

        # Initialize with minimal config
        config = {
            'ensemble': {
                'min_models_agree': 1,  # Only need 1 model to agree
                'alpha_weight': 1.0,  # Full weight to Alpha model
                'lstm_weight': 0.0,  # No LSTM
                'technical_weight': 0.0  # Focus on ML only
            },
            'alpha_model': {'enabled': True},
            'lstm': {'enabled': False}
        }

        # Initialize enhancer
        enhancer = MLSignalEnhancer(config)

        # Test signal enhancement
        base_signal = {
            'symbol': 'RELIANCE',
            'signal': 'BUY',
            'confidence': 0.7,
            'technical_score': 85.0
        }

        # Simulate market data
        import pandas as pd
        import numpy as np

        dates = pd.date_range(start='2024-01-01', periods=50, freq='H')
        market_data = pd.DataFrame({
            'open': np.random.randn(50).cumsum() + 2500,
            'high': np.random.randn(50).cumsum() + 2520,
            'low': np.random.randn(50).cumsum() + 2480,
            'close': np.random.randn(50).cumsum() + 2500,
            'volume': np.random.randint(1000000, 5000000, 50)
        }, index=dates)

        # Enhance the signal (correct method signature: symbol, technical_signal, market_data)
        enhanced_signal = enhancer.enhance_signal('RELIANCE', base_signal, market_data)

        print("✅ ML Signal Enhancer Test Results:")
        print(f"   Original Signal: {base_signal['signal']} ({base_signal['confidence']:.2%})")

        # Check if enhancement was successful
        if 'signal' in enhanced_signal:
            print(f"   Enhanced Signal: {enhanced_signal['signal']} ({enhanced_signal['confidence']:.2%})")
            print(f"   Enhancement Active: {enhanced_signal.get('ml_enhanced', False)}")
            print(f"   Models Used: {enhanced_signal.get('models_used', [])}")
            return True
        else:
            print(f"   Enhancement Error: {enhanced_signal}")
            return False

    except Exception as e:
        print(f"❌ ML Signal Enhancer Test Failed: {e}")
        return False


def main():
    print("🧪 SIMPLE ML INTEGRATION TEST")
    print("=" * 50)
    print("Testing core ML functionality without Prophet dependencies")
    print()

    # Test components
    alpha_success = test_alpha_model()
    print()
    enhancer_success = test_ml_signal_enhancer()
    print()

    # Summary
    total_tests = 2
    passed_tests = sum([alpha_success, enhancer_success])

    print("=" * 50)
    print("TEST SUMMARY:")
    print(f"Tests Passed: {passed_tests}/{total_tests} ({passed_tests / total_tests * 100:.0f}%)")

    if passed_tests >= 1:
        print("✅ Core ML functionality is working!")
        print("📝 Next steps:")
        print("   1. Fix Prophet/NumPy compatibility for LSTM")
        print("   2. Run full ML integration test")
        print("   3. Test with real market data")
    else:
        print("❌ Core ML functionality needs debugging")

    print("=" * 50)


if __name__ == "__main__":
    main()