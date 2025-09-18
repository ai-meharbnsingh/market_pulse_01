# src/ai_trading/risk_calculator.py

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP


@dataclass
class RiskParameters:
    """Personal risk parameters for trading psychology compliance"""
    max_daily_loss_percent: float
    max_position_size_percent: float
    max_open_positions: int
    max_sector_allocation_percent: float
    kelly_fraction: float = 0.25  # Conservative Kelly multiplier


class KellyCalculator:
    """
    Kelly Criterion position sizing with psychological safeguards
    Integrates with AI framework for bias checking
    """

    def __init__(self, ai_engine=None, risk_params: RiskParameters = None):
        self.ai_engine = ai_engine
        self.risk_params = risk_params or self._default_risk_params()
        self.logger = logging.getLogger(__name__)

    def _default_risk_params(self) -> RiskParameters:
        """Conservative default risk parameters"""
        return RiskParameters(
            max_daily_loss_percent=2.0,
            max_position_size_percent=5.0,
            max_open_positions=6,
            max_sector_allocation_percent=30.0
        )

    def calculate_kelly_position_size(
            self,
            win_rate: float,
            avg_win_percent: float,
            avg_loss_percent: float,
            current_capital: float
    ) -> Dict[str, float]:
        """
        Calculate optimal position size using Kelly Criterion

        Args:
            win_rate: Historical win rate (0.0 to 1.0)
            avg_win_percent: Average winning trade percentage
            avg_loss_percent: Average losing trade percentage (positive number)
            current_capital: Current trading capital

        Returns:
            Dict with position sizing recommendations
        """
        try:
            # Kelly Criterion calculation
            # f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate

            if avg_loss_percent <= 0:
                raise ValueError("Average loss percentage must be positive")

            b = avg_win_percent / avg_loss_percent  # Win/Loss ratio
            p = win_rate  # Win probability
            q = 1 - win_rate  # Loss probability

            # Raw Kelly percentage
            kelly_percentage = (b * p - q) / b

            # Apply conservative multiplier (typically 25% of Kelly)
            conservative_kelly = kelly_percentage * self.risk_params.kelly_fraction

            # Apply personal risk limits
            position_percent = min(
                conservative_kelly,
                self.risk_params.max_position_size_percent / 100
            )

            # Calculate actual position size
            position_size = current_capital * max(0, position_percent)

            return {
                'raw_kelly_percent': round(kelly_percentage * 100, 2),
                'conservative_kelly_percent': round(conservative_kelly * 100, 2),
                'recommended_position_percent': round(position_percent * 100, 2),
                'recommended_position_size': round(position_size, 2),
                'risk_adjusted': position_percent < conservative_kelly,
                'within_limits': position_percent <= self.risk_params.max_position_size_percent / 100
            }

        except Exception as e:
            self.logger.error(f"Kelly calculation error: {e}")
            return self._safe_position_size(current_capital)

    def _safe_position_size(self, current_capital: float) -> Dict[str, float]:
        """Fallback safe position size when Kelly calculation fails"""
        safe_percent = min(2.0, self.risk_params.max_position_size_percent) / 100
        return {
            'raw_kelly_percent': 0.0,
            'conservative_kelly_percent': 0.0,
            'recommended_position_percent': safe_percent * 100,
            'recommended_position_size': current_capital * safe_percent,
            'risk_adjusted': True,
            'within_limits': True,
            'fallback_used': True
        }

    async def ai_psychology_check(self, trade_idea: Dict) -> Dict[str, str]:
        """
        Use AI to check for psychological biases in trading decision

        Args:
            trade_idea: Dict containing trade details

        Returns:
            AI analysis of potential psychological biases
        """
        if not self.ai_engine:
            return {"status": "no_ai", "message": "AI engine not configured"}

        psychology_prompt = f"""
        Analyze this trade idea for psychological biases:

        Stock: {trade_idea.get('symbol', 'Unknown')}
        Entry reason: {trade_idea.get('reason', 'Not specified')}
        Position size: {trade_idea.get('position_percent', 0)}% of capital
        Recent performance: {trade_idea.get('recent_pnl', 'Unknown')}

        Check for these biases:
        1. FOMO - Are they chasing momentum?
        2. Overconfidence - Is position size too large after recent wins?
        3. Revenge trading - Trying to recover from recent losses?
        4. Confirmation bias - Only seeing positive signals?

        Provide:
        1. Bias risk score (1-10, where 10 is highest risk)
        2. Primary concern if any
        3. Recommendation (proceed/reduce size/wait/avoid)
        4. One sentence reasoning
        """

        try:
            from antifragile_framework.providers.api_abstraction_layer import ChatMessage

            response = await self.ai_engine.execute_request(
                model_priority_map={
                    "anthropic": ["claude-3-5-sonnet-20240620"],
                    "openai": ["gpt-4o"],
                    "google_gemini": ["gemini-1.5-flash-latest"]
                },
                messages=[
                    ChatMessage(role="user", content=psychology_prompt)
                ],
                max_estimated_cost_usd=0.01,
                request_id=f"psychology_check_{trade_idea.get('symbol', 'unknown')}"
            )

            return {
                "status": "success",
                "ai_analysis": response.content,
                "model_used": response.model_used,
                "cost": getattr(response, 'estimated_cost_usd', 0)
            }

        except Exception as e:
            self.logger.error(f"AI psychology check failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "fallback_advice": "Proceed with caution - manual bias check required"
            }