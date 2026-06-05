"""
Electric Crab Extensions

This file adds three advanced capabilities:
1. Real Polymarket market data collector
2. Lightweight RL-style adaptive risk weighting
3. GPU batch scoring support with PyTorch
"""

import json
import math
from typing import List, Dict, Any, Optional

import httpx
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from electric_crab_core import MarketEvent


# =========================================================
# 1. Real Polymarket Data Collector
# =========================================================

class RealPolymarketCollector:
    """
    Fetch real public market data from Polymarket Gamma API.

    Current implementation uses:
    - event title
    - first market in each event
    - Yes price as market_probability
    - market volume
    - liquidity if available

    Some advanced features such as whale_ratio and sentiment_score
    still need extra data sources, so this class fills them with
    conservative default values for MVP demo.
    """

    BASE_URL = "https://gamma-api.polymarket.com"

    async def fetch_active_events(self, limit: int = 5) -> List[MarketEvent]:
        url = f"{self.BASE_URL}/events"

        params = {
            "limit": limit,
            "closed": "false",
            "active": "true"
        }

        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        events = []

        for raw_event in data:
            parsed = self._parse_event(raw_event)
            if parsed:
                events.append(parsed)

        return events

    def _parse_event(self, raw_event: Dict[str, Any]) -> Optional[MarketEvent]:
        event_id = str(raw_event.get("id", "unknown"))
        title = raw_event.get("title") or raw_event.get("question") or "Untitled Market"

        markets = raw_event.get("markets", [])
        if not markets:
            return None

        market = markets[0]

        market_probability = self._extract_yes_probability(market)
        volume = self._safe_float(
            market.get("volume")
            or market.get("volumeNum")
            or raw_event.get("volume")
            or raw_event.get("volumeNum")
            or 0
        )

        liquidity = self._safe_float(
            market.get("liquidity")
            or market.get("liquidityNum")
            or raw_event.get("liquidity")
            or raw_event.get("liquidityNum")
            or 0
        )

        # Convert liquidity into 0-1 score.
        # This is a demo heuristic. Tune it after collecting real data.
        liquidity_score = min(liquidity / 100_000, 1.0) if liquidity > 0 else 0.25

        # MVP placeholders.
        # Later:
        # - whale_ratio can come from holder concentration / wallet analytics
        # - volatility can come from price history
        # - sentiment_score can come from news/social models
        whale_ratio = 0.25
        volatility = abs(market_probability - 0.5)
        sentiment_score = 0.0

        return MarketEvent(
            event_id=event_id,
            title=title,
            market_probability=round(market_probability, 4),
            volume=round(volume, 2),
            whale_ratio=round(whale_ratio, 4),
            liquidity_score=round(liquidity_score, 4),
            volatility=round(volatility, 4),
            sentiment_score=round(sentiment_score, 4),
        )

    def _extract_yes_probability(self, market: Dict[str, Any]) -> float:
        """
        Polymarket market data often has outcomes and outcomePrices.
        Example:
        outcomes = ["Yes", "No"]
        outcomePrices = ["0.42", "0.58"]

        This method tries to find the Yes price.
        """

        outcomes = market.get("outcomes")
        prices = market.get("outcomePrices")

        outcomes = self._decode_json_array(outcomes)
        prices = self._decode_json_array(prices)

        if not outcomes or not prices:
            return 0.5

        for index, outcome in enumerate(outcomes):
            if str(outcome).lower() == "yes" and index < len(prices):
                return self._safe_probability(prices[index])

        # Fallback: first outcome price
        return self._safe_probability(prices[0])

    def _decode_json_array(self, value: Any) -> List[Any]:
        if value is None:
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []

        return []

    def _safe_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _safe_probability(self, value: Any) -> float:
        try:
            prob = float(value)
            return float(np.clip(prob, 0.01, 0.99))
        except (TypeError, ValueError):
            return 0.5


# =========================================================
# 2. Lightweight RL-Style Adaptive Weight Agent
# =========================================================

class AdaptiveRiskWeightAgent:
    """
    Lightweight RL-style adaptive module.

    Instead of using heavy Stable-Baselines3, this MVP uses a simple
    online reward update mechanism.

    Goal:
    - Reward the system if high-risk predictions are later confirmed.
    - Penalize the system if it overreacts or underreacts.
    - Slowly adapt feature weights.

    This is good enough for a competition demo and easy to explain.
    """

    def __init__(self):
        self.weights = {
            "deviation": 0.35,
            "whale_ratio": 0.25,
            "low_liquidity": 0.20,
            "volatility": 0.15,
            "sentiment": 0.05,
        }

        self.learning_rate = 0.05
        self.reward_history = []

    def calculate_adaptive_risk_points(
        self,
        deviation: float,
        whale_ratio: float,
        liquidity_score: float,
        volatility: float,
        sentiment_score: float
    ) -> float:
        """
        Convert market signals into adaptive risk points from 0 to 100.
        """

        normalized = {
            "deviation": min(deviation / 0.35, 1.0),
            "whale_ratio": min(whale_ratio / 0.85, 1.0),
            "low_liquidity": min((1.0 - liquidity_score), 1.0),
            "volatility": min(volatility / 0.45, 1.0),
            "sentiment": min(abs(sentiment_score), 1.0),
        }

        risk_score = 0.0

        for key, value in normalized.items():
            risk_score += self.weights[key] * value

        return round(float(np.clip(risk_score * 100, 0, 100)), 2)

    def update_policy(
        self,
        predicted_risk_level: str,
        actual_outcome_shift: float,
        factor_values: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        RL-style policy update.

        actual_outcome_shift:
        - Example: after market resolves or after 24h, compare old market price
          and later real/settled probability.
        - Larger shift means the earlier market was more suspicious.

        Reward logic:
        - If system predicted HIGH/MEDIUM and large shift happened, reward.
        - If system predicted LOW and large shift happened, penalize.
        - If system predicted HIGH but no shift happened, penalize slightly.
        """

        high_shift = actual_outcome_shift >= 0.20
        medium_shift = actual_outcome_shift >= 0.10

        if predicted_risk_level == "HIGH" and high_shift:
            reward = 1.0
        elif predicted_risk_level == "MEDIUM" and medium_shift:
            reward = 0.6
        elif predicted_risk_level == "LOW" and high_shift:
            reward = -1.0
        elif predicted_risk_level == "HIGH" and not medium_shift:
            reward = -0.5
        else:
            reward = 0.1

        self.reward_history.append(reward)

        # Update weights based on normalized feature contribution.
        total = sum(abs(v) for v in factor_values.values()) + 1e-9

        for key, value in factor_values.items():
            if key in self.weights:
                contribution = abs(value) / total
                self.weights[key] += self.learning_rate * reward * contribution

        # Normalize weights so they sum to 1.
        weight_sum = sum(max(v, 0.01) for v in self.weights.values())

        for key in self.weights:
            self.weights[key] = max(self.weights[key], 0.01) / weight_sum

        return {
            "reward": reward,
            "updated_weights": self.weights,
            "reward_history_size": len(self.reward_history)
        }


# =========================================================
# 3. GPU Batch Scorer
# =========================================================

class GPUBatchScorer:
    """
    GPU-accelerated batch probability scorer.

    This does not replace your RandomForest model.
    It gives you a GPU-ready scoring path for large event batches.

    In a future version, this can be replaced by:
    - PyTorch MLP
    - Transformer
    - LSTM
    - GNN
    """

    def __init__(self):
        self.enabled = TORCH_AVAILABLE
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Fast vectorized scoring.

        Input:
        X = standardized feature matrix

        Output:
        model probabilities from 0.01 to 0.99
        """

        if not self.enabled:
            return self._cpu_score(X)

        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

            # Simple GPU formula:
            # market probability signal + liquidity - whale - volatility + sentiment
            weights = torch.tensor(
                [0.20, 0.02, -0.18, 0.16, -0.12, 0.08],
                dtype=torch.float32,
                device=self.device
            )

            logits = tensor @ weights
            probs = torch.sigmoid(logits)

            probs = torch.clamp(probs, 0.01, 0.99)
            return probs.detach().cpu().numpy()

    def _cpu_score(self, X: np.ndarray) -> np.ndarray:
        weights = np.array([0.20, 0.02, -0.18, 0.16, -0.12, 0.08])
        logits = X @ weights
        probs = 1 / (1 + np.exp(-logits))
        return np.clip(probs, 0.01, 0.99)

    def status(self) -> Dict[str, Any]:
        return {
            "torch_available": TORCH_AVAILABLE,
            "device": self.device,
            "gpu_enabled": self.device == "cuda"
        }