"""
Electric Crab Extensions

Advanced capabilities:
1. Real Polymarket public market data collector
2. Deep learning probability scorer with PyTorch
3. Reinforcement-learning-style adaptive risk optimizer
4. GPU / CPU vectorized batch scoring
5. Async parallel utilities
6. xapi.to gateway integration

This file does NOT import electric_crab_core.py.
That avoids circular import issues.
"""

import os
import json
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


# =========================================================
# 1. Real Polymarket Data Collector
# =========================================================

class RealPolymarketCollector:
    """
    Fetch real public market data from Polymarket Gamma API.

    MVP fields:
    - event_id
    - title
    - market_probability
    - volume
    - whale_ratio
    - liquidity_score
    - volatility
    - sentiment_score
    - data_quality

    Current limitation:
    Some fields are heuristic or placeholder unless additional APIs are connected:
    - whale_ratio
    - sentiment_score
    - volatility
    """

    BASE_URL = "https://gamma-api.polymarket.com"

    async def fetch_active_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        if not HTTPX_AVAILABLE:
            raise RuntimeError(
                "httpx is not installed. Run: pip install httpx"
            )

        url = f"{self.BASE_URL}/events"

        params = {
            "limit": limit,
            "closed": "false",
            "active": "true",
            "order": "volume",
            "ascending": "false",
        }

        async with httpx.AsyncClient(timeout=25) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        results = []

        if not isinstance(data, list):
            return results

        for raw_event in data:
            parsed = self._parse_event(raw_event)
            if parsed:
                results.append(parsed)

        return results

    def _parse_event(self, raw_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        event_id = str(raw_event.get("id", "unknown"))

        title = (
            raw_event.get("title")
            or raw_event.get("question")
            or raw_event.get("slug")
            or "Untitled Market"
        )

        markets = raw_event.get("markets", [])

        if not markets or not isinstance(markets, list):
            return None

        market = markets[0]

        if not isinstance(market, dict):
            return None

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

        liquidity_score = min(liquidity / 100_000, 1.0) if liquidity > 0 else 0.25

        whale_ratio = self._estimate_whale_ratio_placeholder()
        volatility = self._estimate_volatility_from_probability(market_probability)
        sentiment_score = self._estimate_sentiment_placeholder()

        return {
            "event_id": event_id,
            "title": title,
            "market_probability": round(float(np.clip(market_probability, 0.01, 0.99)), 4),
            "volume": round(float(volume), 2),
            "whale_ratio": round(float(np.clip(whale_ratio, 0.0, 1.0)), 4),
            "liquidity_score": round(float(np.clip(liquidity_score, 0.0, 1.0)), 4),
            "volatility": round(float(np.clip(volatility, 0.0, 1.0)), 4),
            "sentiment_score": round(float(np.clip(sentiment_score, -1.0, 1.0)), 4),
            "data_quality": {
                "market_probability": "real_polymarket_gamma_api",
                "volume": "real_polymarket_gamma_api",
                "liquidity_score": "heuristic_from_gamma_liquidity",
                "whale_ratio": "placeholder_needs_onchain_wallet_data",
                "volatility": "heuristic_from_probability_uncertainty",
                "sentiment_score": "placeholder_needs_social_signal_api",
            },
        }

    def _extract_yes_probability(self, market: Dict[str, Any]) -> float:
        outcomes = self._decode_json_array(market.get("outcomes"))
        prices = self._decode_json_array(market.get("outcomePrices"))

        if outcomes and prices:
            for index, outcome in enumerate(outcomes):
                if str(outcome).lower() == "yes" and index < len(prices):
                    return self._safe_probability(prices[index])

            return self._safe_probability(prices[0])

        for key in [
            "lastTradePrice",
            "bestAsk",
            "bestBid",
            "price",
            "probability",
        ]:
            if key in market:
                return self._safe_probability(market.get(key))

        return 0.5

    def _estimate_whale_ratio_placeholder(self) -> float:
        return 0.25

    def _estimate_volatility_from_probability(self, market_probability: float) -> float:
        uncertainty = 1.0 - abs(market_probability - 0.5) * 2.0
        return min(max(uncertainty * 0.35, 0.02), 0.50)

    def _estimate_sentiment_placeholder(self) -> float:
        return 0.0

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
            return float(np.clip(float(value), 0.01, 0.99))
        except (TypeError, ValueError):
            return 0.5


# =========================================================
# 2. Deep Learning Probability Scorer
# =========================================================

class DeepLearningProbabilityScorer:
    """
    Lightweight deep learning probability model.

    Uses PyTorch MLP:
    input features -> hidden layers -> probability

    If PyTorch is not available, it falls back to a NumPy scorer.

    Current MVP uses synthetic targets.
    Production version should train on historical resolved markets.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 32,
        epochs: int = 120,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.enabled = TORCH_AVAILABLE
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.model = None
        self.is_trained = False

        if self.enabled:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            ).to(self.device)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.array([])

        if not self.enabled:
            return self._numpy_fallback(X)

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = self._build_synthetic_targets(X)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss_fn = nn.BCELoss()

        self.model.train()

        for _ in range(self.epochs):
            preds = self.model(X_tensor)
            loss = loss_fn(preds, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.is_trained = True

        self.model.eval()

        with torch.no_grad():
            probs = self.model(X_tensor).detach().cpu().numpy().reshape(-1)

        return np.clip(probs, 0.01, 0.99)

    def _build_synthetic_targets(self, X: np.ndarray) -> np.ndarray:
        targets = []

        for row in X:
            market_prob = row[0]
            volume_score = row[1]
            whale_ratio = row[2]
            liquidity_score = row[3]
            volatility = row[4]
            sentiment_scaled = row[5]

            sentiment = sentiment_scaled * 2 - 1

            prob = (
                0.45 * market_prob
                + 0.10 * volume_score
                - 0.18 * whale_ratio
                + 0.18 * liquidity_score
                - 0.12 * volatility
                + 0.08 * sentiment
                + 0.25
            )

            targets.append(float(np.clip(prob, 0.01, 0.99)))

        return np.array(targets)

    def _numpy_fallback(self, X: np.ndarray) -> np.ndarray:
        weights = np.array([0.45, 0.10, -0.18, 0.18, -0.12, 0.08])
        bias = 0.25
        logits = X @ weights + bias
        probs = 1 / (1 + np.exp(-logits))
        return np.clip(probs, 0.01, 0.99)

    def status(self) -> Dict[str, Any]:
        return {
            "torch_available": TORCH_AVAILABLE,
            "device": self.device,
            "gpu_enabled": self.device == "cuda",
            "model_type": "PyTorch MLP" if self.enabled else "NumPy fallback",
            "is_trained": self.is_trained,
            "training_data": "synthetic_targets_for_mvp_demo",
        }


# =========================================================
# 3. RL-Style Adaptive Risk Optimizer
# =========================================================

class RLAdaptiveRiskOptimizer:
    """
    Lightweight reinforcement-learning-style risk optimizer.

    This is not a full PPO/DQN implementation.
    It is a demo-friendly online policy update mechanism.

    It adapts feature weights based on reward feedback after market resolution.
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

    def calculate_risk_points(
        self,
        deviation: float,
        whale_ratio: float,
        liquidity_score: float,
        volatility: float,
        sentiment_score: float,
        agent_disagreement: float = 0.0,
    ) -> float:
        normalized = {
            "deviation": min(deviation / 0.35, 1.0),
            "whale_ratio": min(whale_ratio / 0.85, 1.0),
            "low_liquidity": min(1.0 - liquidity_score, 1.0),
            "volatility": min(volatility / 0.5, 1.0),
            "sentiment": min(abs(sentiment_score), 1.0),
        }

        risk_score = 0.0

        for key, value in normalized.items():
            risk_score += self.weights[key] * value

        risk_score += min(agent_disagreement / 0.25, 1.0) * 0.10

        return round(float(np.clip(risk_score * 100, 0, 100)), 2)

    def update_policy(
        self,
        predicted_risk_level: str,
        actual_outcome_shift: float,
        factor_values: Dict[str, float],
    ) -> Dict[str, Any]:
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

        total = sum(abs(v) for v in factor_values.values()) + 1e-9

        for key, value in factor_values.items():
            if key in self.weights:
                contribution = abs(value) / total
                self.weights[key] += self.learning_rate * reward * contribution

        weight_sum = sum(max(v, 0.01) for v in self.weights.values())

        for key in self.weights:
            self.weights[key] = max(self.weights[key], 0.01) / weight_sum

        return {
            "reward": reward,
            "updated_weights": self.weights,
            "reward_history_size": len(self.reward_history),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "learning_rate": self.learning_rate,
            "reward_history_size": len(self.reward_history),
            "optimizer_type": "rl_style_online_weight_update",
        }


# =========================================================
# 4. GPU / CPU Batch Scorer
# =========================================================

class GPUBatchScorer:
    """
    GPU vectorized scoring path.

    If PyTorch or CUDA is not available, it falls back to CPU scoring.
    """

    def __init__(self):
        self.enabled = TORCH_AVAILABLE
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.array([])

        if not self.enabled:
            return self._cpu_score(X)

        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

            weights = torch.tensor(
                [0.42, 0.08, -0.16, 0.18, -0.10, 0.08],
                dtype=torch.float32,
                device=self.device,
            )

            logits = tensor @ weights + 0.15
            probs = torch.sigmoid(logits)

            return torch.clamp(probs, 0.01, 0.99).detach().cpu().numpy()

    def _cpu_score(self, X: np.ndarray) -> np.ndarray:
        weights = np.array([0.42, 0.08, -0.16, 0.18, -0.10, 0.08])
        logits = X @ weights + 0.15
        probs = 1 / (1 + np.exp(-logits))
        return np.clip(probs, 0.01, 0.99)

    def status(self) -> Dict[str, Any]:
        return {
            "torch_available": TORCH_AVAILABLE,
            "device": self.device,
            "gpu_enabled": self.device == "cuda",
            "scorer_type": "Torch vectorized scorer" if self.enabled else "NumPy CPU scorer",
        }


# =========================================================
# 5. Async Parallel Utility
# =========================================================

class ParallelBatchRunner:
    """
    Simple concurrency utility.

    Useful for:
    - fetching multiple markets
    - running parallel lightweight tasks
    """

    def __init__(self, concurrency: int = 10):
        self.concurrency = concurrency

    async def run(self, tasks):
        semaphore = asyncio.Semaphore(self.concurrency)

        async def limited_task(task):
            async with semaphore:
                return await task

        return await asyncio.gather(*(limited_task(task) for task in tasks))


# =========================================================
# 6. xapi.to Gateway Client
# =========================================================

class XAPIGatewayClient:
    """
    xapi.to gateway client for Electric Crab.

    This client builds xapi.to Skill / CLI tasks that can be shown in demo or executed.

    Supported tasks:
    - @Mention All audit notification
    - Real-time market research
    - Twitter/X social signal research
    - Crypto price lookup

    Environment variables:
    - XAPI_API_KEY
    - XAPI_ENABLE_CLI=true
    """

    def __init__(self):
        self.api_key = os.getenv("XAPI_API_KEY", "")
        self.enable_cli = os.getenv("XAPI_ENABLE_CLI", "false").lower() == "true"

    def build_mention_all_task(
        self,
        project_name: str,
        audit_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        summary = self._build_audit_summary(audit_results)

        xapi_prompt = (
            "/xapi\n"
            "通过可用的社交、消息或通知 API 发送下面这条项目审计通知。"
            "如果当前没有可用发送通道，就返回可复制的通知内容。\n\n"
            f"@Mention All\n\n"
            f"{summary}"
        )

        return {
            "gateway": "xapi.to",
            "type": "xapi_gateway_task",
            "project": project_name,
            "capability_intent": "social_or_notification_send",
            "mention": "@Mention All",
            "xapi_prompt": xapi_prompt,
            "message": f"@Mention All\n\n{summary}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "Electric Crab – The Market Guardian",
        }

    def build_market_research_task(
        self,
        market_title: str,
    ) -> Dict[str, Any]:
        xapi_prompt = (
            "/xapi\n"
            f"搜索这个预测市场相关的最新公开信息，并总结可能影响概率的关键因素：{market_title}"
        )

        return {
            "gateway": "xapi.to",
            "type": "xapi_gateway_task",
            "capability_intent": "web.search.realtime",
            "market_title": market_title,
            "xapi_prompt": xapi_prompt,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "Electric Crab – The Market Guardian",
        }

    def build_twitter_signal_task(
        self,
        query: str,
    ) -> Dict[str, Any]:
        xapi_prompt = (
            "/xapi\n"
            f"搜索 Twitter/X 上关于「{query}」的最新讨论，提取情绪、热度和关键观点。"
        )

        return {
            "gateway": "xapi.to",
            "type": "xapi_gateway_task",
            "capability_intent": "twitter.user_tweets_or_search",
            "query": query,
            "xapi_prompt": xapi_prompt,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "Electric Crab – The Market Guardian",
        }

    def build_crypto_price_task(
        self,
        token_symbol: str,
    ) -> Dict[str, Any]:
        token_symbol = token_symbol.upper().strip()

        xapi_prompt = (
            "/xapi\n"
            f"查询 {token_symbol} 的最新价格、24h 变化和基础 token metadata。"
        )

        return {
            "gateway": "xapi.to",
            "type": "xapi_gateway_task",
            "capability_intent": "crypto.token.price",
            "token_symbol": token_symbol,
            "xapi_prompt": xapi_prompt,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "Electric Crab – The Market Guardian",
        }

    async def run_task(
        self,
        task: Dict[str, Any],
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        if dry_run:
            return {
                "executed": False,
                "dry_run": True,
                "reason": "Dry run mode. Use argument 'send' to attempt real xapi.to CLI execution.",
                "task": task,
            }

        if not self.enable_cli:
            return {
                "executed": False,
                "dry_run": False,
                "reason": "XAPI_ENABLE_CLI is not true. Set $env:XAPI_ENABLE_CLI='true'.",
                "task": task,
            }

        if not self.api_key:
            return {
                "executed": False,
                "dry_run": False,
                "reason": "XAPI_API_KEY is missing. Set $env:XAPI_API_KEY='your_key'.",
                "task": task,
            }

        try:
            process = await asyncio.create_subprocess_exec(
                "npx",
                "xapi-to",
                "run",
                "--input",
                json.dumps(task, ensure_ascii=False),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            return {
                "executed": process.returncode == 0,
                "dry_run": False,
                "return_code": process.returncode,
                "stdout": stdout.decode("utf-8", errors="ignore")[:2000],
                "stderr": stderr.decode("utf-8", errors="ignore")[:2000],
                "task": task,
            }

        except Exception as exc:
            return {
                "executed": False,
                "dry_run": False,
                "error": str(exc),
                "task": task,
            }

    def _build_audit_summary(
        self,
        audit_results: List[Dict[str, Any]],
    ) -> str:
        if not audit_results:
            return "Electric Crab completed an audit, but no market results were generated."

        high_count = sum(1 for item in audit_results if item.get("risk_level") == "HIGH")
        medium_count = sum(1 for item in audit_results if item.get("risk_level") == "MEDIUM")
        low_count = sum(1 for item in audit_results if item.get("risk_level") == "LOW")

        lines = [
            "Electric Crab Multi-Agent Polymarket Audit Completed.",
            "",
            f"Audited markets: {len(audit_results)}",
            f"Risk summary: HIGH={high_count}, MEDIUM={medium_count}, LOW={low_count}",
            "",
            "Top results:",
        ]

        for index, item in enumerate(audit_results[:5], start=1):
            title = item.get("title", "Untitled Market")
            market_prob = round(item.get("market_probability", 0) * 100, 2)
            model_prob = round(item.get("model_probability", 0) * 100, 2)
            predicted_outcome = item.get("predicted_outcome", "UNKNOWN")
            outcome_confidence = round(item.get("outcome_confidence", 0) * 100, 2)
            deviation = round(item.get("deviation", 0) * 100, 2)
            risk = item.get("risk_level", "UNKNOWN")
            trust = item.get("trust_score", "N/A")
            prediction_hash = item.get("prediction_hash", "N/A")
            factors = item.get("main_factors", [])

            if isinstance(factors, list):
                factors_text = ", ".join(factors[:3])
            else:
                factors_text = str(factors)

            lines.append(
                f"{index}. {title}\n"
                f"   Prediction: {predicted_outcome} ({outcome_confidence}%)\n"
                f"   Market Probability: {market_prob}%\n"
                f"   Electric Crab Probability: {model_prob}%\n"
                f"   Deviation: {deviation}%\n"
                f"   Risk: {risk}\n"
                f"   Trust Score: {trust}\n"
                f"   Factors: {factors_text}\n"
                f"   Prediction Hash: {prediction_hash}"
            )

        lines.append("")
        lines.append("Powered by xapi.to gateway + Electric Crab multi-agent audit system.")
        lines.append("Showcase only. Not financial advice.")

        return "\n".join(lines)


# =========================================================
# 7. Module Health Check
# =========================================================

def extension_health_check() -> Dict[str, Any]:
    """
    Quick diagnostic helper.

    Usage:
    python -c "from electric_crab_extensions import extension_health_check; print(extension_health_check())"
    """

    return {
        "extensions_loaded": True,
        "httpx_available": HTTPX_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "xapi_gateway_available": True,
        "classes": [
            "RealPolymarketCollector",
            "DeepLearningProbabilityScorer",
            "RLAdaptiveRiskOptimizer",
            "GPUBatchScorer",
            "ParallelBatchRunner",
            "XAPIGatewayClient",
        ],
    }