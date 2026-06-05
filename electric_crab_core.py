"""
Electric Crab – The Market Guardian

Multi-Agent AI Prediction Market System

Features:
- Simulated prediction market audit
- Real Polymarket Gamma API support
- Multi-agent probability estimation
- Classical ML probability model
- Deep learning probability model through extensions
- RL-style adaptive risk optimizer through extensions
- GPU / CPU batch scoring through extensions
- xapi.to gateway task generation
- @Mention All audit notification task
- xAPI / xapi.to showcase output

Command examples:
    python electric_crab_core.py
    python electric_crab_core.py notify
    python electric_crab_core.py notify send
    python electric_crab_core.py real
    python electric_crab_core.py real notify
    python electric_crab_core.py real notify send
"""

import os
import json
import random
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# xapi.to Config
# =========================================================

# Do NOT hardcode real API keys in code submitted to GitHub.
# Set it in PowerShell:
# $env:XAPI_API_KEY="your_key_here"
# $env:XAPI_ENABLE_CLI="true"

os.environ.setdefault("XAPI_ENABLE_CLI", "false")


# =========================================================
# Data Schema
# =========================================================

@dataclass
class MarketEvent:
    event_id: str
    title: str
    market_probability: float
    volume: float
    whale_ratio: float
    liquidity_score: float
    volatility: float
    sentiment_score: float


@dataclass
class AuditResult:
    event_id: str
    title: str
    market_probability: float
    model_probability: float
    deviation: float
    risk_level: str
    trust_score: float
    main_factors: List[str]
    tldr: str
    xapi: Dict[str, Any]


# =========================================================
# Import Extensions
# =========================================================

try:
    from electric_crab_extensions import (
        RealPolymarketCollector,
        DeepLearningProbabilityScorer,
        RLAdaptiveRiskOptimizer,
        GPUBatchScorer,
        XAPIGatewayClient,
    )
    EXTENSIONS_AVAILABLE = True
except ImportError:
    EXTENSIONS_AVAILABLE = False

    RealPolymarketCollector = None
    DeepLearningProbabilityScorer = None
    RLAdaptiveRiskOptimizer = None
    GPUBatchScorer = None
    XAPIGatewayClient = None


# =========================================================
# Fallback xapi.to Gateway Client
# =========================================================

class FallbackXAPIGatewayClient:
    """
    Fallback xapi.to client.

    Used only when electric_crab_extensions.py does not provide XAPIGatewayClient.
    It builds xapi.to tasks and can optionally try CLI execution.
    """

    def __init__(self):
        self.api_key = os.getenv("XAPI_API_KEY", "")
        self.enable_cli = os.getenv("XAPI_ENABLE_CLI", "false").lower() == "true"

    def build_mention_all_task(
        self,
        project_name: str,
        audit_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        summary = self._build_audit_summary(audit_results)

        xapi_prompt = (
            "/xapi\n"
            "通过可用的社交、消息或通知 API 发送下面这条项目审计通知。"
            "如果没有可用发送通道，就返回可复制的通知内容。\n\n"
            f"@Mention All\n\n{summary}"
        )

        return {
            "gateway": "xapi.to",
            "type": "xapi_gateway_task",
            "project": project_name,
            "capability_intent": "social_or_notification_send",
            "mention": "@Mention All",
            "xapi_prompt": xapi_prompt,
            "message": f"@Mention All\n\n{summary}",
            "source": "Electric Crab – The Market Guardian",
        }

    def build_market_research_task(self, market_title: str) -> Dict[str, Any]:
        return {
            "gateway": "xapi.to",
            "type": "xapi_gateway_task",
            "capability_intent": "web.search.realtime",
            "market_title": market_title,
            "xapi_prompt": (
                "/xapi\n"
                f"搜索这个预测市场相关的最新公开信息，并总结可能影响概率的关键因素：{market_title}"
            ),
        }

    def build_twitter_signal_task(self, query: str) -> Dict[str, Any]:
        return {
            "gateway": "xapi.to",
            "type": "xapi_gateway_task",
            "capability_intent": "twitter.user_tweets_or_search",
            "query": query,
            "xapi_prompt": (
                "/xapi\n"
                f"搜索 Twitter/X 上关于「{query}」的最新讨论，提取情绪、热度和关键观点。"
            ),
        }

    def build_crypto_price_task(self, token_symbol: str) -> Dict[str, Any]:
        return {
            "gateway": "xapi.to",
            "type": "xapi_gateway_task",
            "capability_intent": "crypto.token.price",
            "token_symbol": token_symbol,
            "xapi_prompt": (
                "/xapi\n"
                f"查询 {token_symbol} 的最新价格、24h 变化和基础 token metadata。"
            ),
        }

    async def run_task(self, task: Dict[str, Any], dry_run: bool = True) -> Dict[str, Any]:
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

    def _build_audit_summary(self, audit_results: List[Dict[str, Any]]) -> str:
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
            deviation = round(item.get("deviation", 0) * 100, 2)
            risk = item.get("risk_level", "UNKNOWN")
            trust = item.get("trust_score", "N/A")

            lines.append(
                f"{index}. {title}\n"
                f"   Market Probability: {market_prob}%\n"
                f"   Electric Crab Probability: {model_prob}%\n"
                f"   Deviation: {deviation}%\n"
                f"   Risk: {risk}\n"
                f"   Trust Score: {trust}"
            )

        lines.append("")
        lines.append("Powered by xapi.to gateway + Electric Crab multi-agent audit system.")
        lines.append("Showcase only. Not financial advice.")

        return "\n".join(lines)


# =========================================================
# Demo Data Collector
# =========================================================

class DemoDataCollector:
    async def fetch_event(self, event_id: str, title: str) -> MarketEvent:
        await asyncio.sleep(0.03)

        return MarketEvent(
            event_id=event_id,
            title=title,
            market_probability=round(random.uniform(0.20, 0.85), 4),
            volume=round(random.uniform(10_000, 1_000_000), 2),
            whale_ratio=round(random.uniform(0.05, 0.85), 4),
            liquidity_score=round(random.uniform(0.20, 1.00), 4),
            volatility=round(random.uniform(0.02, 0.45), 4),
            sentiment_score=round(random.uniform(-1.00, 1.00), 4),
        )

    async def fetch_batch(self, events: List[Dict[str, str]]) -> List[MarketEvent]:
        tasks = [
            self.fetch_event(event["event_id"], event["title"])
            for event in events
        ]

        return await asyncio.gather(*tasks)


# =========================================================
# Feature Engineering
# =========================================================

class FeatureEngineer:
    feature_names = [
        "market_probability",
        "volume_score",
        "whale_ratio",
        "liquidity_score",
        "volatility",
        "sentiment_scaled",
    ]

    def transform_events(self, events: List[MarketEvent]) -> np.ndarray:
        return np.array([self.transform_one(event) for event in events], dtype=float)

    def transform_one(self, event: MarketEvent) -> np.ndarray:
        market_probability = float(np.clip(event.market_probability, 0.01, 0.99))
        volume_score = float(np.clip(np.log10(event.volume + 1) / 6.0, 0.0, 1.0))
        whale_ratio = float(np.clip(event.whale_ratio, 0.0, 1.0))
        liquidity_score = float(np.clip(event.liquidity_score, 0.0, 1.0))
        volatility = float(np.clip(event.volatility, 0.0, 1.0))
        sentiment_scaled = float(np.clip((event.sentiment_score + 1.0) / 2.0, 0.0, 1.0))

        return np.array(
            [
                market_probability,
                volume_score,
                whale_ratio,
                liquidity_score,
                volatility,
                sentiment_scaled,
            ],
            dtype=float,
        )


# =========================================================
# Classical ML Probability Model
# =========================================================

class ClassicalMLProbabilityModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=120,
            max_depth=6,
            random_state=42,
        )
        self.is_trained = False

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.array([])

        y = self._build_synthetic_targets(X)
        self.model.fit(X, y)
        self.is_trained = True

        preds = self.model.predict(X)
        return np.clip(preds, 0.01, 0.99)

    def _build_synthetic_targets(self, X: np.ndarray) -> np.ndarray:
        targets = []

        for row in X:
            market_probability = row[0]
            volume_score = row[1]
            whale_ratio = row[2]
            liquidity_score = row[3]
            volatility = row[4]
            sentiment_scaled = row[5]
            sentiment = sentiment_scaled * 2 - 1

            prob = (
                0.50 * market_probability
                + 0.08 * volume_score
                - 0.14 * whale_ratio
                + 0.16 * liquidity_score
                - 0.10 * volatility
                + 0.07 * sentiment
                + 0.20
            )

            targets.append(float(np.clip(prob, 0.01, 0.99)))

        return np.array(targets)


# =========================================================
# Multi-Agent Specialist Agents
# =========================================================

class PriceAgent:
    name = "PriceAgent"

    def analyze(self, event: MarketEvent) -> Dict[str, Any]:
        return {
            "agent": self.name,
            "probability": event.market_probability,
            "confidence": 0.55,
            "signal": "MARKET_PRICE_BASELINE",
            "explanation": "Uses market implied probability as the baseline.",
        }


class LiquidityAgent:
    name = "LiquidityAgent"

    def analyze(self, event: MarketEvent) -> Dict[str, Any]:
        if event.liquidity_score < 0.35:
            probability = event.market_probability * 0.65 + 0.5 * 0.35
            confidence = 0.78
            signal = "LOW_LIQUIDITY_WARNING"
            explanation = "Low liquidity can make the market easier to distort."
        else:
            probability = event.market_probability
            confidence = 0.58
            signal = "LIQUIDITY_NORMAL"
            explanation = "Liquidity does not trigger a major warning."

        return {
            "agent": self.name,
            "probability": round(float(np.clip(probability, 0.01, 0.99)), 4),
            "confidence": confidence,
            "signal": signal,
            "explanation": explanation,
        }


class WhaleAgent:
    name = "WhaleAgent"

    def analyze(self, event: MarketEvent) -> Dict[str, Any]:
        if event.whale_ratio >= 0.55:
            probability = event.market_probability * 0.55 + 0.5 * 0.45
            confidence = 0.82
            signal = "WHALE_CONCENTRATION_WARNING"
            explanation = "High whale concentration may indicate concentrated pressure or manipulation risk."
        else:
            probability = event.market_probability
            confidence = 0.55
            signal = "WHALE_RISK_NORMAL"
            explanation = "No strong whale concentration warning detected."

        return {
            "agent": self.name,
            "probability": round(float(np.clip(probability, 0.01, 0.99)), 4),
            "confidence": confidence,
            "signal": signal,
            "explanation": explanation,
        }


class VolatilityAgent:
    name = "VolatilityAgent"

    def analyze(self, event: MarketEvent) -> Dict[str, Any]:
        if event.volatility >= 0.30:
            probability = event.market_probability * 0.60 + 0.5 * 0.40
            confidence = 0.72
            signal = "HIGH_VOLATILITY_WARNING"
            explanation = "High volatility indicates unstable market pricing."
        else:
            probability = event.market_probability
            confidence = 0.50
            signal = "VOLATILITY_NORMAL"
            explanation = "Volatility does not trigger a major warning."

        return {
            "agent": self.name,
            "probability": round(float(np.clip(probability, 0.01, 0.99)), 4),
            "confidence": confidence,
            "signal": signal,
            "explanation": explanation,
        }


class SentimentAgent:
    name = "SentimentAgent"

    def analyze(self, event: MarketEvent) -> Dict[str, Any]:
        probability = event.market_probability + 0.08 * event.sentiment_score

        if abs(event.sentiment_score) >= 0.70:
            confidence = 0.70
            signal = "EXTREME_SENTIMENT_WARNING"
            explanation = "Extreme sentiment may signal hype, panic, or narrative-driven distortion."
        else:
            confidence = 0.45
            signal = "SENTIMENT_NORMAL"
            explanation = "Sentiment signal appears moderate."

        return {
            "agent": self.name,
            "probability": round(float(np.clip(probability, 0.01, 0.99)), 4),
            "confidence": confidence,
            "signal": signal,
            "explanation": explanation,
        }


class ClassicalMLAgent:
    name = "ClassicalMLAgent"

    def analyze(self, probability: float) -> Dict[str, Any]:
        return {
            "agent": self.name,
            "probability": round(float(np.clip(probability, 0.01, 0.99)), 4),
            "confidence": 0.72,
            "signal": "CLASSICAL_ML_ESTIMATE",
            "explanation": "RandomForest estimates independent probability from engineered features.",
        }


class DeepLearningAgent:
    name = "DeepLearningAgent"

    def analyze(self, probability: float, enabled: bool) -> Dict[str, Any]:
        return {
            "agent": self.name,
            "probability": round(float(np.clip(probability, 0.01, 0.99)), 4),
            "confidence": 0.80 if enabled else 0.50,
            "signal": "DEEP_LEARNING_ESTIMATE" if enabled else "DEEP_LEARNING_FALLBACK",
            "explanation": "Neural network estimates probability from market features.",
        }


class GPUScoringAgent:
    name = "GPUScoringAgent"

    def analyze(self, probability: float, enabled: bool) -> Dict[str, Any]:
        return {
            "agent": self.name,
            "probability": round(float(np.clip(probability, 0.01, 0.99)), 4),
            "confidence": 0.65,
            "signal": "GPU_BATCH_SCORE" if enabled else "CPU_BATCH_SCORE",
            "explanation": "Vectorized batch scoring path for scalable market monitoring.",
        }


class XAPIGatewayAgent:
    name = "XAPIGatewayAgent"

    def analyze(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "agent": self.name,
            "capability": "xapi.to external API gateway",
            "task_count": len(tasks),
            "signal": "XAPI_GATEWAY_READY" if tasks else "XAPI_NO_TASKS",
            "explanation": (
                "Prepares xapi.to tasks for real-time research, social intelligence, "
                "crypto data lookup, and @Mention All audit notification."
            ),
        }


# =========================================================
# Multi-Agent Coordinator
# =========================================================

class MultiAgentCoordinator:
    def __init__(self):
        self.price_agent = PriceAgent()
        self.liquidity_agent = LiquidityAgent()
        self.whale_agent = WhaleAgent()
        self.volatility_agent = VolatilityAgent()
        self.sentiment_agent = SentimentAgent()
        self.classical_agent = ClassicalMLAgent()
        self.deep_agent = DeepLearningAgent()
        self.gpu_agent = GPUScoringAgent()

    def aggregate(
        self,
        event: MarketEvent,
        classical_probability: float,
        deep_probability: float,
        gpu_probability: float,
        deep_enabled: bool,
        gpu_enabled: bool,
    ) -> Dict[str, Any]:
        votes = [
            self.price_agent.analyze(event),
            self.liquidity_agent.analyze(event),
            self.whale_agent.analyze(event),
            self.volatility_agent.analyze(event),
            self.sentiment_agent.analyze(event),
            self.classical_agent.analyze(classical_probability),
            self.deep_agent.analyze(deep_probability, deep_enabled),
            self.gpu_agent.analyze(gpu_probability, gpu_enabled),
        ]

        weighted_sum = 0.0
        confidence_sum = 0.0

        for vote in votes:
            weighted_sum += vote["probability"] * vote["confidence"]
            confidence_sum += vote["confidence"]

        ensemble_probability = weighted_sum / confidence_sum
        ensemble_probability = round(float(np.clip(ensemble_probability, 0.01, 0.99)), 4)

        probabilities = [vote["probability"] for vote in votes]
        disagreement = round(float(np.std(probabilities)), 4)

        if disagreement >= 0.18:
            consensus_level = "LOW_CONSENSUS"
        elif disagreement >= 0.08:
            consensus_level = "MEDIUM_CONSENSUS"
        else:
            consensus_level = "HIGH_CONSENSUS"

        warning_signals = [
            vote["signal"]
            for vote in votes
            if "WARNING" in vote["signal"]
        ]

        if not warning_signals:
            warning_signals = ["NO_MAJOR_WARNING"]

        return {
            "ensemble_probability": ensemble_probability,
            "agent_votes": votes,
            "agent_disagreement": disagreement,
            "consensus_level": consensus_level,
            "warning_signals": warning_signals,
        }


# =========================================================
# Risk Engine
# =========================================================

class RiskEngine:
    def score(
        self,
        event: MarketEvent,
        model_probability: float,
        agent_disagreement: float = 0.0,
        rl_optimizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        deviation = abs(event.market_probability - model_probability)
        factors = []

        if deviation >= 0.25:
            factors.append("Large probability deviation")
        elif deviation >= 0.12:
            factors.append("Moderate probability deviation")

        if event.whale_ratio >= 0.55:
            factors.append("High whale concentration")

        if event.liquidity_score <= 0.35:
            factors.append("Low liquidity")

        if event.volatility >= 0.30:
            factors.append("High market volatility")

        if abs(event.sentiment_score) >= 0.70:
            factors.append("Extreme sentiment signal")

        if agent_disagreement >= 0.18:
            factors.append("Low multi-agent consensus")

        if rl_optimizer is not None:
            risk_points = rl_optimizer.calculate_risk_points(
                deviation=deviation,
                whale_ratio=event.whale_ratio,
                liquidity_score=event.liquidity_score,
                volatility=event.volatility,
                sentiment_score=event.sentiment_score,
                agent_disagreement=agent_disagreement,
            )
        else:
            risk_points = 0

            if deviation >= 0.25:
                risk_points += 35
            elif deviation >= 0.12:
                risk_points += 20

            if event.whale_ratio >= 0.55:
                risk_points += 25

            if event.liquidity_score <= 0.35:
                risk_points += 20

            if event.volatility >= 0.30:
                risk_points += 15

            if abs(event.sentiment_score) >= 0.70:
                risk_points += 10

            if agent_disagreement >= 0.18:
                risk_points += 15

            risk_points = min(risk_points, 100)

        trust_score = round(max(0, 100 - risk_points), 2)

        if risk_points >= 65:
            risk_level = "HIGH"
        elif risk_points >= 35:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        if not factors:
            factors.append("Market probability is broadly aligned with model estimate")

        return {
            "deviation": round(float(deviation), 4),
            "risk_points": round(float(risk_points), 2),
            "risk_level": risk_level,
            "trust_score": trust_score,
            "main_factors": factors,
        }


# =========================================================
# Explainer
# =========================================================

class Explainer:
    def generate_tldr(
        self,
        event: MarketEvent,
        model_probability: float,
        risk_level: str,
        factors: List[str],
    ) -> str:
        market_pct = round(event.market_probability * 100, 2)
        model_pct = round(model_probability * 100, 2)
        factor_text = ", ".join(factors[:3])

        return (
            f"Electric Crab flags this market as {risk_level}. "
            f"The market implies {market_pct}% while the multi-agent model estimates {model_pct}%. "
            f"Main drivers: {factor_text}."
        )

    def decision_showcase(
        self,
        event: MarketEvent,
        model_probability: float,
        risk_level: str,
        trust_score: float,
    ) -> Dict[str, Any]:
        signed_deviation = model_probability - event.market_probability

        if signed_deviation >= 0.10:
            signal = "MARKET_MAY_BE_UNDERPRICED"
            insight = "The multi-agent model estimates a higher probability than the market."
        elif signed_deviation <= -0.10:
            signal = "MARKET_MAY_BE_OVERPRICED"
            insight = "The multi-agent model estimates a lower probability than the market."
        else:
            signal = "MARKET_FAIRLY_ALIGNED"
            insight = "The market probability and model estimate are broadly aligned."

        if risk_level == "HIGH":
            action_label = "High caution"
        elif risk_level == "MEDIUM":
            action_label = "Review carefully"
        else:
            action_label = "Low concern"

        return {
            "decision_signal": signal,
            "action_label": action_label,
            "insight": insight,
            "trust_score": trust_score,
            "disclaimer": "Showcase only. Not financial advice.",
        }


# =========================================================
# Electric Crab Agent
# =========================================================

class ElectricCrabAgent:
    def __init__(
        self,
        use_real_data: bool = False,
        use_deep_learning: bool = True,
        use_reinforcement_learning: bool = True,
        use_gpu: bool = True,
    ):
        self.use_real_data = use_real_data
        self.use_deep_learning = use_deep_learning
        self.use_reinforcement_learning = use_reinforcement_learning
        self.use_gpu = use_gpu

        self.demo_collector = DemoDataCollector()
        self.real_collector = None

        self.feature_engineer = FeatureEngineer()
        self.classical_model = ClassicalMLProbabilityModel()
        self.coordinator = MultiAgentCoordinator()
        self.risk_engine = RiskEngine()
        self.explainer = Explainer()
        self.xapi_agent = XAPIGatewayAgent()

        self.deep_model = None
        self.rl_optimizer = None
        self.gpu_scorer = None
        self.xapi_gateway = None

        if EXTENSIONS_AVAILABLE:
            if use_real_data and RealPolymarketCollector is not None:
                self.real_collector = RealPolymarketCollector()

            if use_deep_learning and DeepLearningProbabilityScorer is not None:
                self.deep_model = DeepLearningProbabilityScorer(input_dim=6)

            if use_reinforcement_learning and RLAdaptiveRiskOptimizer is not None:
                self.rl_optimizer = RLAdaptiveRiskOptimizer()

            if use_gpu and GPUBatchScorer is not None:
                self.gpu_scorer = GPUBatchScorer()

            if XAPIGatewayClient is not None:
                self.xapi_gateway = XAPIGatewayClient()

        if self.xapi_gateway is None:
            self.xapi_gateway = FallbackXAPIGatewayClient()

    async def audit_demo_events(self, events: List[Dict[str, str]]) -> List[AuditResult]:
        market_events = await self.demo_collector.fetch_batch(events)
        return self._audit_market_events(
            market_events=market_events,
            data_source="simulated",
        )

    async def audit_real_polymarket_events(self, limit: int = 5) -> List[AuditResult]:
        if self.real_collector is None:
            raise RuntimeError(
                "Real Polymarket collector is not enabled. "
                "Use ElectricCrabAgent(use_real_data=True) and make sure extensions are installed."
            )

        raw_events = await self.real_collector.fetch_active_events(limit=limit)

        market_events = [
            MarketEvent(**raw_event)
            for raw_event in raw_events
        ]

        return self._audit_market_events(
            market_events=market_events,
            data_source="polymarket_gamma_api",
        )

    def _audit_market_events(
        self,
        market_events: List[MarketEvent],
        data_source: str,
    ) -> List[AuditResult]:
        if not market_events:
            return []

        X = self.feature_engineer.transform_events(market_events)

        classical_probs = self.classical_model.fit_predict(X)

        if self.deep_model is not None:
            deep_probs = self.deep_model.fit_predict(X)
            deep_status = self.deep_model.status()
            deep_enabled = True
        else:
            deep_probs = classical_probs
            deep_status = {"enabled": False, "reason": "Deep model not available."}
            deep_enabled = False

        if self.gpu_scorer is not None:
            gpu_probs = self.gpu_scorer.score_batch(X)
            gpu_status = self.gpu_scorer.status()
            gpu_enabled = gpu_status.get("gpu_enabled", False)
        else:
            gpu_probs = classical_probs
            gpu_status = {"enabled": False, "reason": "GPU scorer not available."}
            gpu_enabled = False

        results = []

        for index, event in enumerate(market_events):
            multi_agent = self.coordinator.aggregate(
                event=event,
                classical_probability=float(classical_probs[index]),
                deep_probability=float(deep_probs[index]),
                gpu_probability=float(gpu_probs[index]),
                deep_enabled=deep_enabled,
                gpu_enabled=gpu_enabled,
            )

            model_probability = multi_agent["ensemble_probability"]

            risk = self.risk_engine.score(
                event=event,
                model_probability=model_probability,
                agent_disagreement=multi_agent["agent_disagreement"],
                rl_optimizer=self.rl_optimizer,
            )

            tldr = self.explainer.generate_tldr(
                event=event,
                model_probability=model_probability,
                risk_level=risk["risk_level"],
                factors=risk["main_factors"],
            )

            decision = self.explainer.decision_showcase(
                event=event,
                model_probability=model_probability,
                risk_level=risk["risk_level"],
                trust_score=risk["trust_score"],
            )

            xapi_research_tasks = self.build_xapi_research_tasks_for_event(event)
            xapi_gateway_signal = self.xapi_agent.analyze(xapi_research_tasks)

            xapi = {
                "platform": "xapi.to",
                "actor": {
                    "name": "Electric Crab",
                    "type": "Multi-Agent AI Prediction Market System",
                },
                "verb": {
                    "id": "multi_agent_prediction_audit",
                    "display": "audited prediction market and prepared xapi.to gateway tasks",
                },
                "object": {
                    "id": event.event_id,
                    "definition": {
                        "name": event.title,
                        "type": (
                            "real_polymarket_event"
                            if data_source == "polymarket_gamma_api"
                            else "simulated_prediction_market_event"
                        ),
                    },
                },
                "result": {
                    "market_probability": event.market_probability,
                    "model_probability": model_probability,
                    "deviation": risk["deviation"],
                    "risk_points": risk["risk_points"],
                    "risk_level": risk["risk_level"],
                    "trust_score": risk["trust_score"],
                    "main_factors": risk["main_factors"],
                    "decision_showcase": decision,
                    "multi_agent": multi_agent,
                    "xapi_gateway_agent": xapi_gateway_signal,
                    "xapi_gateway_tasks": xapi_research_tasks,
                    "system_features": {
                        "data_source": data_source,
                        "deep_learning_enabled": deep_enabled,
                        "reinforcement_learning_enabled": self.rl_optimizer is not None,
                        "gpu_batch_scoring": gpu_status,
                        "deep_learning_status": deep_status,
                    },
                },
            }

            results.append(
                AuditResult(
                    event_id=event.event_id,
                    title=event.title,
                    market_probability=event.market_probability,
                    model_probability=model_probability,
                    deviation=risk["deviation"],
                    risk_level=risk["risk_level"],
                    trust_score=risk["trust_score"],
                    main_factors=risk["main_factors"],
                    tldr=tldr,
                    xapi=xapi,
                )
            )

        return results

    def build_xapi_research_tasks_for_event(self, event: MarketEvent) -> List[Dict[str, Any]]:
        if self.xapi_gateway is None:
            return []

        tasks = [
            self.xapi_gateway.build_market_research_task(
                market_title=event.title,
            )
        ]

        title_upper = event.title.upper()

        if "BTC" in title_upper or "BITCOIN" in title_upper:
            tasks.append(
                self.xapi_gateway.build_crypto_price_task(
                    token_symbol="BTC",
                )
            )

        if "ETH" in title_upper or "ETHEREUM" in title_upper:
            tasks.append(
                self.xapi_gateway.build_crypto_price_task(
                    token_symbol="ETH",
                )
            )

        return tasks

    def build_xapi_research_tasks(self, results: List[AuditResult]) -> List[Dict[str, Any]]:
        tasks = []

        for result in results:
            event = MarketEvent(
                event_id=result.event_id,
                title=result.title,
                market_probability=result.market_probability,
                volume=0,
                whale_ratio=0,
                liquidity_score=0,
                volatility=0,
                sentiment_score=0,
            )

            tasks.extend(self.build_xapi_research_tasks_for_event(event))

        return tasks

    async def call_xapi_mention_all(
        self,
        results: List[AuditResult],
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        if self.xapi_gateway is None:
            return {
                "executed": False,
                "error": "xapi.to gateway client is not available.",
            }

        audit_results = [
            {
                "event_id": result.event_id,
                "title": result.title,
                "market_probability": result.market_probability,
                "model_probability": result.model_probability,
                "deviation": result.deviation,
                "risk_level": result.risk_level,
                "trust_score": result.trust_score,
                "main_factors": result.main_factors,
            }
            for result in results
        ]

        task = self.xapi_gateway.build_mention_all_task(
            project_name="Electric Crab – The Market Guardian",
            audit_results=audit_results,
        )

        return await self.xapi_gateway.run_task(
            task=task,
            dry_run=dry_run,
        )


# =========================================================
# Printing Helpers
# =========================================================

def print_results(results: List[AuditResult]):
    for result in results:
        print("\n==============================")
        print(result.title)
        print("==============================")
        print("Market Probability:", result.market_probability)
        print("Multi-Agent Model Probability:", result.model_probability)
        print("Deviation:", result.deviation)
        print("Risk:", result.risk_level)
        print("Trust Score:", result.trust_score)
        print("Factors:", result.main_factors)
        print("TL;DR:", result.tldr)

        xapi_result = result.xapi["result"]
        decision = xapi_result["decision_showcase"]
        multi_agent = xapi_result["multi_agent"]
        xapi_gateway_agent = xapi_result["xapi_gateway_agent"]

        print("Decision Signal:", decision["decision_signal"])
        print("Action Label:", decision["action_label"])
        print("Consensus:", multi_agent["consensus_level"])
        print("Agent Disagreement:", multi_agent["agent_disagreement"])
        print("Warning Signals:", multi_agent["warning_signals"])
        print("xapi.to Gateway Signal:", xapi_gateway_agent["signal"])
        print("xapi.to Task Count:", xapi_gateway_agent["task_count"])

        print("\nAgent Votes:")
        for vote in multi_agent["agent_votes"]:
            print(
                f"- {vote['agent']}: "
                f"prob={vote['probability']}, "
                f"confidence={vote['confidence']}, "
                f"signal={vote['signal']}"
            )


def print_xapi_tasks(tasks: List[Dict[str, Any]]):
    print("\n==============================")
    print("xapi.to Gateway Research Tasks")
    print("==============================")

    if not tasks:
        print("No xapi.to tasks generated.")
        return

    for index, task in enumerate(tasks, start=1):
        print(f"\nTask {index}:")
        print("Capability Intent:", task.get("capability_intent"))
        print("Prompt:")
        print(task.get("xapi_prompt"))
        print("---")


def print_xapi_notify_result(result: Dict[str, Any]):
    print("\n==============================")
    print("xapi.to @Mention All Task")
    print("==============================")
    print("Executed:", result.get("executed"))
    print("Dry Run:", result.get("dry_run"))

    if result.get("reason"):
        print("Reason:", result.get("reason"))

    if result.get("error"):
        print("Error:", result.get("error"))

    if result.get("stdout"):
        print("\nSTDOUT:")
        print(result.get("stdout"))

    if result.get("stderr"):
        print("\nSTDERR:")
        print(result.get("stderr"))

    print("\nTask Preview:")
    print(json.dumps(result.get("task"), indent=2, ensure_ascii=False))


# =========================================================
# Demo Runners
# =========================================================

async def demo(notify: bool = False, send: bool = False):
    agent = ElectricCrabAgent(
        use_real_data=False,
        use_deep_learning=True,
        use_reinforcement_learning=True,
        use_gpu=True,
    )

    events = [
        {
            "event_id": "demo-001",
            "title": "Will Candidate X win the election?",
        },
        {
            "event_id": "demo-002",
            "title": "Will BTC close above $100K this month?",
        },
        {
            "event_id": "demo-003",
            "title": "Will Team A win the final?",
        },
    ]

    results = await agent.audit_demo_events(events)
    print_results(results)

    xapi_tasks = agent.build_xapi_research_tasks(results)
    print_xapi_tasks(xapi_tasks)

    if notify:
        xapi_notify_result = await agent.call_xapi_mention_all(
            results=results,
            dry_run=not send,
        )
        print_xapi_notify_result(xapi_notify_result)


async def real_demo(notify: bool = False, send: bool = False):
    agent = ElectricCrabAgent(
        use_real_data=True,
        use_deep_learning=True,
        use_reinforcement_learning=True,
        use_gpu=True,
    )

    results = await agent.audit_real_polymarket_events(limit=5)
    print_results(results)

    xapi_tasks = agent.build_xapi_research_tasks(results)
    print_xapi_tasks(xapi_tasks)

    if notify:
        xapi_notify_result = await agent.call_xapi_mention_all(
            results=results,
            dry_run=not send,
        )
        print_xapi_notify_result(xapi_notify_result)


# =========================================================
# CLI Entry
# =========================================================

if __name__ == "__main__":
    import sys

    args = [arg.lower() for arg in sys.argv[1:]]

    use_real = "real" in args
    notify = "notify" in args
    send = "send" in args

    if send and not notify:
        notify = True

    print("\nElectric Crab – The Market Guardian")
    print("Mode:", "REAL POLYMARKET" if use_real else "SIMULATED DEMO")
    print("xapi.to Notify:", notify)
    print("xapi.to Send Mode:", send)
    print("xapi.to CLI Enabled:", os.getenv("XAPI_ENABLE_CLI", "false"))
    print("xapi.to API Key Present:", bool(os.getenv("XAPI_API_KEY")))
    print("Extensions Available:", EXTENSIONS_AVAILABLE)

    if use_real:
        asyncio.run(real_demo(notify=notify, send=send))
    else:
        asyncio.run(demo(notify=notify, send=send))