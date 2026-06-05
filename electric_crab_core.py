"""
Electric Crab – The Market Guardian (Enhanced Core)

Core AI Agent module with extensions:
- Simulated or real prediction market data collection
- Feature engineering
- Probability estimation
- Risk scoring
- Explainable audit report
- xAPI-style structured output
- RL-style adaptive risk weighting (optional)
- GPU batch scoring (optional)
"""

import random
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# =========================
# Import extensions (optional)
# =========================
try:
    from electric_crab_extensions import (
        RealPolymarketCollector,
        AdaptiveRiskWeightAgent,
        GPUBatchScorer
    )
    EXTENSIONS_AVAILABLE = True
except ImportError:
    EXTENSIONS_AVAILABLE = False

# =========================
# Data Schema
# =========================
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

# =========================
# Data Collector
# =========================
class DataCollector:
    """
    Demo collector.
    In production, replace with real data sources.
    """
    async def fetch_event(self, event_id: str, title: str) -> MarketEvent:
        await asyncio.sleep(0.05)
        return MarketEvent(
            event_id=event_id,
            title=title,
            market_probability=round(random.uniform(0.25, 0.85), 4),
            volume=round(random.uniform(10_000, 1_000_000), 2),
            whale_ratio=round(random.uniform(0.05, 0.85), 4),
            liquidity_score=round(random.uniform(0.1, 1.0), 4),
            volatility=round(random.uniform(0.02, 0.45), 4),
            sentiment_score=round(random.uniform(-1.0, 1.0), 4),
        )

    async def fetch_batch(self, events: List[Dict[str, str]]) -> List[MarketEvent]:
        tasks = [self.fetch_event(e["event_id"], e["title"]) for e in events]
        return await asyncio.gather(*tasks)

# =========================
# Feature Engineering
# =========================
class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            "market_probability",
            "volume",
            "whale_ratio",
            "liquidity_score",
            "volatility",
            "sentiment_score",
        ]

    def to_dataframe(self, events: List[MarketEvent]) -> pd.DataFrame:
        return pd.DataFrame([asdict(event) for event in events])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        features = df[self.feature_columns].copy()
        return self.scaler.fit_transform(features)

    def single_features(self, event: MarketEvent) -> Dict[str, float]:
        return {k: getattr(event, k) for k in self.feature_columns}

# =========================
# Probability Model
# =========================
class ProbabilityModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=80, max_depth=5, random_state=42)
        self.is_trained = False

    def train_demo_model(self, X: np.ndarray):
        synthetic_y = []
        for row in X:
            base = 0.5
            noise = np.random.normal(0, 0.05)
            prob = base + 0.15 * row[0] - 0.12 * row[2] + 0.10 * row[3] - 0.08 * row[4] + 0.06 * row[5] + noise
            synthetic_y.append(float(np.clip(prob, 0.01, 0.99)))
        self.model.fit(X, synthetic_y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            self.train_demo_model(X)
        return np.clip(self.model.predict(X), 0.01, 0.99)

# =========================
# Risk Scoring
# =========================
class RiskEngine:
    def score(self, event: MarketEvent, model_probability: float) -> Dict[str, Any]:
        market_probability = event.market_probability
        deviation = abs(market_probability - model_probability)
        risk_points = 0
        factors = []

        if deviation >= 0.25:
            risk_points += 35
            factors.append("Large probability deviation")
        elif deviation >= 0.12:
            risk_points += 20
            factors.append("Moderate probability deviation")
        if event.whale_ratio >= 0.55:
            risk_points += 25
            factors.append("High whale concentration")
        if event.liquidity_score <= 0.35:
            risk_points += 20
            factors.append("Low liquidity")
        if event.volatility >= 0.30:
            risk_points += 15
            factors.append("High market volatility")
        if abs(event.sentiment_score) >= 0.70:
            risk_points += 10
            factors.append("Extreme sentiment signal")

        risk_points = min(risk_points, 100)
        trust_score = max(0, 100 - risk_points)

        if risk_points >= 65:
            risk_level = "HIGH"
        elif risk_points >= 35:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        if not factors:
            factors.append("Market probability is broadly aligned with model estimate")

        return {"deviation": round(deviation, 4),
                "risk_level": risk_level,
                "trust_score": round(trust_score, 2),
                "main_factors": factors}

# =========================
# Explainability
# =========================
class Explainer:
    def generate_tldr(self, event: MarketEvent, model_probability: float, risk_level: str, factors: List[str]) -> str:
        market_pct = round(event.market_probability * 100, 2)
        model_pct = round(model_probability * 100, 2)
        factor_text = ", ".join(factors[:3])
        return f"Electric Crab flags this market as {risk_level}. The market implies {market_pct}% while the model estimates {model_pct}%. Main drivers: {factor_text}."

# =========================
# Electric Crab Agent (Enhanced)
# =========================
class ElectricCrabAgent:
    def __init__(self, use_real_data=False, use_gpu=False, use_adaptive_rl=False):
        self.collector = DataCollector()
        self.feature_engineer = FeatureEngineer()
        self.model = ProbabilityModel()
        self.risk_engine = RiskEngine()
        self.explainer = Explainer()

        # Extensions
        self.use_real_data = use_real_data
        self.use_gpu = use_gpu
        self.use_adaptive_rl = use_adaptive_rl

        self.real_collector = RealPolymarketCollector() if EXTENSIONS_AVAILABLE and use_real_data else None
        self.gpu_scorer = GPUBatchScorer() if EXTENSIONS_AVAILABLE and use_gpu else None
        self.adaptive_agent = AdaptiveRiskWeightAgent() if EXTENSIONS_AVAILABLE and use_adaptive_rl else None

    # ---- audit simulated events ----
    async def audit_events(self, events: List[Dict[str, str]]) -> List[AuditResult]:
        market_events = await self.collector.fetch_batch(events)
        return await self._audit_events_core(market_events)

    # ---- audit real Polymarket events ----
    async def audit_real_polymarket_events(self, limit: int = 5) -> List[AuditResult]:
        if self.real_collector is None:
            raise RuntimeError("Real data module not available or use_real_data=False")
        market_events = await self.real_collector.fetch_active_events(limit=limit)
        return await self._audit_events_core(market_events, real=True)

    async def _audit_events_core(self, market_events: List[MarketEvent], real=False) -> List[AuditResult]:
        df = self.feature_engineer.to_dataframe(market_events)
        X = self.feature_engineer.transform(df)

        # GPU scoring
        if self.gpu_scorer is not None:
            model_probs = self.gpu_scorer.score_batch(X)
        else:
            model_probs = self.model.predict(X)

        results = []
        for event, model_prob in zip(market_events, model_probs):
            model_prob = float(round(model_prob, 4))
            risk = self.risk_engine.score(event, model_prob)

            # RL-style adaptive weighting
            if self.adaptive_agent is not None:
                adaptive_points = self.adaptive_agent.calculate_adaptive_risk_points(
                    deviation=risk["deviation"],
                    whale_ratio=event.whale_ratio,
                    liquidity_score=event.liquidity_score,
                    volatility=event.volatility,
                    sentiment_score=event.sentiment_score
                )
                if adaptive_points >= 65:
                    risk["risk_level"] = "HIGH"
                elif adaptive_points >= 35:
                    risk["risk_level"] = "MEDIUM"
                else:
                    risk["risk_level"] = "LOW"
                risk["trust_score"] = round(100 - adaptive_points, 2)

            tldr = self.explainer.generate_tldr(
                event=event,
                model_probability=model_prob,
                risk_level=risk["risk_level"],
                factors=risk["main_factors"]
            )

            xapi_result = {
                "actor": {"name": "Electric Crab", "type": "AI Agent"},
                "verb": {"id": "audit", "display": "audited prediction market"},
                "object": {"id": event.event_id, "definition": {"name": event.title, "type": "real_polymarket_event" if real else "prediction_market_event"}},
                "result": {**risk, "market_probability": event.market_probability, "model_probability": model_prob}
            }

            results.append(AuditResult(
                event_id=event.event_id,
                title=event.title,
                market_probability=event.market_probability,
                model_probability=model_prob,
                deviation=risk["deviation"],
                risk_level=risk["risk_level"],
                trust_score=risk["trust_score"],
                main_factors=risk["main_factors"],
                tldr=tldr,
                xapi=xapi_result
            ))

        return results

    async def audit_one(self, event_id: str, title: str) -> AuditResult:
        results = await self.audit_events([{"event_id": event_id, "title": title}])
        return results[0]

# =========================
# Local Demo
# =========================
async def demo():
    agent = ElectricCrabAgent()
    events = [
        {"event_id": "pm-001", "title": "Will Candidate X win the election?"},
        {"event_id": "pm-002", "title": "Will BTC close above $100K this month?"},
        {"event_id": "pm-003", "title": "Will Team A win the final?"}
    ]
    results = await agent.audit_events(events)
    for result in results:
        print("\n==============================")
        print(result.title)
        print("==============================")
        print("Market Probability:", result.market_probability)
        print("Model Probability:", result.model_probability)
        print("Deviation:", result.deviation)
        print("Risk:", result.risk_level)
        print("Trust Score:", result.trust_score)
        print("Factors:", result.main_factors)
        print("TL;DR:", result.tldr)

async def real_demo():
    agent = ElectricCrabAgent(use_real_data=True, use_gpu=True, use_adaptive_rl=True)
    results = await agent.audit_real_polymarket_events(limit=5)
    for result in results:
        print("\n==============================")
        print(result.title)
        print("==============================")
        print("Market Probability:", result.market_probability)
        print("Model Probability:", result.model_probability)
        print("Deviation:", result.deviation)
        print("Risk:", result.risk_level)
        print("Trust Score:", result.trust_score)
        print("Factors:", result.main_factors)
        print("TL;DR:", result.tldr)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "real":
        asyncio.run(real_demo())
    else:
        asyncio.run(demo())