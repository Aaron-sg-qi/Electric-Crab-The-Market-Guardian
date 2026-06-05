"""
FastAPI app for Electric Crab – The Market Guardian

Run:
uvicorn app:app --reload
"""

from typing import List
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from electric_crab_core import ElectricCrabAgent


app = FastAPI(
    title="Electric Crab – The Market Guardian",
    description="AI Agent for prediction market risk auditing and anti-manipulation detection.",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = ElectricCrabAgent()


class EventInput(BaseModel):
    event_id: str
    title: str


class BatchAuditRequest(BaseModel):
    events: List[EventInput]


@app.get("/")
def home():
    return {
        "project": "Electric Crab – The Market Guardian",
        "description": "An AI Agent that audits prediction markets, detects manipulation, scores risk, and outputs explainable xAPI results.",
        "endpoints": {
            "single_audit": "POST /audit",
            "batch_audit": "POST /audit/batch",
            "demo": "GET /demo"
        }
    }


@app.post("/audit")
async def audit_event(event: EventInput):
    result = await agent.audit_one(
        event_id=event.event_id,
        title=event.title
    )

    return asdict(result)


@app.post("/audit/batch")
async def audit_batch(request: BatchAuditRequest):
    events = [
        {
            "event_id": event.event_id,
            "title": event.title
        }
        for event in request.events
    ]

    results = await agent.audit_events(events)

    return {
        "count": len(results),
        "results": [asdict(result) for result in results]
    }


@app.get("/demo")
async def demo():
    demo_events = [
        {
            "event_id": "pm-001",
            "title": "Will Candidate X win the election?"
        },
        {
            "event_id": "pm-002",
            "title": "Will BTC close above $100K this month?"
        },
        {
            "event_id": "pm-003",
            "title": "Will Team A win the final?"
        }
    ]

    results = await agent.audit_events(demo_events)

    return {
        "agent": "Electric Crab",
        "role": "The Market Guardian",
        "results": [asdict(result) for result in results]
    }


@app.get("/pitch")
def pitch():
    return {
        "name": "Electric Crab",
        "tagline": "The Market Guardian",
        "one_liner": "An electric-fast AI Agent that monitors prediction markets, detects manipulation, quantifies risk, and protects market trust.",
        "narrative": [
            "Prediction markets are becoming the internet's truth machines.",
            "But prices can be distorted by whales, low liquidity, and sentiment manipulation.",
            "Electric Crab audits market probabilities and assigns trust scores in real time."
        ],
        "tracks": [
            "AI Agent",
            "Security / Risk Agent",
            "AI for Transparency",
            "Best use of xAPI"
        ]
    }