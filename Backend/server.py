import os
import certifi
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict
from bson import ObjectId

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as aioredis

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CLAIM_WINDOW_SECONDS = int(os.getenv("CLAIM_WINDOW_SECONDS", "45"))
SEARCH_RADIUS_METERS = int(os.getenv("SEARCH_RADIUS_METERS", "10000"))

# ==============================================================================
# DATABASE & REDIS CLIENTS
# ==============================================================================
mongo_client = AsyncIOMotorClient(MONGO_URI, tlsCAFile=certifi.where())
db = mongo_client["skynex_dispatch"]
redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)

# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================
class GeoPoint(BaseModel):
    type: str = "Point"
    coordinates: List[float] = Field(..., description="[longitude, latitude]")

class OrganizationCreate(BaseModel):
    name: str
    type: str = Field(..., description="'NGO', 'SHELTER', or 'COMMUNITY_KITCHEN'")
    phone: str
    location: GeoPoint
    daily_otp: str = "4892"
    manager_face_embedding: Optional[List[float]] = None

class DonationCreate(BaseModel):
    donor_id: str
    food_title: str
    category: str
    serving_count: int
    safe_until: datetime
    allergens: List[str] = Field(..., description="Mandatory allergen declaration")
    location: GeoPoint
    pickup_address: str

class ClaimRequest(BaseModel):
    ngo_id: str
    accept: bool

class VerificationRequest(BaseModel):
    donation_id: str
    courier_id: str
    latitude: float
    longitude: float
    verification_mode: str = Field("FACE_MATCH", description="'FACE_MATCH' or 'OTP'")
    otp_code: Optional[str] = None
    face_encoding: Optional[List[float]] = None

# ==============================================================================
# NOTIFICATION HELPER
# ==============================================================================
async def dispatch_notification(phone: str, donation_title: str, servings: int, safe_until: datetime):
    safe_str = safe_until.strftime("%H:%M UTC")
    msg = (
        f"🚨 URGENT SURPLUS FOOD DISPATCH\n"
        f"Food: {donation_title} ({servings} portions)\n"
        f"Safe Until: {safe_str}\n"
        f"Reply ACCEPT to lock or PASS within {CLAIM_WINDOW_SECONDS}s."
    )
    print(f"\n[DISPATCH NOTIFIER -> {phone}]\n{msg}\n")

# ==============================================================================
# CORE MATCHING ENGINE & AUTO-FALLBACK
# ==============================================================================
async def match_and_offer_donation(donation_id: str):
    donation = await db.donations.find_one({"_id": ObjectId(donation_id)})
    if not donation or donation.get("status") != "PENDING_CLAIM":
        return

    now = datetime.now(timezone.utc)
    safe_until_dt = donation["safe_until"]
    if safe_until_dt.tzinfo is None:
        safe_until_dt = safe_until_dt.replace(tzinfo=timezone.utc)

    if safe_until_dt <= now:
        await db.donations.update_one(
            {"_id": ObjectId(donation_id)},
            {"$set": {"status": "EXPIRED"}}
        )
        print(f"[ENGINE] Donation {donation_id} expired before being claimed.")
        return

    rejected_ids = [ObjectId(rid) for rid in donation.get("rejected_by", [])]
    donor_coords = donation["location"]["coordinates"]

    cursor = db.organizations.find({
        "type": {"$in": ["NGO", "SHELTER", "COMMUNITY_KITCHEN"]},
        "verified": True,
        "_id": {"$nin": rejected_ids},
        "location": {
            "$near": {
                "$geometry": {"type": "Point", "coordinates": donor_coords},
                "$maxDistance": SEARCH_RADIUS_METERS
            }
        }
    })

    nearby_recipients = await cursor.to_list(length=10)
    if not nearby_recipients:
        print(f"[ENGINE] No available recipients found nearby for donation {donation_id}.")
        return

    candidate = nearby_recipients[0]
    candidate_id = str(candidate["_id"])
    lock_key = f"claim_lock:{donation_id}"

    acquired = await redis_client.set(lock_key, candidate_id, ex=CLAIM_WINDOW_SECONDS, nx=True)
    if not acquired:
        return

    await db.donations.update_one(
        {"_id": ObjectId(donation_id)},
        {
            "$set": {
                "current_offered_to": candidate_id,
                "claim_offer_started": datetime.now(timezone.utc)
            }
        }
    )

    asyncio.create_task(
        dispatch_notification(
            phone=candidate.get("phone", "0000000000"),
            donation_title=donation["food_title"],
            servings=donation["serving_count"],
            safe_until=safe_until_dt
        )
    )

    asyncio.create_task(monitor_claim_expiry(donation_id, candidate_id))

async def monitor_claim_expiry(donation_id: str, candidate_id: str):
    await asyncio.sleep(CLAIM_WINDOW_SECONDS)

    lock_key = f"claim_lock:{donation_id}"
    current_holder = await redis_client.get(lock_key)

    if current_holder == candidate_id:
        print(f"[AUTO-FALLBACK] Recipient {candidate_id} timed out. Reassigning donation {donation_id}...")
        await redis_client.delete(lock_key)
        await db.donations.update_one(
            {"_id": ObjectId(donation_id)},
            {"$addToSet": {"rejected_by": candidate_id}}
        )
        await match_and_offer_donation(donation_id)

# ==============================================================================
# ON-DEVICE BIOMETRIC COMPARISON (DPDP COMPLIANT)
# ==============================================================================
def verify_face_embeddings(received: Optional[List[float]], stored: Optional[List[float]], threshold: float = 0.55) -> bool:
    if not received or not stored:
        return False
    v1 = np.array(received, dtype=float)
    v2 = np.array(stored, dtype=float)
    distance = np.linalg.norm(v1 - v2)
    return bool(distance < threshold)

# ==============================================================================
# WEBSOCKET REAL-TIME TELEMETRY
# ==============================================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, donation_id: str, websocket: WebSocket):
        await websocket.accept()
        if donation_id not in self.active_connections:
            self.active_connections[donation_id] = []
        self.active_connections[donation_id].append(websocket)

    def disconnect(self, donation_id: str, websocket: WebSocket):
        if donation_id in self.active_connections:
            self.active_connections[donation_id].remove(websocket)
            if not self.active_connections[donation_id]:
                del self.active_connections[donation_id]

    async def broadcast(self, donation_id: str, message: dict):
        if donation_id in self.active_connections:
            for connection in self.active_connections[donation_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

ws_manager = ConnectionManager()

# ==============================================================================
# FASTAPI APPLICATION SETUP
# ==============================================================================
app = FastAPI(
    title="Dispatch Engine API",
    description="Time-aware surplus food redistribution backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db():
    await db.organizations.create_index([("location", "2dsphere")])
    await db.donations.create_index([("location", "2dsphere")])
    await db.donations.create_index([("status", 1)])
    await db.audit_ledger.create_index([("donation_id", 1)])
    print("[SYSTEM] MongoDB 2dsphere indexes verified.")

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.post("/api/organizations", status_code=status.HTTP_201_CREATED)
async def register_organization(payload: OrganizationCreate):
    doc = payload.model_dump()
    doc["verified"] = True
    doc["created_at"] = datetime.now(timezone.utc)
    result = await db.organizations.insert_one(doc)
    return {"status": "SUCCESS", "organization_id": str(result.inserted_id)}

@app.post("/api/donations", status_code=status.HTTP_201_CREATED)
async def post_surplus_donation(payload: DonationCreate):
    now = datetime.now(timezone.utc)
    safe_until_dt = payload.safe_until
    if safe_until_dt.tzinfo is None:
        safe_until_dt = safe_until_dt.replace(tzinfo=timezone.utc)

    if safe_until_dt <= now:
        raise HTTPException(status_code=400, detail="safe_until timestamp must be in the future.")

    if not payload.allergens:
        raise HTTPException(status_code=400, detail="Mandatory allergen declaration cannot be empty.")

    doc = payload.model_dump()
    doc.update({
        "status": "PENDING_CLAIM",
        "rejected_by": [],
        "created_at": now,
        "current_offered_to": None,
        "claimed_by": None
    })

    result = await db.donations.insert_one(doc)
    donation_id = str(result.inserted_id)

    await match_and_offer_donation(donation_id)

    return {"status": "SUCCESS", "donation_id": donation_id}

@app.post("/api/donations/{donation_id}/claim")
async def process_claim(donation_id: str, claim: ClaimRequest):
    lock_key = f"claim_lock:{donation_id}"
    lock_holder = await redis_client.get(lock_key)

    if lock_holder != claim.ngo_id:
        raise HTTPException(status_code=403, detail="Claim window expired or not offered to this recipient.")

    if claim.accept:
        await redis_client.delete(lock_key)
        await db.donations.update_one(
            {"_id": ObjectId(donation_id)},
            {
                "$set": {
                    "status": "CLAIMED",
                    "claimed_by": claim.ngo_id,
                    "claimed_at": datetime.now(timezone.utc)
                }
            }
        )
        return {"status": "LOCKED", "message": "Donation secured for pickup."}
    else:
        await redis_client.delete(lock_key)
        await db.donations.update_one(
            {"_id": ObjectId(donation_id)},
            {"$addToSet": {"rejected_by": claim.ngo_id}}
        )
        await match_and_offer_donation(donation_id)
        return {"status": "PASSED", "message": "Offer declined. Cascaded to next match."}

@app.post("/api/deliveries/verify")
async def verify_delivery(payload: VerificationRequest):
    donation = await db.donations.find_one({"_id": ObjectId(payload.donation_id)})
    if not donation:
        raise HTTPException(status_code=404, detail="Donation record not found.")

    if donation.get("status") not in ["CLAIMED", "IN_TRANSIT"]:
        raise HTTPException(status_code=400, detail="Donation not in deliverable state.")

    recipient_id = donation.get("claimed_by")
    recipient = await db.organizations.find_one({"_id": ObjectId(recipient_id)})
    if not recipient:
        raise HTTPException(status_code=400, detail="Target recipient record not found.")

    is_verified = False

    if payload.verification_mode == "OTP":
        if payload.otp_code and payload.otp_code == recipient.get("daily_otp", "4892"):
            is_verified = True
    elif payload.verification_mode == "FACE_MATCH":
        is_verified = verify_face_embeddings(
            payload.face_encoding,
            recipient.get("manager_face_embedding")
        )

    if not is_verified:
        raise HTTPException(status_code=400, detail="Handover verification failed.")

    servings = donation.get("serving_count", 0)
    kg_diverted = round(servings * 0.42, 2)
    co2e_saved_tons = round(kg_diverted * 0.0025, 4)

    audit_entry = {
        "donation_id": payload.donation_id,
        "donor_id": donation.get("donor_id"),
        "recipient_id": recipient_id,
        "courier_id": payload.courier_id,
        "delivered_at": datetime.now(timezone.utc),
        "geotag": {"latitude": payload.latitude, "longitude": payload.longitude},
        "mode_used": payload.verification_mode,
        "meals_saved": servings,
        "waste_diverted_kg": kg_diverted,
        "co2e_avoided_tons": co2e_saved_tons
    }

    await db.audit_ledger.insert_one(audit_entry)
    await db.donations.update_one(
        {"_id": ObjectId(payload.donation_id)},
        {"$set": {"status": "DELIVERED"}}
    )

    audit_entry["_id"] = str(audit_entry["_id"])
    return {"status": "DELIVERED", "audit": audit_entry}

@app.get("/api/reports/impact")
async def get_cumulative_impact():
    records = await db.audit_ledger.find({}).to_list(length=1000)
    return {
        "total_deliveries": len(records),
        "total_meals_saved": sum(r.get("meals_saved", 0) for r in records),
        "total_waste_diverted_kg": round(sum(r.get("waste_diverted_kg", 0.0) for r in records), 2),
        "total_co2e_avoided_tons": round(sum(r.get("co2e_avoided_tons", 0.0) for r in records), 4)
    }

@app.websocket("/ws/transit/{donation_id}")
async def transit_websocket(websocket: WebSocket, donation_id: str):
    await ws_manager.connect(donation_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            payload = {
                "donation_id": donation_id,
                "latitude": data.get("lat"),
                "longitude": data.get("lng"),
                "speed_kmh": data.get("speed", 25.0),
                "eta_minutes": data.get("eta", 10),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await ws_manager.broadcast(donation_id, payload)
    except WebSocketDisconnect:
        ws_manager.disconnect(donation_id, websocket)
