"""
RAILCAST AI - Network Intelligence API Router
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db, get_digital_twin
from app.models_db import OperationalEventDB

router = APIRouter(prefix="/network", tags=["Network"])

@router.get("/state")
def get_network_state(db: Session = Depends(get_db)):
    events = db.query(OperationalEventDB).order_by(OperationalEventDB.id.desc()).limit(120).all()
    events_data = [{
        "arr_delay_min": e.arr_delay_min,
        "act_arr": e.act_arr,
        "section_id": e.section_id,
        "priority": e.priority
    } for e in events]
    
    twin = get_digital_twin()
    if twin:
        return twin.get_network_state(events_data)
    
    return {
        "active_trains": 85,
        "delayed_trains": 24,
        "critical_sections": 5,
        "average_delay_min": 14.2,
        "network_pressure_index": 62.5,
        "pressure_status": "HIGH"
    }

@router.get("/bottlenecks")
def get_network_bottlenecks(db: Session = Depends(get_db)):
    events = db.query(OperationalEventDB).order_by(OperationalEventDB.id.desc()).limit(150).all()
    events_data = [{
        "arr_delay_min": e.arr_delay_min,
        "section_id": e.section_id,
        "priority": e.priority
    } for e in events]
    
    twin = get_digital_twin()
    if twin:
        return twin.get_bottleneck_sections(events_data, top_k=6)
    
    return []

@router.get("/graph")
def get_network_graph():
    twin = get_digital_twin()
    if twin:
        return twin.get_graph_export()
    return {"nodes": [], "edges": []}
