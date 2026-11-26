"""
RAILCAST AI - Database Initialization & Seeding Engine
"""

import os
import json
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings
from app.models_db import Base, StationDB, SectionDB, TrainDB, OperationalEventDB, AlertDB
from ml.simulator import RailwaySimulator
from ml.digital_twin import RailwayDigitalTwin
from ml.trainer import RailcastModelPipeline

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Global instances
SIMULATOR = None
DIGITAL_TWIN = None
MODEL_PIPELINE = None

def get_simulator():
    global SIMULATOR
    return SIMULATOR

def get_digital_twin():
    global DIGITAL_TWIN
    return DIGITAL_TWIN

def get_model_pipeline():
    global MODEL_PIPELINE
    return MODEL_PIPELINE

def init_and_seed_db():
    global SIMULATOR, DIGITAL_TWIN, MODEL_PIPELINE
    Base.metadata.create_all(bind=engine)

    # Initialize Simulator
    if SIMULATOR is None:
        print("[INFO] Initializing Railway Operations Simulator...")
        SIMULATOR = RailwaySimulator(seed=42)

    # Initialize Digital Twin Graph
    if DIGITAL_TWIN is None:
        print("[INFO] Building Railway Digital Twin Graph...")
        DIGITAL_TWIN = RailwayDigitalTwin(SIMULATOR.stations, SIMULATOR.sections)

    db = SessionLocal()
    try:
        existing_stns = db.query(StationDB).count()
        if existing_stns == 0:
            print(f"[INFO] Seeding {len(SIMULATOR.stations)} stations...")
            for stn in SIMULATOR.stations.values():
                db.add(StationDB(
                    code=stn["code"],
                    name=stn["name"],
                    lat=stn["lat"],
                    lon=stn["lon"],
                    corridor=stn["corridor"],
                    platforms=stn["platforms"],
                    congestion_index=stn["congestion_index"]
                ))

            print(f"[INFO] Seeding {len(SIMULATOR.sections)} railway sections...")
            for sec in SIMULATOR.sections:
                db.add(SectionDB(
                    section_id=sec["section_id"],
                    from_station=sec["from_station"],
                    to_station=sec["to_station"],
                    distance_km=sec["distance_km"],
                    max_speed_kmh=sec["max_speed_kmh"],
                    tracks=sec["tracks"],
                    base_capacity=sec["base_capacity"],
                    current_congestion=sec["current_congestion"],
                    weather=sec["weather"]
                ))

            print(f"[INFO] Seeding {len(SIMULATOR.train_routes)} train routes...")
            for tr in SIMULATOR.train_routes:
                db.add(TrainDB(
                    train_id=tr["train_id"],
                    train_name=tr["train_name"],
                    train_type=tr["train_type"],
                    priority=tr["priority"],
                    max_speed=tr["max_speed"],
                    base_dwell_min=tr["base_dwell_min"],
                    recovery_factor=tr["recovery_factor"],
                    route_json=json.dumps(tr["route"])
                ))

            db.commit()

            print("[INFO] Running Causal Operational Simulation (Generating ~10,000+ training events)...")
            df_events = SIMULATOR.simulate_journeys(num_days=7)

            print(f"[INFO] Saving {len(df_events)} operational events to SQLite Database...")
            db_events = []
            for _, row in df_events.iterrows():
                db_events.append(OperationalEventDB(
                    journey_id=row["journey_id"],
                    train_id=row["train_id"],
                    train_name=row["train_name"],
                    train_type=row["train_type"],
                    priority=row["priority"],
                    station_code=row["station_code"],
                    station_name=row["station_name"],
                    sequence_idx=int(row["sequence_idx"]),
                    is_origin=bool(row["is_origin"]),
                    is_destination=bool(row["is_destination"]),
                    sched_arr=str(row["sched_arr"]),
                    act_arr=str(row["act_arr"]),
                    arr_delay_min=float(row["arr_delay_min"]),
                    sched_dep=str(row["sched_dep"]),
                    act_dep=str(row["act_dep"]),
                    dep_delay_min=float(row["dep_delay_min"]),
                    dwell_sched_min=float(row["dwell_sched_min"]),
                    dwell_act_min=float(row["dwell_act_min"]),
                    dwell_impact_min=float(row["dwell_impact_min"]),
                    section_id=row["section_id"],
                    section_congestion=float(row["section_congestion"]),
                    weather=row["weather"],
                    preceding_conflict_min=float(row["preceding_conflict_min"]),
                    recovery_min=float(row["recovery_min"]),
                    distance_from_origin_km=float(row["distance_from_origin_km"]),
                    remaining_distance_km=float(row["remaining_distance_km"])
                ))

            db.bulk_save_objects(db_events)
            db.commit()

        # Check for alerts
        if db.query(AlertDB).count() == 0:
            print("[INFO] Generating Initial Operational Risk Alerts...")
            db.add(AlertDB(
                alert_type="HIGH_ETA_RISK",
                risk_level="HIGH",
                target_id="12005",
                target_name="NDLS-CNB Rajdhani / Shatabdi",
                title="High ETA Delay Amplification Expected",
                message="Train 12005 currently delayed by 18 min. High congestion on section SEC_CNB_PRYJ predicted to amplify destination delay to +32 min.",
                probability=0.88,
                confidence=0.91
            ))
            db.add(AlertDB(
                alert_type="NETWORK_CASCADE_RISK",
                risk_level="CRITICAL",
                target_id="SEC_CNB_PRYJ",
                target_name="Kanpur - Prayagraj Trunk Section",
                title="Network Delay Cascade Detected",
                message="Section SEC_CNB_PRYJ congestion index reached 0.82. Estimated 4 active trains experiencing signal aspect delay cascade (44 total train-minutes).",
                probability=0.94,
                confidence=0.95
            ))
            db.commit()

        # Check or train ML Model Pipeline
        if MODEL_PIPELINE is None:
            print(f"[INFO] Loading ML Model Pipeline ({settings.MODEL_VERSION})...")
            loaded = RailcastModelPipeline.load(settings.MODEL_VERSION)
            if loaded is not None and getattr(loaded, "is_trained", False):
                MODEL_PIPELINE = loaded
                print(f"[INFO] ML Model Pipeline ({settings.MODEL_VERSION}) successfully loaded from registry.")
            else:
                print("[INFO] Training GBDT & Quantile Uncertainty Model Pipeline...")
                events_count = db.query(OperationalEventDB).count()
                if events_count >= 1000:
                    print(f"[INFO] Querying existing operational events ({min(events_count, 15000)} records) for training...")
                    records = db.query(OperationalEventDB).limit(15000).all()
                    df_events = pd.DataFrame([{
                        "journey_id": e.journey_id,
                        "train_id": e.train_id,
                        "train_name": e.train_name,
                        "train_type": e.train_type,
                        "priority": e.priority,
                        "station_code": e.station_code,
                        "station_name": e.station_name,
                        "sequence_idx": e.sequence_idx,
                        "is_origin": e.is_origin,
                        "is_destination": e.is_destination,
                        "sched_arr": e.sched_arr,
                        "act_arr": e.act_arr,
                        "arr_delay_min": e.arr_delay_min,
                        "sched_dep": e.sched_dep,
                        "act_dep": e.act_dep,
                        "dep_delay_min": e.dep_delay_min,
                        "dwell_sched_min": e.dwell_sched_min,
                        "dwell_act_min": e.dwell_act_min,
                        "dwell_impact_min": e.dwell_impact_min,
                        "section_id": e.section_id,
                        "section_congestion": e.section_congestion,
                        "weather": e.weather,
                        "preceding_conflict_min": e.preceding_conflict_min,
                        "recovery_min": e.recovery_min,
                        "distance_from_origin_km": e.distance_from_origin_km,
                        "remaining_distance_km": e.remaining_distance_km
                    } for e in records])
                else:
                    print("[INFO] Simulating fresh operational events for training...")
                    df_events = SIMULATOR.simulate_journeys(num_days=7)

                MODEL_PIPELINE = RailcastModelPipeline(model_version=settings.MODEL_VERSION)
                metrics = MODEL_PIPELINE.fit(df_events)
                print(f"[INFO] ML Model Pipeline trained successfully! MAE: {metrics.get('MAE')} min, Coverage 80%: {metrics.get('Coverage80')}%, Latency: {metrics.get('InferenceLatencyMs')} ms")

    finally:
        db.close()
