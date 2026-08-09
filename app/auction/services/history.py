"""
SQLite prediction-history store for the Auction Bid Price module, ported from
Module 4's Flask app.py with the same schema, pointed at this project's own
outputs/auction/predictions.db (a fresh database, not Module 4's dev data).
"""
from __future__ import annotations

import sqlite3
from typing import Any, Optional

from app.auction.config import DB_PATH


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            species TEXT,
            region TEXT,
            season TEXT,
            auction_type TEXT,
            competition_level TEXT,
            diameter_cm REAL,
            length_m REAL,
            volume_m3 REAL,
            straightness_score REAL,
            taper_score REAL,
            visible_defects_score REAL,
            internal_defect_risk REAL,
            density_kg_m3 REAL,
            moisture_content REAL,
            quality_grade INTEGER,
            avg_market_price_species REAL,
            price_volatility REAL,
            market_demand_index REAL,
            supply_index REAL,
            export_demand_index REAL,
            num_expected_bidders INTEGER,
            predicted_value REAL,
            low_bound REAL,
            high_bound REAL,
            starting_bid REAL,
            expected_final_price REAL,
            sale_probability REAL
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(inputs: dict[str, Any], result: dict[str, Any]) -> Optional[int]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = """
        INSERT INTO predictions_history (
            species, region, season, auction_type, competition_level,
            diameter_cm, length_m, volume_m3, straightness_score,
            taper_score, visible_defects_score, internal_defect_risk,
            density_kg_m3, moisture_content, quality_grade,
            avg_market_price_species, price_volatility, market_demand_index,
            supply_index, export_demand_index, num_expected_bidders,
            predicted_value, low_bound, high_bound, starting_bid,
            expected_final_price, sale_probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.execute(query, (
        inputs['species'], inputs['region'], inputs['season'], inputs['auction_type'], inputs['competition_level'],
        float(inputs['diameter_cm']), float(inputs['length_m']), float(inputs['volume_m3']), float(inputs['straightness_score']),
        float(inputs['taper_score']), float(inputs['visible_defects_score']), float(inputs['internal_defect_risk']),
        float(inputs['density_kg_m3']), float(inputs['moisture_content']), int(inputs['quality_grade']),
        float(inputs['avg_market_price_species']), float(inputs['price_volatility']), float(inputs['market_demand_index']),
        float(inputs['supply_index']), float(inputs['export_demand_index']), int(inputs['num_expected_bidders']),
        result['predicted_value'], result['low_bound'], result['high_bound'], result['starting_bid'],
        result['expected_final_price'], result['sale_probability'],
    ))
    inserted_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inserted_id


def get_history(limit: int = 5) -> list[dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, species, diameter_cm, length_m, predicted_value
        FROM predictions_history
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            'id': r[0], 'timestamp': r[1], 'species': r[2],
            'diameter_cm': r[3], 'length_m': r[4], 'predicted_value': r[5],
        }
        for r in rows
    ]


def get_prediction_by_id(prediction_id: int) -> Optional[sqlite3.Row]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions_history WHERE id = ?", (prediction_id,))
    row = cursor.fetchone()
    conn.close()
    return row


init_db()
