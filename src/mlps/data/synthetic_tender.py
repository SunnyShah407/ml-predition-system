from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticTenderConfig:
    n_tenders: int = 8000
    n_buyers: int = 120
    n_vendors: int = 450
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"
    seed: int = 42


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _date_range_uniform(
    rng: np.random.Generator, start_date: str, end_date: str, n: int
) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end <= start:
        raise ValueError("end_date must be after start_date")
    start_ns = start.value
    end_ns = end.value
    samples = rng.integers(low=start_ns, high=end_ns, size=n, dtype=np.int64)
    return pd.to_datetime(samples)


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def generate_synthetic_tenders(cfg: SyntheticTenderConfig) -> pd.DataFrame:
    """
    Generate a tender-level dataset for regression.

    Target:
      - winning_total_price: simulated winning bid total (currency units)

    Notes:
      - Rows represent a single tender with the winning vendor's characteristics embedded.
      - This is meant for ML benchmarking and is not a substitute for real procurement data.
    """
    rng = _rng(cfg.seed)

    # --- Controlled vocab / dimensions ---
    categories = np.array(
        [
            "IT_SERVICES",
            "OFFICE_SUPPLIES",
            "MEDICAL_EQUIPMENT",
            "CONSTRUCTION",
            "MAINTENANCE",
            "LOGISTICS",
            "SECURITY_SERVICES",
            "CATERING",
        ],
        dtype=object,
    )
    regions = np.array(["NORTH", "SOUTH", "EAST", "WEST", "CENTRAL"], dtype=object)
    procurement_methods = np.array(
        ["OPEN", "RESTRICTED", "FRAMEWORK", "DIRECT"], dtype=object
    )
    contract_types = np.array(["GOODS", "SERVICES", "WORKS"], dtype=object)

    # --- Entities ---
    buyer_ids = np.array([f"B{idx:04d}" for idx in range(cfg.n_buyers)], dtype=object)
    vendor_ids = np.array([f"V{idx:05d}" for idx in range(cfg.n_vendors)], dtype=object)

    # Buyer "strictness" affects evaluation and discounts (higher -> harder to win, better quality)
    buyer_strictness = rng.beta(2.0, 2.5, size=cfg.n_buyers)  # 0..1
    buyer_budget_scale = rng.lognormal(mean=10.0, sigma=0.6, size=cfg.n_buyers)

    # Vendor profiles
    vendor_quality = rng.beta(2.2, 1.8, size=cfg.n_vendors)  # 0..1
    vendor_reliability = rng.beta(2.0, 2.0, size=cfg.n_vendors)  # 0..1
    vendor_capacity = rng.lognormal(mean=4.0, sigma=0.5, size=cfg.n_vendors)  # scale
    vendor_margin = np.clip(rng.normal(loc=0.12, scale=0.05, size=cfg.n_vendors), 0.03, 0.30)

    # Category base costs (unit price) and complexity effects
    category_base_unit = {
        "IT_SERVICES": 220.0,
        "OFFICE_SUPPLIES": 12.0,
        "MEDICAL_EQUIPMENT": 850.0,
        "CONSTRUCTION": 140.0,
        "MAINTENANCE": 95.0,
        "LOGISTICS": 70.0,
        "SECURITY_SERVICES": 55.0,
        "CATERING": 18.0,
    }
    category_complexity = {
        "IT_SERVICES": 0.75,
        "OFFICE_SUPPLIES": 0.10,
        "MEDICAL_EQUIPMENT": 0.85,
        "CONSTRUCTION": 0.70,
        "MAINTENANCE": 0.55,
        "LOGISTICS": 0.45,
        "SECURITY_SERVICES": 0.40,
        "CATERING": 0.25,
    }

    # --- Tender generation ---
    tender_ids = np.array([f"T{idx:06d}" for idx in range(cfg.n_tenders)], dtype=object)
    created_at = _date_range_uniform(rng, cfg.start_date, cfg.end_date, cfg.n_tenders)
    category = rng.choice(categories, size=cfg.n_tenders, replace=True)
    region = rng.choice(regions, size=cfg.n_tenders, replace=True)
    procurement_method = rng.choice(procurement_methods, size=cfg.n_tenders, replace=True, p=[0.62, 0.16, 0.15, 0.07])
    contract_type = rng.choice(contract_types, size=cfg.n_tenders, replace=True, p=[0.50, 0.35, 0.15])

    buyer_idx = rng.integers(0, cfg.n_buyers, size=cfg.n_tenders)
    buyer_id = buyer_ids[buyer_idx]

    # Tender size drivers
    # Quantity is heavy-tailed; works/IT tenders tend to be larger in value
    base_qty = rng.lognormal(mean=3.3, sigma=0.8, size=cfg.n_tenders)  # ~ 10..1000
    qty_multiplier = np.where(contract_type == "WORKS", 4.0, np.where(contract_type == "SERVICES", 1.8, 1.0))
    quantity = np.maximum(1.0, base_qty * qty_multiplier)

    # Complexity relates to category and method (open tends to be more standardized)
    cat_complex = np.vectorize(category_complexity.get)(category).astype(float)
    method_complex = np.where(procurement_method == "OPEN", 0.85, np.where(procurement_method == "RESTRICTED", 1.0, np.where(procurement_method == "FRAMEWORK", 0.9, 1.15)))
    tender_complexity = np.clip(cat_complex * method_complex + rng.normal(0, 0.05, cfg.n_tenders), 0.05, 1.0)

    # Delivery urgency affects price: shorter time -> higher cost
    delivery_days = (
        np.maximum(3.0, rng.normal(loc=35.0, scale=18.0, size=cfg.n_tenders))
        * np.where(contract_type == "WORKS", 1.8, 1.0)
    )
    delivery_days = np.clip(delivery_days, 3.0, 180.0)
    is_urgent = (delivery_days <= 14.0).astype(int)

    # Competition intensity: open tends to attract more bidders
    base_bidders = rng.poisson(lam=6.0, size=cfg.n_tenders) + 1
    bidders = base_bidders + np.where(procurement_method == "OPEN", rng.poisson(3.0, cfg.n_tenders), 0)
    bidders = np.clip(bidders, 2, 20)

    # Choose a "winner" vendor index with a bias toward quality/reliability and capacity fit.
    # For each tender, sample a candidate set and pick argmax of a score.
    winner_vendor_idx = np.empty(cfg.n_tenders, dtype=int)
    winner_score = np.empty(cfg.n_tenders, dtype=float)

    # Precompute vendor score components
    vendor_score_base = 0.55 * vendor_quality + 0.35 * vendor_reliability + 0.10 * (vendor_capacity / np.max(vendor_capacity))
    vendor_score_base = np.clip(vendor_score_base, 0.0, 1.0)

    for i in range(cfg.n_tenders):
        k = int(bidders[i])
        candidates = rng.choice(cfg.n_vendors, size=k, replace=False)

        # Capacity fit: very large tenders penalize low capacity vendors
        size_pressure = np.log1p(quantity[i]) * (0.6 + 0.8 * tender_complexity[i])
        cap_fit = 1.0 / (1.0 + np.exp(size_pressure - np.log1p(vendor_capacity[candidates])))

        b_strict = buyer_strictness[buyer_idx[i]]
        # Buyer strictness increases weight on quality/reliability
        score = (
            (0.60 + 0.20 * b_strict) * vendor_quality[candidates]
            + (0.30 + 0.10 * b_strict) * vendor_reliability[candidates]
            + 0.20 * cap_fit
        )
        score = score + rng.normal(0, 0.03, size=k)
        j = int(np.argmax(score))
        winner_vendor_idx[i] = int(candidates[j])
        winner_score[i] = float(score[j])

    winner_vendor_id = vendor_ids[winner_vendor_idx]
    vendor_q = vendor_quality[winner_vendor_idx]
    vendor_r = vendor_reliability[winner_vendor_idx]
    vendor_cap = vendor_capacity[winner_vendor_idx]
    vendor_m = vendor_margin[winner_vendor_idx]

    # Inflation / seasonality index from date
    created_month = created_at.month.values
    created_year = created_at.year.values
    year_offset = created_year - created_year.min()
    inflation_index = 1.0 + 0.035 * year_offset + 0.015 * np.sin(2 * np.pi * (created_month - 1) / 12.0)
    inflation_index = np.clip(inflation_index, 0.95, 1.25)

    # Base unit price by category + region factor + urgency + complexity
    base_unit = np.vectorize(category_base_unit.get)(category).astype(float)
    region_factor = np.where(region == "CENTRAL", 1.00, np.where(region == "NORTH", 1.04, np.where(region == "SOUTH", 1.02, np.where(region == "EAST", 1.03, 1.01))))
    urgency_factor = 1.0 + 0.10 * is_urgent
    complexity_factor = 1.0 + 0.30 * tender_complexity

    # Economies of scale: higher quantity reduces unit price
    scale_discount = 1.0 - 0.10 * np.tanh(np.log1p(quantity) / 4.0)  # ~ up to 10% discount
    scale_discount = np.clip(scale_discount, 0.85, 1.0)

    # Buyer budget influences how "premium" the final paid price is (not always lowest)
    buyer_budget = buyer_budget_scale[buyer_idx]
    buyer_budget_norm = (np.log1p(buyer_budget) - np.mean(np.log1p(buyer_budget))) / (np.std(np.log1p(buyer_budget)) + 1e-9)
    buyer_premium = 1.0 + 0.03 * np.tanh(buyer_budget_norm)

    # Competition reduces price; more bidders => stronger pressure
    competition_discount = 1.0 - 0.06 * np.tanh((bidders - 2) / 6.0)
    competition_discount = np.clip(competition_discount, 0.88, 1.0)

    # Winner vendor: higher quality/reliability tends to command slightly higher price, but
    # strong buyer strictness and competition counteract it.
    vendor_premium = 1.0 + 0.10 * (vendor_q - 0.5) + 0.05 * (vendor_r - 0.5)

    # Total price formation
    expected_unit_price = (
        base_unit
        * region_factor
        * urgency_factor
        * complexity_factor
        * scale_discount
        * inflation_index
        * buyer_premium
        * vendor_premium
        * (1.0 + vendor_m)
        * competition_discount
    )

    # Add heteroscedastic noise increasing with complexity and urgency
    noise_scale = 0.06 + 0.10 * tender_complexity + 0.05 * is_urgent
    unit_price = expected_unit_price * np.exp(rng.normal(0.0, noise_scale, size=cfg.n_tenders))

    winning_total_price = unit_price * quantity
    winning_total_price = np.maximum(winning_total_price, 1.0)

    df = pd.DataFrame(
        {
            "tender_id": tender_ids,
            "created_at": created_at,
            "buyer_id": buyer_id,
            "region": region,
            "category": category,
            "contract_type": contract_type,
            "procurement_method": procurement_method,
            "bidders": bidders.astype(int),
            "delivery_days": delivery_days.astype(int),
            "is_urgent": is_urgent.astype(int),
            "tender_complexity": tender_complexity.astype(float),
            "quantity": quantity.astype(float),
            "winner_vendor_id": winner_vendor_id,
            "winner_vendor_quality": vendor_q.astype(float),
            "winner_vendor_reliability": vendor_r.astype(float),
            "winner_vendor_capacity": vendor_cap.astype(float),
            "inflation_index": inflation_index.astype(float),
            "winning_unit_price": unit_price.astype(float),
            "winning_total_price": winning_total_price.astype(float),
        }
    )

    # Convenience time features for ML
    df["created_year"] = df["created_at"].dt.year.astype(int)
    df["created_month"] = df["created_at"].dt.month.astype(int)
    df["created_dayofweek"] = df["created_at"].dt.dayofweek.astype(int)

    return df

