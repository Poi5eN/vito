#!/usr/bin/env python3
"""Deterministic startup/financial metrics used by VITO.

The model can request these calculations, but the arithmetic itself is kept
outside the language model for reproducibility.
"""

from __future__ import annotations

from math import isfinite


def _finite(x: float) -> float:
    x = float(x)
    if not isfinite(x):
        raise ValueError("metric inputs must be finite")
    return x


def growth_rate(old: float, new: float) -> float:
    old, new = _finite(old), _finite(new)
    if old == 0:
        raise ValueError("growth_rate is undefined when the old value is zero")
    return (new - old) / abs(old)


def gross_margin(revenue: float, cogs: float) -> float:
    revenue, cogs = _finite(revenue), _finite(cogs)
    if revenue <= 0:
        raise ValueError("revenue must be greater than zero")
    return (revenue - cogs) / revenue


def runway_months(cash: float, monthly_net_burn: float) -> float:
    cash, monthly_net_burn = _finite(cash), _finite(monthly_net_burn)
    if monthly_net_burn <= 0:
        return float("inf")
    return cash / monthly_net_burn


def ltv_to_cac(ltv: float, cac: float) -> float:
    ltv, cac = _finite(ltv), _finite(cac)
    if cac <= 0:
        raise ValueError("CAC must be greater than zero")
    return ltv / cac


def nrr(beginning_revenue: float, expansion: float, contraction: float, churn: float) -> float:
    beginning_revenue, expansion, contraction, churn = map(
        _finite, (beginning_revenue, expansion, contraction, churn)
    )
    if beginning_revenue <= 0:
        raise ValueError("beginning revenue must be greater than zero")
    return (beginning_revenue + expansion - contraction - churn) / beginning_revenue
