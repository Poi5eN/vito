from scripts.metrics import growth_rate, gross_margin, runway_months, ltv_to_cac, nrr


def test_metrics():
    assert growth_rate(100, 125) == 0.25
    assert gross_margin(100, 40) == 0.6
    assert runway_months(120, 10) == 12
    assert ltv_to_cac(500, 100) == 5
    assert nrr(100, 20, 5, 5) == 1.10
