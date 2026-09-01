from optimization.applicability import ApplicabilityDomainCalibrator


def test_ad_detects_in_and_out_domain():
    columns = ["formula::A", "process::T"]
    X = [
        [20 + i, 220 + i]
        for i in range(20)
    ]
    ad = ApplicabilityDomainCalibrator(
        feature_columns=columns,
        X=X,
        dropped_rows=0,
    )

    inside = ad.evaluate(
        {
            "formula::A": 29.5,
            "process::T": 229.5,
        }
    )
    outside = ad.evaluate(
        {
            "formula::A": 100,
            "process::T": 229.5,
        }
    )

    assert inside["status"] in {"IN_DOMAIN", "BORDERLINE"}
    assert outside["status"] == "OUT_OF_DOMAIN"
    assert "formula::A" in outside["outside_features"]
