from src.experiments.demographic_baseline_fastlink import field_spec


def test_our_cut_points_become_fastlinks_agree_and_partial_thresholds():
    assert field_spec("first_name") == "first_name:jw:0.94:0.85"
    assert field_spec("address") == "address:lv:0.9:0.7"
    assert field_spec("birth_year") == "birth_year:number:0.5:1.5"   # same year / at most one year apart
    assert field_spec("zip_code") == "zip_code:exact:0:0"
