def test_fallback_import():
    from src.totally_unknown import counterfactual_style_swap, profile_model
    assert callable(counterfactual_style_swap)
    assert callable(profile_model)
