from climatevision.analytics.carbon import estimate_carbon_loss

def test_carbon_math_amazon_tropical_moist():
    """Valida se a matemática do carbono bate com os fatores do IPCC (Issue #14)"""
    # 100 pixels de 10x10m = 10.000m2 = 1 Hectare exato
    resultado = estimate_carbon_loss(
        deforested_pixels=100, 
        pixel_size_m=10.0, 
        forest_type="tropical_moist", 
        region="amazon"
    )
    
    assert resultado["hectares"] == 1.0
    assert abs(resultado["carbon_tonnes"] - 201.07) < 0.1