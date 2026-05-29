def test_pipeline_inicial():
    """Prueba básica para validar que Pytest y GitHub Actions están conectados"""
    valor_esperado = "MLOps-Challenge"
    valor_actual = "MLOps-Challenge"
    assert valor_actual == valor_esperado

def test_cobertura_dummy():
    # Una operación matemática simple para asegurar ejecución de líneas
    resultado = 2 + 2
    assert resultado == 4