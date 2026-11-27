from icsm_as.evaluation.metrics import normalized_capacity_ratio,hq_referenced_ratio

def test_denominators_are_separate():
    assert normalized_capacity_ratio(9.0,10.0)==0.9
    assert hq_referenced_ratio(9.0,9.5)==9.0/9.5
