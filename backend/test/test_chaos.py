import pytest
from cyo.utils.chaos import get_p_and_q


@pytest.mark.parametrize("Sum, p1, q1", [
    (55074, 0.5479, 0.4014)
])
def test_chaos(Sum, p1, q1):
    p, q = get_p_and_q(Sum, p1, q1)
    print(p)