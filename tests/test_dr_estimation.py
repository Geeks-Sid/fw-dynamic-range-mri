import pytest

from DR_estimation import get_a  # import must not need command line arguments


@pytest.mark.parametrize(
    "x, expected",
    [
        (0, 2.2),  # pure noise
        (1, 1.8),
        (2.99, 2.2 - 0.4 * 2.99),
        (3, 1),  # threshold is inclusive
        (10, 1),
    ],
)
def test_get_a_follows_the_paper(x, expected):
    assert get_a(1.0, x) == pytest.approx(expected)


def test_get_a_scales_by_sigma():
    assert get_a(2.0, 2.0) == pytest.approx(1.8)  # x = Mn / sigma = 1
