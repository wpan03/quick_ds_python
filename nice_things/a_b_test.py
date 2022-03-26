import statsmodels.stats.api as sms
from math import ceil


def get_sample_size_for_prop_test(
    effect_size: float,
    base_rate: float = 0.01,
    alpha: float = 0.01,
    power: float = 0.8,
    ratio: float = 1.0,
) -> int:
    """Calculate the Required Sample Size for Proportion Test"""
    effect_size = sms.proportion_effectsize(base_rate, base_rate + effect_size)

    required_n = sms.NormalIndPower().solve_power(
        effect_size, power=power, alpha=alpha, ratio=ratio
    )
    return ceil(required_n)
