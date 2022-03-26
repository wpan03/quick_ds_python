import statsmodels.stats.api as sms
from math import ceil

# Reference: https://towardsdatascience.com/ab-testing-with-python-e5964dd66143#2.-Collecting-and-preparing-the-data


def get_sample_size_for_prop_test(
    base_rate: float,
    target_rate: float,
    alpha: float = 0.01,
    the_power: float = 0.8,
    ratio: float = 1.0,
) -> int:
    """Calculate the Required Sample Size for Proportion Test"""
    effect_size = sms.proportion_effectsize(base_rate, target_rate)
    required_n = sms.NormalIndPower().solve_power(
        effect_size, power=the_power, alpha=alpha, ratio=ratio
    )
    return ceil(required_n)
