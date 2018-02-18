from math import exp, log
from scipy.stats import norm
from datetime import datetime


def d_j(j, S, K, r, v, T):
    return (log(S/K) + (r + ((-1)**(j-1)) * 0.5 * v * v) * T)/(v * (T**0.5))


def option_price(CP, S, K, r, v, T):
    start = datetime.now()
    price = CP * S * norm.cdf(CP * d_j(1, S, K, r, v, T)) - CP * K * exp(-r * T) * norm.cdf(CP * d_j(2, S, K, r, v, T))
    end = datetime.now()
    time = (end - start).total_seconds()
    return [time, price]
