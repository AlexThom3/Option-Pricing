import numpy as np
from scipy.stats import norm

"""Black-Scholes pricing functions for European call and put options."""

from math import exp, log, sqrt

from scipy.stats import norm


def calculate_d1_d2(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> tuple[float, float]:
    """
    Calculate the Black-Scholes d1 and d2 terms.

    Parameters
    ----------
    spot:
        Current price of the underlying asset.
    strike:
        Exercise price of the option.
    maturity:
        Time to maturity in years.
    rate:
        Continuously compounded risk-free interest rate.
    volatility:
        Annualised volatility of the underlying asset.

    Returns
    -------
    tuple[float, float]
        The values of d1 and d2.
    """
    if spot <= 0:
        raise ValueError("Spot price must be greater than zero.")

    if strike <= 0:
        raise ValueError("Strike price must be greater than zero.")

    if maturity <= 0:
        raise ValueError("Maturity must be greater than zero.")

    if volatility <= 0:
        raise ValueError("Volatility must be greater than zero.")

    d1 = (
        log(spot / strike)
        + (rate + 0.5 * volatility**2) * maturity
    ) / (volatility * sqrt(maturity))

    d2 = d1 - volatility * sqrt(maturity)

    return d1, d2


def european_call_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> float:
    """
    Calculate the Black-Scholes price of a European call option.
    """
    d1, d2 = calculate_d1_d2(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
    )

    call_price = (
        spot * norm.cdf(d1)
        - strike * exp(-rate * maturity) * norm.cdf(d2)
    )

    return call_price


def european_put_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> float:
    """
    Calculate the Black-Scholes price of a European put option.
    """
    d1, d2 = calculate_d1_d2(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
    )

    put_price = (
        strike * exp(-rate * maturity) * norm.cdf(-d2)
        - spot * norm.cdf(-d1)
    )

    return put_price


if __name__ == "__main__":
    call = european_call_price(
        spot=100,
        strike=100,
        maturity=1,
        rate=0.05,
        volatility=0.20,
    )

    put = european_put_price(
        spot=100,
        strike=100,
        maturity=1,
        rate=0.05,
        volatility=0.20,
    )

    print(f"European call price: {call:.4f}")
    print(f"European put price:  {put:.4f}")