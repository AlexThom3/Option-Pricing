"""Black-Scholes pricing functions for European call and put options."""

from math import exp, log, sqrt

from scipy.stats import norm

def d1_d2(S, K, T, r, sigma):
    """Calculate d1 and d2 for the Black-Scholes formula."""
    d1 = (log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2

def black_scholes_call(S, K, T, r, sigma):
    """Calculate the price of a European call option using the Black-Scholes formula."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    call_price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate the price of a European put option using the Black-Scholes formula."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    put_price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Example usage:
if __name__ == "__main__":
    S = 100  # Current stock price
    K = 100  # Strike price
    T = 1    # Time to maturity in years
    r = 0.05 # Risk-free interest rate
    sigma = 0.2 # Volatility

    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = black_scholes_put(S, K, T, r, sigma)

    print(f"Call Option Price: {call_price:.2f}")
    print(f"Put Option Price: {put_price:.2f}")

