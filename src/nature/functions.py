import math
import pandas as pd


def generalized_logistic(
    x: float,
    a: float = 0.0,
    k: float = 1.0,
    b: float = 1.0,
    v: float = 1.0,
    q: float = 1.0,
    c: float = 1.0,
    m: float = 0.0,
):
    """Super-maleable sigmoidal growth function; super-function of the
    von Bertalanffy and Gompertz functions.
    See: https://en.wikipedia.org/wiki/Generalised_logistic_function

    Visually it appears to be -6 to 6 on the x axis. But WHY?

    Args:
        x (float): independent variable
        a (float, optional): the left asymptote, defaults to 0
        k (float, optional): the right asymptote when c=1, defaults to 1
        b (float, optional): growth coefficient, basically exapnds the whole curve along the x axis, defaults to 1
        v (float, optional): > 0, affects near which asymptote maximum growth occurs, defaults to 1
        q (float, optional): related to the output when x=0, kinda moves the growth part of the curve, defaults to 1
        c (float, optional): If not 1, causes upper asymptote to equal (a + (k-a)/(c**(1/v)))
        m (float, optional): starting time, basically the shift along the x axis, defaults to 0

    Returns:
        float: The output of the generalized logistic function for value x
    """
    # TODO Error catch: ensure v must be > 0

    return a + (k - a) / (c + q * math.exp(-b * (x - m))) ** (1 / v)


def parametric_generalized_logistic(
    x: float,
    a: float = 0,
    k: float = 1,
    b: float = 1,
    q: float = 1,
    c: float = 1,
    m: float = 0,
):
    """Super-maleable sigmoidal growth function; super-function of the
    von Bertalanffy and Gompertz functions.
    See: https://en.wikipedia.org/wiki/Generalised_logistic_function
    We have changed the exponent to lock the left asymptote at x=0 (by making what was "m" a constant of 2e)
    and changed "b" to e a direct measure of how far along the x axis (from 0) the right asymptote hits.
    "m" is now the x shift of the start of the sigmoid.

    Args:
        x (float): independent variable
        a (float, optional): the lower (left) asymptote, defaults to 0
        k (float, optional): the upper (right) asymptote when c=1, defaults to 1
        b (float, optional): the distance over which the full saturation (or decay) occurs. Bascially, an x-axis stretch.
        q (float, optional): related to the output when x=0, kinda moves the growth part of the curve, defaults to 1
        c (float, optional): If not 1, causes upper asymptote to equal (a + (k-a)/(c**(1/v)))
        m (float, optional): x shift of start of sigmoid, defaults to 0.

    Returns:
        float: The output of the parametric generalized logistic function for value x
    """
    return a + (k - a) / (c + q * math.exp(-2 * math.e * (2 * (x - m) / b - 1)))


def von_bertalanffy(x: float, l: float, k: float, t0: float):
    """Growth saturation, with fast initial growth that tapers off.

    Args:
        x (float): independent variable
        l (float): vertical asymptote
        k (float): growth coefficient
        t0 (float): x intercept when y=0

    Returns:
        float: The output of the von Bertalanffy function for value x
    """
    return l * (1 - math.exp(-k(x - t0)))


def gompertz(x: float, a: float, b: float, c: float):
    """Growth saturation, with slow initial and final growth but fast intermediate.

    Args:
        x (float): independent variable
        a (float): vertical asymptote
        b (float): horizontal placement (not sure on exact relationship)
        c (float): growth coefficient

    Returns:
        float: The output of the Gompertz function for value x
    """
    return a * math.exp(-b * math.exp(-c * x))


def grouped_weighted_avg(
    values: pd.Series, weights: pd.Series, by: pd.Series | bool = False
) -> float | pd.Series:
    """
    Function to calculate a weighted average, with optional functionality to perform a grouped weighted average

    Parameters:
        values (pd.Series): values to average
        weights (pd.Series): weights corresponding to values
        by (pd.Series or bool): optional categorical grouping series to bin values and weights. Defaults to False (will not run)

    Returns:
        float or pd.Series: the weighted average (single value or pd.Series grouped by category)
    """
    if by:
        return (values.fillna(0) * weights.fillna(0)).groupby(
            by
        ).sum() / weights.fillna(0).groupby(by).sum()
    else:
        return (values.fillna(0) * weights.fillna(0)).sum() / weights.fillna(0).sum()
