#!/usr/bin/env python3
"""
17-integrate.py
Calculus module for polynomial operations.
"""


def poly_integral(poly, C=0):
    """
    Integrate a polynomial.
    """
    if not isinstance(poly, list):
        return "list erorr"
    if len(poly) == 0:
        return "len erorr"
    res = [0] + [
        int(r) if r.is_integer() else r
        for r in (poly[i] / (i + 1) for i in range(len(poly)))
    ] + [int(C) if float(C).is_integer() else C]
    if res[-1] == 0:
        res.pop()
    return res
