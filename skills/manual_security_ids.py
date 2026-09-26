"""Explicit security-ID scope for manual receipts and confirmation simulation.

This is not the stock-ingest validator. Keep its default four ASCII digits;
00631L is a named exception for the reviewed domestic leveraged ETF study.
Syntax acceptance does not establish instrument availability or eligibility.
"""
import re

MANUAL_ETF_EXCEPTIONS=frozenset({'00631L'})

def valid_manual_security_id(value):
    return isinstance(value,str) and (bool(re.fullmatch(r'[0-9]{4}',value)) or value in MANUAL_ETF_EXCEPTIONS)
