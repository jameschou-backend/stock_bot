#!/usr/bin/env python3
"""Capture at most one official disclosure response; never approves a book."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.official_review_packet import capture

if __name__ == '__main__':
    result = capture()
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result['status'] == 'ok' else 1)
