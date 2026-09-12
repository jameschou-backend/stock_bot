#!/usr/bin/env python3
"""Build an explicit local CA bundle after validating TPEx's missing intermediate.

No global trust-store or .env changes. No disabled TLS or unverified root trust.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import certifi
import requests
try:
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
except ImportError as exc:
    raise ImportError('TLS audit requires cryptography; install with: python -m pip install cryptography') from exc
from scripts.research_exit_scenarios import write, sha

URL = 'https://sslserver.twca.com.tw/cacert/Cyber_SSL_2023.crt'


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    response = requests.get(URL, timeout=30, verify=certifi.where())
    response.raise_for_status()
    raw = response.content
    try:
        cert = x509.load_der_x509_certificate(raw)
    except ValueError:
        cert = x509.load_pem_x509_certificate(raw)
    intermediate = output / 'intermediate.pem'
    intermediate.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    # The peer certificate is untrusted until openssl verify below succeeds.
    peer = subprocess.run(['openssl', 's_client', '-connect', 'www.tpex.org.tw:443',
        '-servername', 'www.tpex.org.tw', '-showcerts'], input='', capture_output=True,
        text=True, timeout=30)
    certificates = re.findall(r'-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----', peer.stdout, re.S)
    if not certificates:
        raise ValueError('TPEx did not present a certificate')
    leaf = output / 'leaf.pem'
    leaf.write_text(certificates[0]+'\n')
    verify = subprocess.run(['openssl', 'verify', '-CAfile', certifi.where(),
        '-untrusted', str(intermediate), '-verify_hostname', 'www.tpex.org.tw', str(leaf)],
        capture_output=True, text=True, timeout=30)
    if verify.returncode:
        raise ValueError('Chain does not validate against existing certifi roots: '+verify.stderr)
    bundle = output / 'bundle.pem'
    bundle.write_bytes(Path(certifi.where()).read_bytes()+b'\n'+intermediate.read_bytes())
    evidence = dict(observed_at=datetime.now(timezone.utc).isoformat(), issuer_url=URL,
        issuer_subject=cert.subject.rfc4514_string(), issuer_parent=cert.issuer.rfc4514_string(),
        trusted_roots_sha256=sha(certifi.where()), intermediate_sha256=sha(intermediate),
        leaf_sha256=sha(leaf), bundle_sha256=sha(bundle),
        chain_and_hostname_verified=True, verification=verify.stdout.strip(),
        global_trust_modified=False, tls_verification_disabled=False,
        usage='Explicit REQUESTS_CA_BUNDLE for this process only', code_sha256=sha(__file__))
    write(output / 'verification.json', evidence)
    print(bundle.resolve())
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output)
