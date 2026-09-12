from types import SimpleNamespace

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
import certifi

from scripts import prepare_tpex_tls_chain as tls


@pytest.mark.parametrize('valid', [True, False])
def test_bundle_requires_existing_roots_and_hostname_validation(tmp_path, monkeypatch, valid):
    # A public installed certificate supplies parseable bytes; subprocess is
    # mocked so this tests the trust boundary without contacting any server.
    pem = open(certifi.where(), 'rb').read()
    first = pem[pem.index(b'-----BEGIN CERTIFICATE-----'):]
    first = first[:first.index(b'-----END CERTIFICATE-----')+len(b'-----END CERTIFICATE-----')]
    cert = x509.load_pem_x509_certificate(first)
    def get(url, **kwargs):
        assert url == tls.URL and kwargs['verify'] == certifi.where()
        return SimpleNamespace(content=cert.public_bytes(serialization.Encoding.DER), raise_for_status=lambda: None)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if command[1] == 's_client':
            return SimpleNamespace(stdout=first.decode(), returncode=0)
        assert command[1] == 'verify'
        assert command[command.index('-CAfile')+1] == certifi.where()
        assert command[command.index('-verify_hostname')+1] == 'www.tpex.org.tw'
        assert '-untrusted' in command
        return SimpleNamespace(returncode=0 if valid else 1, stdout='OK', stderr='bad chain')
    monkeypatch.setattr(tls.requests, 'get', get)
    monkeypatch.setattr(tls.subprocess, 'run', run)
    if valid:
        result = tls.prepare(tmp_path)
        assert result['chain_and_hostname_verified']
        assert not result['tls_verification_disabled']
        assert (tmp_path/'bundle.pem').exists()
    else:
        with pytest.raises(ValueError, match='existing certifi roots'):
            tls.prepare(tmp_path)
        assert not (tmp_path/'bundle.pem').exists()
