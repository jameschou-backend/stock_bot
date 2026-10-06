import hashlib
import json

import pytest

from scripts.publish_research_terminal import BRIDGE, publish


def test_bookmark_migration_preserves_original_and_rejects_unknown_changes(tmp_path):
    source = tmp_path / '.cache/multi-strategy-scanner/20261006-oct05-v2'
    source.mkdir(parents=True)
    original = b'<html>sealed original scanner</html>'
    (source / 'index.html').write_bytes(original)
    (source / 'receipt.json').write_text(json.dumps({'files_sha256': {
        'index.html': hashlib.sha256(original).hexdigest()}}))
    publish(tmp_path)
    output = tmp_path / 'artifacts/reports'
    current = output / 'multi_strategy_scanner.html'
    assert current.read_text() == BRIDGE
    assert (output / 'multi_strategy_scanner_snapshot_20261006.html').read_bytes() == original
    publish(tmp_path)
    current.write_text('user change')
    with pytest.raises(ValueError, match='其他變更'):
        publish(tmp_path)
    assert current.read_text() == 'user change'
