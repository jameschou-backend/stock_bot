import json
import pytest
from scripts.export_rotation_2024 import verify_rotation


def test_publication_cannot_compare_a_run_with_itself(tmp_path):
    with pytest.raises(ValueError,match='independent'):
        verify_rotation(tmp_path,tmp_path,tmp_path/'out.json')


def test_old_three_arm_report_cannot_publish_six_arm_rotation(tmp_path):
    for name in ('a','b'):
        folder=tmp_path/name;folder.mkdir()
        (folder/'report.json').write_text(json.dumps({'cases':{'original':{},'cap40':{},'benchmark':{}}}))
    with pytest.raises(ValueError,match='fixed rotation'):
        verify_rotation(tmp_path/'a',tmp_path/'b',tmp_path/'out.json')
