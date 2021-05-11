from utils import get_info
import pytest
import os


@pytest.mark.parametrize(
    "fname, label, machine_id, audio_id",
    [
        (
            "normal_id_06_00000099.wav",
            "normal",
            "06",
            "0099",
        ),
        (
            "anomaly_id_00_00000001.wav",
            "anomaly",
            "00",
            "0001",
        ),
        (
            os.path.join("pump", "test", "normal_id_02_00000123.wav"),
            "normal",
            "02",
            "0123",
        ),
        (
            os.path.join("pump", "test", "anomaly_id_04_00000045.wav"),
            "anomaly",
            "04",
            "0045",
        ),
        (
            os.path.join("pump", "test", "normal_id_02_00000100.wav"),
            "normal",
            "02",
            "0100",
        ),
        (
            os.path.join("pump", "test", "anomaly_id_00_00000002.wav"),
            "anomaly",
            "00",
            "0002",
        ),
    ],
)
def test_audio_fname(fname, label, machine_id, audio_id):
    assert (label, machine_id, audio_id) == get_info(fname)
