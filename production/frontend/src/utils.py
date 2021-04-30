import os
import re
from typing import List, Tuple


def get_files(split: str) -> List[str]:  # TODO: Finish it
    dir_root = os.path.join("..", "media", "audios")
    return [
        os.path.join(dir_root, "anomaly_id_00_00000000.wav"),
        os.path.join(dir_root, "anomaly_id_00_00000001.wav"),
        os.path.join(dir_root, "anomaly_id_00_00000002.wav"),
        os.path.join(dir_root, "anomaly_id_00_00000003.wav"),
    ]


def get_train_files(machine_id: str) -> List[str]:  # TODO: Finish it
    return get_files("train")


def get_test_files() -> List[str]:  # ? Not used yet
    return get_files("test")


def get_feature_img(fname: str) -> str:
    parts = fname.split(os.path.sep)
    fname = f"{parts[-1][:-4]}.png"
    dir_root = os.path.join("..", "media", "images")
    return os.path.join(dir_root, fname)


def get_info(fname: str) -> Tuple[str, str, str]:
    info = re.search(r"(normal|anomaly)_id_(\d{2})_(\d{8})", fname)
    label, machine_id, audio_id = info.groups()
    return label, machine_id, audio_id
