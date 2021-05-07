"""pump dataset."""

import tensorflow_datasets.public_api as tfds
import tensorflow as tf
import os
import re

_DESCRIPTION = """
This dataset belongs to the "development dataset" for the DCASE 2020 Challenge Task 2 "Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring"
"""

_CITATION = """
@dataset{yuma_koizumi_2020_3678171,
  author       = {Yuma Koizumi and
                  Yohei Kawaguchi and
                  Keisuke Imoto and
                  Toshiki Nakamura and
                  Yuki Nikaido and
                  Ryo Tanabe and
                  Harsh Purohit and
                  Kaori Suefusa and
                  Takashi Endo and
                  Masahito Yasuda and
                  Noboru Harada},
  title        = {DCASE 2020 Challenge Task 2 Development Dataset},
  month        = mar,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.3678171},
  url          = {https://doi.org/10.5281/zenodo.3678171}
}
"""


_HOMEPAGE_URL = "https://zenodo.org/record/3678171"
_DOWNLOAD_URL = "https://zenodo.org/record/3678171/files/dev_data_pump.zip"


class Pump(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for pump dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "audio": tfds.features.Audio(
                        file_format="wav", sample_rate=16_000, shape=(160_000,)
                    ),
                    "audio/id": tfds.features.Text(),
                    "audio/machine": tfds.features.Text(),
                    "audio/split": tfds.features.ClassLabel(names=["train", "test"]),
                    "label": tfds.features.ClassLabel(names=["normal", "anomaly"]),
                }
            ),
            supervised_keys=("audio", "label"),
            homepage=_HOMEPAGE_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(_DOWNLOAD_URL)

        return {
            "train": self._generate_examples(os.path.join(path, "pump", "train")),
            "test": self._generate_examples(os.path.join(path, "pump", "test")),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        for root, _, file_name in tf.io.gfile.walk(path):
            for idx, fname in enumerate(file_name):
                path = os.path.join(root, fname)
                if fname.endswith(".wav"):
                    info = re.search(
                        r"(train|test).(normal|anomaly)_id_(\d{2})_\d{4}(\d{4})",
                        path,
                    )
                    split, label, machine_id, audio_id = info.groups()
                    example = {
                        "audio": path,
                        "audio/id": audio_id,
                        "audio/machine": machine_id,
                        "audio/split": split,
                        "label": label,
                    }
                    yield idx, example
