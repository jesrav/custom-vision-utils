import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import yaml


class ImageDataSetInterface(ABC):
    @abstractmethod
    def append(self, other) -> None:
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def from_config(self, yaml_config_path: Union[Path, str]):
        pass

    @abstractmethod
    def get_config(self):
        pass

    def write_config(self, outfile: Union[Path, str]):
        with open(outfile, "w") as f:
            yaml.dump(json.loads(self.get_config().json()), f)
