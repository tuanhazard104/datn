# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from typing import Optional, Sequence, Union

import fire

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


if __package__ in (None, ""):
    from segmenter import Segmenter, run_segmenter
else:
    from .segmenter import Segmenter, run_segmenter


class InferClass:
    def __init__(self, config_file: Optional[Union[str, Sequence[str]]] = None, rank: int = 0, **override):
        override["infer#enabled"] = True
        self.segmenter = Segmenter(rank=rank, config_file=config_file, config_dict=override)

    def infer(self, image_file):
        pred = self.segmenter.infer_image(image_file, save_mask=False)
        return pred


def run(config_file: Optional[Union[str, Sequence[str]]] = None, **override):
    override["infer#enabled"] = True
    run_segmenter(config_file=config_file, **override)


if __name__ == "__main__":
    fire.Fire()
