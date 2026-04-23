#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Filesystem helpers for RECAP derived datasets."""

from __future__ import annotations

import shutil
from pathlib import Path


def prepare_derived_dataset_root(
    output_root: str | Path | None, input_root: str | Path | None
) -> Path | None:
    """Ensure a derived dataset root is fresh and does not alias the source dataset."""
    if output_root is None:
        return None

    output_path = Path(output_root).expanduser().resolve()
    input_path = None if input_root is None else Path(input_root).expanduser().resolve()
    if input_path is not None and output_path == input_path:
        raise ValueError(f"Refusing to overwrite source dataset root: {output_path}")

    if output_path.exists():
        shutil.rmtree(output_path)
    return output_path
