# This file is part of pelagos_py.
#
# Copyright 2025-2026 National Oceanography Centre and The Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np


def safe_median_datetime(x: np.ndarray, axis=None, **kwargs) -> np.datetime64:
    """
    Safely compute the median of datetime64[ns] array using pandas.

    Parameters
    ----------
    x : np.ndarray
        A 1D array of datetime64 values.

    Returns
    -------
    np.datetime64
        Median datetime or NaT if input is empty/all-NaT.
    """
    x = pd.to_datetime(x)

    if isinstance(x, pd.DatetimeIndex):
        x = pd.Series(x)

    if x.empty or x.isna().all():
        return np.datetime64("NaT")

    return x.median()


def add_datetime_secondary_xaxis(ax, position="top", rotation=45):
    """
    Add a secondary datetime x-axis (on top) that mirrors the main x-axis ticks and labels.
    """
    # Create secondary axis with identity transform
    secax = ax.secondary_xaxis(position, functions=(lambda x: x, lambda x: x))

    # Copy tick locator and formatter from the main axis
    secax.xaxis.set_major_locator(ax.xaxis.get_major_locator())
    secax.xaxis.set_major_formatter(ax.xaxis.get_major_formatter())

    # Set label and rotation
    secax.set_xlabel("Datetime")
    secax.tick_params(rotation=rotation)

    # Optional: alignment tweak
    for label in secax.get_xticklabels():
        label.set_ha("left")

    return secax
