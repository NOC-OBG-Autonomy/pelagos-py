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

"""Archive file of unused QC tests for reference."""

# import xarray as xr
# from datetime import datetime
# from geodatasets import get_path
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import matplotlib
# import numpy as np
# import shapely as sh
# import geopandas
#
#
#
# def range_test(df, variable_name, limits, flag):
#     # Not the most efficient implementation because of second for loop.
#     df = df.with_columns(
#         ((pl.col(variable_name) > limits[0]) & (pl.col(variable_name) < limits[1]))
#         .not_()
#         .cast(pl.Int64)
#         * flag.alias(f"{variable_name}_QC")
#     )
#     return df
#
#
# class test_template:
#     """
#     Target Variable:
#     Flag Number:
#     Variables Flagged:
#     ? description ?
#     """
#
#     name = ""
#     required_variables = []
#     qc_outputs = []
#
#     def __init__(self, df):
#         self.df = df
#         self.flags = None
#
#     def return_qc(self):
#         self.flags = None  # replace with processing of some kind
#         return self.flags
#
#     def plot_diagnostics(self):
#         # Any relevant diagnostic
#         pass
#
# # --------------------------- Universal tests ----------------------------

# def global_range_test(df):
#     """
#     Target Variable: PRES, TEMP, PRAC_SALINITY
#     Test Number: 6
#     Flag Number: 4, 3 (bad data, probably bad data)
#     Checks that the pressure, temperature and practically salinity do not lie outside expected
#     global extremes.
#     """
#     # Structured (variable_to_test, [lower_limit, upper_limit], variables_to_flag, flag)
#     test_calls = (
#         ("PRES", [-np.inf, -5], ["PRES", "TEMP", "PRAC_SALINITY"], 4),
#         ("PRES", [-5, -2.4], ["PRES", "TEMP", "PRAC_SALINITY"], 3),
#         ("TEMP", [-2.5, 40], ["TEMP"], 4),
#         ("PRAC_SALINITY", [2, 41], ["PRAC_SALINITY"], 4),
#     )
#
#
#
#     return df
#
#
# def regional_range_test(df):
#     """
#     Target Variable: TEMP, PRAC_SALINITY
#     Test Number: 7
#     Flag Number: 4 (bad data)
#     Checks that the temperature and practically salinity do not lie outside expected
#     regional (Mediterranean and Red Seas) extremes.
#     """
#
#     # Define Red and Mediterranean Sea areas
#     Red_Sea = sh.geometry.Polygon(
#         [(40, 10), (45, 14), (35, 30), (30, 30), (40, 10)]
#     )  # (lon, lat)
#     Med_Sea = sh.geometry.Polygon(
#         [(-6, 30), (40, 30), (35, 40), (20, 42), (15, 50), (-5, 40), (-6, 30)]
#     )
#     # Check if data falls in those areas
#     for poly, name in zip([Red_Sea, Med_Sea], ["in_red_sea", "in_med_sea"]):
#         df = df.with_columns(
#             pl.shape(["LONGITUDE", "LATITUDE"])
#             .map_batches(
#                 lambda x: sh.contains_xy(
#                     poly, x.struct.field("LONGITUDE"), x.struct.field("LATITUDE")
#                 )
#             )
#             .alias(f"{name}")
#         )
#
#     # define regional temperature and salinity limits
#     limits = {
#         "red": {"TEMP": (21, 40), "PRAC_SALINITY": (2, 41)},
#         "med": {"TEMP": (10, 40), "PRAC_SALINITY": (2, 40)},
#     }
#     # Check if the data satisfies these limits
#     for region, var_lims in limits.items():
#         for var, lims in var_lims.items():
#             df = df.with_columns(
#                 ((pl.col(var) > lims[0]) & (pl.col(var) < lims[1])).not_()
#             ).alias(f"bad_{region}_{var}")
#
#         # for the flagging, data must fail the regional test AND be within that region.
#         for var in ["TEMP", "PRAC_SALINITY"]:
#             df = df.with_columns(
#                 (
#                     (pl.col(f"bad_{region}_{var}") & pl.col(f"in_{region}_sea")).cast(
#                         pl.Int64
#                     )
#                     * 4
#                 ).alias(f"{var}_QC")
#             )
#
#     return df
#
#
# def pressure_increasing_test(df):
#     """
#     Target Variable: PRES
#     Test Number: 8
#     Flag Number: 4 (bad data)
#     Checks for any egregious spikes in pressure between consecutive points.
#     """
#
#     return df
#
#
# def spike_test(df):
#     """
#     Target Variable: TEMP, PRAC_SALINITY
#     Test Number: 9
#     Flag Number: 4 (bad data)
#     Checks for spikes in temperature and prctical salinity between nearest neighbour points points.
#     """
#
#     return df
