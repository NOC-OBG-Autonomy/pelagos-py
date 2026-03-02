# This file is part of the NOC Autonomy Toolbox.
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

"""Writes reports on the current data passed through the pipeline."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
import toolbox.utils.diagnostics as di

#### Custom imports ####
from rstcloth import RstCloth
from datetime import datetime, timezone
import getpass
import platform
import subprocess
import json
from importlib.metadata import version
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from tqdm import tqdm
import numpy as np

from pathlib import Path


def current_info() -> dict:
    """Returns current operator information from when the report is being generated."""

    now = datetime.now(timezone.utc)

    info = {
        "timestamp_utc": now.isoformat(),
        "user": getpass.getuser(),
        "toolbox_version": version(
            "NOCAT"
        ),  #   Normally done with __version__. "NOCAT" placeholder until name is known.
        "python_version": platform.python_version(),
        "system": f"{platform.system()}: {platform.release()}",
    }

    return info


def write_conf_py(
    source_dir,
    project="Pipeline Report",
    author="Unknown",
    master_doc="index",
    subtitle=None,
) -> None:
    """
    Write a minimal Sphinx conf.py suitable for PDF builds.

    To be passed into Sphinx.

    Parameters
    ----------
    source_dir : str or Path
        Directory containing the .rst file(s), where this will be saved.
    project : str
        Project title.
    author : str
        Author name.
    master_doc : str
        Root rst file (without .rst).
    """
    #   TODO: Add mission

    #   Save conf.py in same directory as the .rst files
    source_dir = Path(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    conf_py = source_dir / "conf.py"
    subtitle_line = subtitle or ""

    # year = datetime.now(timezone.utc).year
    # copyright = "{year}, {author}"

    conf_text = f"""
# -- Auto-generated Sphinx configuration --
#   See https://www.sphinx-doc.org/en/master/usage/configuration.html

project = {project!r}
author = {author!r}

copyright = "%Y, {author}"

# version = 
# release = 

extensions = []

templates_path = ["_templates"]
exclude_patterns = []

master_doc = {master_doc!r}

# -- Options for LaTeX output --

latex_elements = {{
  'extraclassoptions': 'openany,oneside',
  'papersize': 'a4paper',

  'maketitle': r'''
\\begin{{titlepage}}
    \\centering
    \\vspace*{{3cm}}

    {{\\Huge \\bfseries {project} \\par}}
    \\vspace{{0.5cm}}

    {{\\Large {subtitle_line} \\par}}
    \\vspace{{1.5cm}}

    {{\\large {author} \\par}}
    \\vfill

    {{\\large \\today \\par}}
\\end{{titlepage}}
'''
}}
   # For cutting out blank pages (intended for single-sided printing)

latex_documents = [
    (
        master_doc,
        "{project.replace(" ", "_")}.tex",
        project,
        author,
        "manual",
    ),
]
"""

    conf_py.write_text(conf_text.strip() + "\n")


def run_sphinx(source_dir, build_dir=None) -> None:
    """
    Build a PDF from a Sphinx source directory using the latexpdf builder.

    This step requires Sphinx binaries to be installed and usable on the current workstation.
    Requires a conf.py to be located in the source directory.

    Parameters
    ----------
    source_dir : str or Path
        Directory containing the .rst and conf.py files.
    build_dir : str or Path
        Directory where Sphinx output can be placed. Defaults to source_dir/_build.
    """
    source_dir = Path(source_dir).resolve()  # For simlinks

    conf_py = source_dir / "conf.py"
    if not conf_py.exists():
        # User needs to run write_conf_py first.
        raise RuntimeError(f"conf.py not found in {source_dir}")

    if build_dir is None:
        build_dir = source_dir / "_build"
    else:
        build_dir = Path(build_dir).resolve()

    subprocess.run(
        [
            "sphinx-build",
            "-M",  # Make-mode, to use a builder
            "latexpdf",  # target
            str(source_dir),
            str(build_dir),
            "-q",  # quiet, comment this and next 4 lines for logger
        ],
        check=True,  # If errors, raise an exception
        capture_output=True,  # Suppress terminal output of stdout and stderr
        text=True,  # Get text output
    )  # See sphinx docs at https://www.sphinx-doc.org/en/master/man/sphinx-build.html


def build_qc_dict(data: xr.Dataset) -> dict:
    """
    Return a dictionary of all QC variable names and their corresponding QC attributes.

    Can be expanded in the future if additional attributes related to testing are added.
    Tests are ID'd using `_flag_cts` suffix in variable test parameters

    Parameters
    ----------
    data : Xarray DataSet
        The top level data containing all the relevant QC variables.

    Returns
    -------
    qc_dict : dict
        Nested dictionaries of QC variables with test names and results.

        Structure:
        {
            "VAR_QC": {
                "test_name": {
                    "params": {...},
                    "flag_counts": {...},
                    "stats": {...},
                },
                "test_name_2": {
                    ...
                },
            }
        }
    TODO: Move to utils? Does it belong here?
    """
    qc_dict = {}
    for var in data.data_vars:
        if not var.endswith("_QC"):
            continue

        attrs = data[var].attrs
        qc_dict[var] = {}

        # ID tests that were run for indexing.
        # _flag_cts seems like the least standardized name to ID qc with
        test_names = [
            attr.replace("_flag_cts", "")
            for attr in attrs
            if attr.endswith("_flag_cts")
        ]

        for test in test_names:
            params_key = f"{test}_params"
            flag_key = f"{test}_flag_cts"
            stats_key = f"{test}_stats"

            # Safely load JSON fields if present (how they were likely saved)
            params = json.loads(attrs[params_key]) if params_key in attrs else {}
            flag_cts = json.loads(attrs[flag_key]) if flag_key in attrs else {}
            stats = json.loads(attrs[stats_key]) if stats_key in attrs else {}

            qc_dict[var][test] = {
                "params": params,
                "flag_counts": flag_cts,
                "stats": stats,
            }

    return qc_dict


def flatten_qc_dict(qc_dict: dict) -> list:
    """
    Flatten QC dictionary into list of table rows.
    
    Intended for use in report metrics (RstCloth).

    Parameters
    ----------
    qc_dict : dict
        Dictionary of QC results.
    
    Returns
    -------
    rows: list of list
        A list of rows suitable for tabular display. Each row is a list:
            [qc_var, test_name, flag, formatted_count]
        - `qc_var` : str, the QC variable name
        - `test_name` : str, the name of the QC test
        - `flag` : str, QC flag value
        - `formatted_count` : str, count formatted with thousands separator
    """
    rows = []

    for qc_var, tests in qc_dict.items():
        if not tests:
            continue

        for test_name, test_data in tests.items():
            stats = test_data.get("stats", {})
            flag_counts = test_data.get("flag_counts", {})

            for flag, count in flag_counts.items():
                if count == 0:
                    continue

                rows.append(
                    [
                        qc_var,
                        test_name,
                        flag,
                        f"{count:,}",
                    ]
                )

    return rows


### RST builders


def run_info_page(rs, params_dict: dict, glatters: dict) -> None:
    """
    Writes a page dedicated to pipeline run information.

    Parameters
    ----------
    rs : RstCloth
        Active RstCloth stream to which the page is written.
    params_dict : dict
        Dictionary of global pipeline parameters.
    glatters : dict
        Dictionary describing the glider and mission. OG1 includes "platform_vocabulary" for consistency.
    """
    rs.h2("Pipeline run information")

    run_data = current_info()

    rs.table(
        data=[[key, str(value)] for key, value in run_data.items()],
        header=["", "Run metadata"],
    )
    rs.table(
        data=[[key, str(value)] for key, value in params_dict.items()],
        header=["", "Pipeline parameter"],
    )
    if "platform_vocabulary" in glatters:  #   May not be in every dataset
        rs.table_list(
            data=[[key, str(value)] for key, value in glatters.items()],
            headers=["", "Glider information"],
            widths=[30, 70],
        )


def add_log(logfile, rs, ncols=4) -> None:
    """
    Add and format the logfile as a table.

    Note: Requires a designated log_file be initialized in the global pipeline configuration parameters.
    """

    rs.h2("Logfile of run")
    rs.newline()

    rows = []
    with open(logfile, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(" - ", maxsplit=ncols - 1)
            if len(parts) != ncols:
                # malformed or unexpected line, skip or log
                continue

            timestamp, level, location, message = parts
            #   Remove date, toolbox for more space for comment
            timestamp = timestamp.split(" ")[1]
            location = location.removeprefix("toolbox.")

            rows.append((timestamp, level, location, message))
        f.close()
    #   Apply enough padding to the rows so that the report registers as "long enough" to format correctly
    minlen = 28  #   approx for A4 in testing
    if len(rows) < minlen:
        blank_row = tuple("" for _ in range(ncols))
        rows.extend([blank_row] * (minlen - len(rows)))

    rs.table_list(
        headers=["Time", "Level", "Location", "Message"],
        data=rows,
        widths=[11, 10, 24, 55],
        # width=100
    )
    rs.newline()


def qc_section(doc, data: xr.Dataset) -> None:
    """
    Wrapper for the QC section.

    Parameters
    ----------
    doc : RstCloth object
        The active RstCloth stream to be written to
    data : xarray.core.dataset.Dataset
        The entire dataset, including attributes
    """
    doc.h2("Quality Control Summary")
    doc.newline()

    #   Summary of flags from each test
    headers = [
        "QC Variable",
        "Test",
        "Flag",
        "Count",
    ]

    qc_dict = build_qc_dict(data)
    rows = flatten_qc_dict(qc_dict)

    if not rows:
        doc.paragraph("No QC tests found.")
        return

    doc.table(headers, rows)
    doc.newline()


def img_rst(doc, fname: str, fields: list = None):
    """
    Inserts image information into the .rst, as RstCloth does not support image insertion.

    See rst directives for image information (https://docutils.sourceforge.io/docs/ref/rst/directives.html#images)

    Parameters
    ----------
    doc : RstCloth object
        The active RstCloth stream to be written to
    fname : str
        The path or filename
    fields : list of tuple
        Image parameters to be written below the directive

    Example
    -------
    img_rst(doc,
            "../examples/data/OG1/testing/fig.png",
            fields=[("height","100px"),("width","100px")])
    would write out
    .. image:: fig.*
       :height: 100px
       :width: 100px
    """
    #   Sphinx is constrained to /outdir, lop that and the extension off
    new_name = fname.split("/")[-1].split(".")[0] + ".*"
    doc.hint(name="image", arg=new_name, fields=fields)
    doc.newline()
    doc.newline()


def basic_geo(doc, data, g_extent, ext, outdir):
    ax0 = plt.axes(projection=ccrs.PlateCarree())
    ax0.set_extent(g_extent, crs=ccrs.PlateCarree())
    ax0.add_feature(cfeature.LAND.with_scale("110m"))
    ax0.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax0.coastlines(resolution="110m")
    ax0.scatter(
        data["LONGITUDE"],
        data["LATITUDE"],
        s=5,
        color="red",
        marker="+",
        transform=ccrs.PlateCarree(),
    )
    plt.title("Glider Track")
    fname = outdir + f"geographic{ext}"
    plt.savefig(fname)
    img_rst(doc, fname)


def inset_geo(
    doc,
    data,
    outdir: str = "./",
    g_extent: list = [7, 25, 54, 65],
    scale: str = "110m",
    ext: str = ".png",
):
    """
    Creates an inset geographic of two plots for additional positional awareness.

    Unlike basic_geo(), this function will create an inset to make it clearer
    where the glider is operating.

    If the chart looks chunky, consider increasing the resolution in the `scale` arg.

    Parameters
    ----------
    doc : RstCloth object
        The active RstCloth stream to be written to
    data : xarray.core.dataset.Dataset
        The entire dataset, including attributes
    outdir : str
        The path to return figures to. Defaults to current directory.
    g_extent : list
        Geographic extent for cartopy geographic plot ([lon1, lon2, lat1, lat2]). Defaults to Baltic Sea.
    scale: str
        Resolution for cartopy to use when adding elements ("10m", "50m", "110m")
    ext : str
        Image filetype extension (.png, .svg, etc.)
    """
    fig = plt.figure(figsize=(8, 6))

    lon = data["LONGITUDE"].values
    lat = data["LATITUDE"].values
    lon_min = np.nanmin(lon)
    lon_max = np.nanmax(lon)
    lat_min = np.nanmin(lat)
    lat_max = np.nanmax(lat)

    #   Get the middle of the glider track
    lon_mid = 0.5 * (lon_min + lon_max)
    lat_mid = 0.5 * (lat_min + lat_max)
    pad = 0.1  #   Add some padding in degrees
    lon_span = (
        lon_max - lon_min
    ) + 2 * pad  # Full lat/lon sizes spanning the mission range
    lat_span = (lat_max - lat_min) + 2 * pad

    #   Use the larger span to force glider image to be a square
    span = max(lon_span, lat_span)
    track_extent = [
        lon_mid - span / 2,
        lon_mid + span / 2,
        lat_mid - span / 2,
        lat_mid + span / 2,
    ]  #   Glider data track extent

    #   Glider track on main axes
    ax_main = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax_main.set_extent(track_extent, crs=ccrs.PlateCarree())

    ax_main.scatter(
        lon,
        lat,
        s=5,
        color="red",
        marker="+",
        transform=ccrs.PlateCarree(),
    )
    gl = ax_main.gridlines(
        draw_labels=True,  # show tick labels
        dms=True,  # degrees, minutes, seconds
        x_inline=False,
        y_inline=False,
        linewidth=0.5,
        color="gray",
        alpha=0.7,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True

    ax_main.coastlines(resolution=scale)
    ax_main.set_title(f"Glider Track: {data.attrs['dataset_id']}")

    # Inset figure on new axes
    inset_ax = fig.add_axes(
        [0.23, 0.05, 0.2, 0.3],  # HA low, VA low, HA hi, VA hi
        projection=ccrs.PlateCarree(),
    )

    inset_ax.set_extent(g_extent, crs=ccrs.PlateCarree())  #   Default Batlic sea
    inset_ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.5)
    inset_ax.add_feature(cfeature.LAND.with_scale(scale))
    inset_ax.add_feature(cfeature.LAKES.with_scale(scale))
    inset_ax.coastlines(resolution=scale)

    inset_ax.plot(
        [
            track_extent[0],
            track_extent[1],
            track_extent[1],
            track_extent[0],
            track_extent[0],
        ],
        [
            track_extent[2],
            track_extent[2],
            track_extent[3],
            track_extent[3],
            track_extent[2],
        ],
        transform=ccrs.PlateCarree(),
        color="red",
        linewidth=1.2,
    )  # Draw box on top of inset

    #   Save the figure and write to .rst
    fname = outdir + f"geographic{ext}"
    plt.savefig(fname)
    plt.close(fig)

    img_rst(doc, fname)


def qc_hist(
    doc,
    data: xr.Dataset,
    outdir: str,
    var: str,
    xlims: list = [-0.6, 9.6],
    hislim=range(10),
    bins=None,
    ext=".png",
):
    """
    Create quick quality control histogram figure.

    Left axis:  Quick plot of QC variable's parent
    Right axis: Bins of each flag type, labeled with # of points

    Parameters
    ----------
    doc : RstCloth object
        The active RstCloth stream to be written to
    data : xarray.core.dataset.Dataset
        The entire dataset, including attributes
    var : str
        The QC variable as listed in `data`
    ext : str
        Image filetype extension (.png, .svg, etc.)
    hislim : array-like
        All potential flags of the selected schema (default Argo = 0 to 9, 10 total)
    bins : array-like
        The sequence of bin edges for collection, matching the dimension of hislim
    xlims : list
        Histogram axis bounds. Defaults to Argo (10 flags) with 0.1 padding on each side
    """

    var_source = var[:-3]  #   TEMP_QC --> TEMP

    fig, axs = plt.subplots(ncols=2, figsize=(8, 4), layout="constrained")

    #   Prepare the histogram
    ylims = [1, len(data[var])]  #   Log axis cannot be 0
    if any(y < 1 for y in ylims):
        raise ValueError
    if bins == None:  #   If not specified, center the bins around each flag integer
        bins = np.arange(len(hislim)) - 0.5

    #   Plot the source variable using xarray.plot for speed
    data[var_source].plot(ax=axs[0])
    axs[0].set_title(f"{var_source}: n={len(data[var_source])}", ha="right")

    data[var].plot.hist(
        yscale="log", bins=bins, xticks=hislim, xlim=xlims, ylim=ylims, ax=axs[1]
    )
    bars = axs[1].containers[0]  #   Number of points in each bin
    axs[1].bar_label(bars, fontsize=7, label_type="center")
    axs[1].set_yscale("log")
    axs[1].set_title(f"{var} flag histogram", ha="right")
    fig.supylabel(data.attrs["dataset_id"])

    fname = outdir + var + ext
    plt.savefig(fname)  #   Save to the outdir
    img_rst(doc, fname)  #   Pass along doc


def make_plots(
    doc,
    data: xr.Dataset,
    outdir: str,
) -> None:
    """
    Wrapper for plotting glider QC variables quickly.

    There are millions of points per variable, which xarray can plot very quickly
    in specific ways.

    Parameters
    ----------
    doc : RstCloth object
        The active RstCloth stream to be written to
    data : xarray.core.dataset.Dataset
        The entire dataset, including attributes
    outdir : str
        The path to return figures to
    ext : str
        Image filetype extension (.png, .svg, etc.)
    g_extent : list
        Geographic extent for cartopy geographic plot. Defaults to Baltic Sea.

    TODO: Define long-term storage for this. Is `diagnostics` the right place?
    """
    doc.h2("Plots")

    # Basic geographic plot
    # basic_geo(doc, data, g_extent, ext, outdir)
    inset_geo(doc, data, outdir, scale="50m")

    qc_vars = [var for var in data.data_vars if "_QC" in var]

    for var in tqdm(
        qc_vars,
        colour="green",
        desc=f"\033[97mProgress \033[0m",
        unit="vars",
    ):
        # Any form of scatter takes ~30 sec, stick with xarray.plot for now (no colorbars, alternative color schemes)
        qc_hist(doc, data, outdir, var)


@register_step
class WriteDataReport(BaseStep):
    """
    Writes a report summarizing the generic plots and statistics of the data.

    Base template:
    * Title page (automatically handled by sphinx)
    * Quality control summary
    * Basic plots
    * Run metadata and pipeline parameters
    * Logfile

    Parameters
    ----------
    title: str
        Name of the report (on title page and filename)
    output_path: str
        Directory to write the report to (must end with a "/")
    build: bool
        Whether to run Sphinx to build the PDF after writing the .rst and conf.py files
    """

    step_name = "Write Data Report"

    def run(self) -> xr.DataArray:
        odir = self.context["global_parameters"]["out_directory"]
        fout = odir + self.parameters.get("fname")
        data = self.context.get("data")

        if "dataset_id" not in data.attrs:
            self.log_warn("Dataset ID missing from OG1 file. Reporting with unk platform information.")
            data.attrs["dataset_id"] = "unknown"

        with open(fout, "w") as output_file:
            doc = RstCloth(output_file)
            doc.h2("RstCloth will consume this h2, some kind of bug.")
            doc.newline()

            qc_section(doc, data)

            self.log("Generating images.")
            make_plots(doc, data, outdir=odir)

            run_info_page(
                doc, self.context["global_parameters"], self.context["data"].attrs
            )

            log_path = odir + self.context["global_parameters"]["log_file"]
            add_log(log_path, doc)

        #   Run sphinx if user defined in step parameters
        if self.parameters.get("build", True):
            # Sphinx requires a conf.py file to build
            self.log("Building PDF report with Sphinx.")
            self.log_warn(
                "Lines below this will not be captured in the run report. See logfile if other steps follow this one."
            )
            write_conf_py(
                odir,
                project=self.parameters["title"],
                author=current_info().get("user"),
                master_doc=self.parameters.get("fname").replace(".rst", ""),
                subtitle=data.attrs.get("dataset_id").replace("_", "-"),
            )
            run_sphinx(
                odir,
                build_dir=odir + "_build",
            )  # TODO: Make this more robust and less hardcoded

        return self.context


### Legacy code below this line


# def run_sphinx(source_dir, build_dir=None):
#     """
#     Build a PDF from a Sphinx source directory.
#     """
#     source_dir = Path(source_dir).resolve()

#     conf_py = source_dir / "conf.py"
#     if not conf_py.exists():
#         raise RuntimeError(f"conf.py not found in {source_dir}")

#     if build_dir is None:
#         build_dir = source_dir / "_build"
#     else:
#         build_dir = Path(build_dir).resolve()

#     latex_dir = build_dir / "latex"

#     subprocess.run(
#         [
#             "sphinx-build",
#             "-q",   # Run sphinx in quiet mode
#             "-c",
#             str(source_dir),  # ← EXPLICIT conf.py location
#             "-b",
#             "latex",
#             str(source_dir),
#             str(latex_dir),
#         ],
#         check=True,
#     )

#     ### Temporary: Figuring out make latexpdf
#     # makefile = latex_dir / "Makefile"
#     # makefile.write_text(
#     #     "SPHINXBUILD   = sphinx-build\n"
#     #     "SOURCEDIR     = ../..\n"
#     #     "BUILDDIR      = .\n"
#     #     "CONFDIR       = ../..\n\n"
#     #     ".PHONY: latexpdf\n\n"
#     #     "latexpdf:\n"
#     #     "\t$(SPHINXBUILD) -c $(CONFDIR) -b latexpdf $(SOURCEDIR) $(BUILDDIR)\n"
#     # )

#     tex_file = latex_dir / "Voto_Glider_Data_Pipeline_Report.tex"

#     subprocess.run(
#         [
#             "pdflatex",
#             "-q",
#             "-interaction=nonstopmode",
#             tex_file.name
#         ],
#         cwd=latex_dir,
#         check=True,
#     )

#     return latex_dir
