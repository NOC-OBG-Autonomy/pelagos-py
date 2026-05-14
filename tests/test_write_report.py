"""Tests the step 'Write Report'"""

#   Test module import
from toolbox.steps.custom import write_report
import pytest
from unittest.mock import (
    patch,
    MagicMock,
)  #   Patch for OS, MagicMock for .rst stream object and function calls

#   Other imports
from datetime import datetime, timezone
import xarray as xr
import json
import numpy as np
from importlib.metadata import PackageNotFoundError


@pytest.fixture
def qc_dataset():
    ds = xr.Dataset()

    attrs = {
        "range_test_flag_cts": json.dumps({"1": 10, "4": 2}),
        "range_test_stats": json.dumps({"mean": 1.2}),
        "range_test_params": json.dumps({"threshold": [-2.5, 40]}),
        "gross_range_test_flag_cts": json.dumps({"1": 8}),
        # intentionally omit stats to test default {}
        "gross_range_test_params": json.dumps({"fail": [0, 20]}),
    }

    ds["TEMP_QC"] = xr.DataArray(np.zeros(5), attrs=attrs)

    #   Non-QC variable for reference
    ds["TEMP"] = xr.DataArray(np.zeros(5))
    ds["LONGITUDE"] = xr.DataArray(np.linspace(10, 11, 5))  #   Near gburg
    ds["LATITUDE"] = xr.DataArray(np.linspace(57, 58, 5))
    ds.attrs["dataset_id"] = "glider_test_run"

    return ds


def test_current_info():
    #   Better test - uses mock to emulate OS with context managers
    #   import getpass becomes import write_report.getpass within mock
    fixed_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    with (
        patch.object(write_report, "datetime") as mock_datetime,
        patch.object(write_report.getpass, "getuser", return_value="aaron-mau"),
        patch.object(write_report, "version", return_value="0.1.dev318+gdaacfb0d8"),
        patch.object(write_report.platform, "python_version", return_value="3.14.2"),
        patch.object(write_report.platform, "system", return_value="Linux"),
        patch.object(write_report.platform, "release", return_value="6.17.0-8-generic"),
    ):
        mock_datetime.now.return_value = fixed_now
        result = write_report.current_info()

    expected = {
        "timestamp_utc": fixed_now.isoformat(),
        "user": "aaron-mau",
        "toolbox_version": "0.1.dev318+gdaacfb0d8",
        "python_version": "3.14.2",
        "system": "Linux: 6.17.0-8-generic",
    }

    assert result == expected

    #   Do it again, but throw the PackageNotFoundError
    with (
        patch.object(write_report, "datetime") as mock_datetime,
        patch.object(write_report.getpass, "getuser", return_value="aaron-mau"),
        patch.object(write_report, "version", side_effect=PackageNotFoundError),
        patch.object(write_report.platform, "python_version", return_value="3.14.2"),
        patch.object(write_report.platform, "system", return_value="Linux"),
        patch.object(write_report.platform, "release", return_value="6.17.0-8-generic"),
    ):
        mock_datetime.now.return_value = fixed_now
        result = write_report.current_info()

    expected["toolbox_version"] = "unknown"

    assert result == expected


def test_write_conf_py(tmp_path):
    conf = tmp_path / "conf.py"
    #   conf.py is written to the specified source directory and contains expected params
    write_report.write_conf_py(
        tmp_path,
        project="Test project",
        author="Author A",
        master_doc="index",
        subtitle="Sub title",
    )
    assert conf.exists()
    content = (conf).read_text()
    assert "Test project" in content
    assert "Author A" in content
    assert "master_doc = 'index'" in content
    assert "Sub title" in content

    #   Nesting
    nested = tmp_path / "a" / "b" / "c"
    write_report.write_conf_py(nested, project="P", author="A")
    assert (nested / "conf.py").exists()

    write_report.write_conf_py(
        tmp_path,
        project="Test project",
        author="Author A",
        master_doc="index",
    )  #   Confirm it still works w/o optional subtitle being defined
    content = (conf).read_text()
    assert "project = 'Test project'" in content


def test_run_sphinx(tmp_path):
    #   Should break if no conf.py is specified
    with pytest.raises(RuntimeError, match="conf.py not found"):
        write_report.run_sphinx(tmp_path)

    (tmp_path / "conf.py").write_text("# dummy config")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        write_report.run_sphinx(tmp_path)

        #   Check subprocess.run was called once
        assert mock_run.call_count == 1

        #   Grab the actual args
        args = mock_run.call_args[0][0]  #   first positional arg in subprocess.run
        assert args[0] == "sphinx-build"
        assert "-M" in args
        assert "latexpdf" in args
        assert str(tmp_path.resolve()) in args

    #   Now with a custom build_dir
    custom_build = tmp_path / "custom_build"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        write_report.run_sphinx(tmp_path, build_dir=custom_build)

        called_args = mock_run.call_args[0][0]
        build_dir_arg = called_args[-2]  #   source_dir, build_dir, -q
        assert build_dir_arg == str(custom_build.resolve())


def test_build_qc_dict(qc_dataset):
    result = write_report.build_qc_dict(qc_dataset)

    assert "TEMP_QC" in result
    assert "range_test" in result["TEMP_QC"]
    assert "gross_range_test" in result["TEMP_QC"]

    range_test = result["TEMP_QC"]["range_test"]

    assert range_test["params"] == {"threshold": [-2.5, 40]}
    assert range_test["flag_counts"] == {"1": 10, "4": 2}
    assert range_test["stats"] == {"mean": 1.2}


def test_flatten_qc_dict(qc_dataset):
    qc_dict = {
        "TEMP_QC": {
            "range_test": {
                "stats": {"min": 0, "max": 30},
                "flag_counts": {
                    0: 0,
                    1: 2212921,
                    2: 0,
                    3: 2500,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0,
                    9: 0,
                },
            }
        },
        "CNDC_QC": {},  # should be skipped entirely
    }

    result = write_report.flatten_qc_dict(qc_dict)

    expected = [
        ["TEMP_QC", "range_test", 1, "2,212,921"],
        ["TEMP_QC", "range_test", 3, "2,500"],
    ]

    assert result == expected
    assert "CNDC_QC" not in [item for sublist in expected for item in sublist]


### RST writers


def test_add_log(tmp_path):
    log_content = """\
2026-02-17 12:56:08 - INFO - toolbox.pipeline - Logging to file: /toolbox/examples/data/OG1/testing/processing.log
2026-02-17 12:56:08 - INFO - toolbox.pipeline - Assembling steps to run from config.
2026-02-17 12:56:08 - INFO - toolbox.pipeline - Step 'Load OG1' added successfully!

2026-02-17 12:56:18 - WARNING - toolbox.pipeline.step.Apply QC - [Apply QC] PROFILE_NUMBER_QC is all 0 after running all QC steps. Check intended QC variables and test requirements.
2026-02-17 12:56:23 - WARNING - toolbox.pipeline.step.Write Data Report - [Write Data Report] Lines below this will not be captured in the run report. See logfile if other steps follow this one.
"""  #   Blank line should get ignored

    logfile = tmp_path / "processing.log"
    logfile.write_text(log_content)

    doc = MagicMock()

    write_report.add_log(str(logfile), doc)
    doc.h2.assert_called_once_with("Logfile of run")

    #   Pull the data out of the output table
    kwargs = doc.table_list.call_args.kwargs
    assert kwargs["headers"] == ["Time", "Level", "Location", "Message"]
    rows = kwargs["data"]

    assert rows[0] == (
        "12:56:08",  #  Shouldn't have a date on it
        "INFO",
        "pipeline",  #  Prefix should be removed
        "Logging to file: /toolbox/examples/data/OG1/testing/processing.log",
    )

    # Check WARNING row with deeper path
    assert rows[3][0] == "12:56:18"
    assert rows[3][1] == "WARNING"
    assert rows[3][2] == "pipeline.step.Apply QC"

    # Ensure padding applied for formatting difficulties
    assert len(rows) >= 28

    doc.newline.assert_called()

    #   Something that cannot be split
    log_content = "This is not a properly formatted log line\n"
    logfile = tmp_path / "bad.log"
    logfile.write_text(log_content)

    doc = MagicMock()
    write_report.add_log(str(logfile), doc)

    kwargs = doc.table_list.call_args.kwargs
    rows = kwargs["data"]
    # Only padding rows should be present
    assert all(all(cell == "" for cell in row) for row in rows)


def test_add_cc_ascii(tmp_path):
    #   Make sure the bad chars get replaced
    ascii_content = """\


--------------------------------------------------------------------------------
                         IOOS Compliance Checker Report                         
                                 Version X.Y.Z                                  
                     Report generated 2026-01-01T00:00:00Z                      
                                     og:1.0                                     
  https://oceangliderscommunity.github.io/OG-format-user-manual/OG_Format.html  
--------------------------------------------------------------------------------
                               Corrective Actions                               
Replace these characters: α, β, γ, σ, μ, °, ±
"""
    # json_content =
    cc_file = tmp_path / "example_check.rst"
    cc_file.write_text(ascii_content)
    doc = MagicMock()

    write_report.add_cc(str(cc_file), doc)
    doc.h2.assert_called_once_with("Compliance Checker results")
    doc.codeblock.assert_called_once()

    codeblock_out = doc.codeblock.call_args.args[0]  #   Extract the actual text
    assert "oceangliderscommunity" in codeblock_out
    for k, v in write_report.REPLACEMENTS.items():
        if k in ascii_content:
            assert v in codeblock_out


def test_add_cc_json(tmp_path):
    cc_data = {
        "og": {
            "scored_points": 478,
            "possible_points": 524,
            "all_priorities": [
                {
                    "name": "Check for mandatory global attributes",
                    "weight": 3,
                    "value": [13, 17],
                    "msgs": [
                        "Global attribute contributing_institutions is missing",
                        "Global attribute contributing_institutions_role is missing",
                        "Global attribute contributing_institutions_role_vocabulary is missing",
                        "Global attribute start_date is missing",
                    ],
                    "children": [],
                },
                {
                    "name": "redundant test",  #   Should be redundant
                    "msgs": ["message 1", "message 2"],
                },
                {
                    "name": "redundant test",
                    "msgs": ["message 3"],
                },
            ],
        }
    }

    ccfile = tmp_path / "cc.json"
    ccfile.write_text(json.dumps(cc_data))

    doc = MagicMock()

    write_report.add_cc(str(ccfile), doc)
    doc.h2.assert_called_once_with("Compliance Checker results")
    doc.h3.assert_called_once_with("og: CC score of 478/524")

    # Break the table out
    kwargs = doc.table_list.call_args.kwargs
    assert kwargs["headers"] == ["Name", "og message"]

    rows = kwargs["data"]
    assert len(rows) == 7
    assert rows[0] == [
        "Check for mandatory global attributes",
        "Global attribute contributing_institutions is missing",
    ]
    assert rows[-1] == ["", "message 3"]  #   Should have a blank first column


def test_qc_section(qc_dataset):
    doc = (
        MagicMock()
    )  #   Fake object that records what is done to it (representing the file)

    write_report.qc_section(doc, qc_dataset)

    #   Header called once, table and newline called however many times
    doc.h2.assert_called_once_with("Quality Control Summary")
    assert doc.table.called
    assert doc.newline.called

    headers, rows = doc.table.call_args[0]

    assert headers == [
        "QC Variable",
        "Test",
        "Flag",
        "Count",
    ]
    assert len(rows) > 0


def test_qc_section_no_rows(qc_dataset):
    #   TODO: Break up tests like this
    doc = MagicMock()

    #   Patch build_qc_dict normally but force flatten_qc_dict to return empty
    with (
        patch.object(write_report, "build_qc_dict", return_value={"dummy": "data"}),
        patch.object(write_report, "flatten_qc_dict", return_value=[]),
    ):
        write_report.qc_section(doc, qc_dataset)

    doc.paragraph.assert_called_once_with("No QC tests found.")
    doc.newline.assert_called_once()  #   At the top
    doc.table.assert_not_called()


def test_info_page(tmp_path):
    doc = MagicMock()
    params = {"out_directory": "/tmp/", "log_file": "run.log"}
    glatters = {
        "glider_serial": "77",
        "dataset_id": "delayed_SEA077_M21",
        "start_date": "20230316T1019",
        "platform_vocabulary": "long NERC link",
    }

    with patch.object(write_report, "current_info", return_value={"user": "user_name"}):
        write_report.run_info_page(doc, params, glatters)

    assert doc.table.call_count == 2  #   2 for each time run_info_page is called
    doc.table_list.assert_called_once()  #   "platform_vocabulary" should make it happen

    headers_arg = doc.table_list.call_args[1]["headers"]
    assert headers_arg == ["", "Glider information"]

    doc.h2.assert_called_once_with("Pipeline run information")


def test_img_rst():
    doc = MagicMock()
    write_report.img_rst(doc, "/some/output/dir/TEMP_QC.png")
    doc.directive.assert_called_once_with(name="image", arg="TEMP_QC.*", fields=None)
    assert doc.newline.call_count == 2


def test_basic_geo(qc_dataset, tmp_path):
    #   Confirm that the file gets made and img_rst is called
    doc = MagicMock()
    mock_ax = MagicMock()
    outdir = str(tmp_path) + "/"
    ext = ".png"
    g_extent = [10, 11, 57, 58]

    with (
        patch.object(write_report.plt, "savefig") as mock_save,
        patch.object(write_report, "img_rst") as mock_img,
        patch.object(write_report.plt, "axes", return_value=mock_ax) as mock_axes,
    ):
        write_report.basic_geo(doc, qc_dataset, g_extent, ext, outdir)

    expected_fname = outdir + "geographic.png"
    mock_save.assert_called_once_with(expected_fname)
    mock_img.assert_called_once_with(doc, expected_fname)

    mock_axes.assert_called_once()
    mock_ax.set_extent.assert_called_once()
    mock_ax.add_feature.assert_called_once()
    mock_ax.gridlines.assert_called_once()
    mock_ax.coastlines.assert_called_once()
    mock_ax.scatter.assert_called_once()


def test_inset_geo(qc_dataset, tmp_path):
    doc = MagicMock()
    mock_fig = (
        MagicMock()
    )  #   for figure testing - fake fig, axes, gridlines (and more)
    mock_ax_main = MagicMock()  #   for ax_main
    mock_inset_ax = MagicMock()  #   for inset_ax
    mock_gl = MagicMock()  #   for the gridlines
    outdir = str(tmp_path) + "/"

    mock_ax_main.gridlines.return_value = mock_gl  #   gridlines()
    mock_fig.add_subplot.return_value = mock_ax_main  #   add_subplot()
    mock_fig.add_axes.return_value = mock_inset_ax  #   add_axes()

    with (
        patch.object(write_report.plt, "savefig") as mock_save,
        patch.object(write_report.plt, "close") as mock_close,
        patch.object(write_report, "img_rst") as mock_img,
        patch.object(write_report.plt, "figure", return_value=mock_fig) as mock_figure,
    ):
        write_report.inset_geo(doc, qc_dataset, outdir=outdir)

    expected_fname = outdir + "geographic.png"

    mock_save.assert_called_once_with(expected_fname)
    mock_close.assert_called_once()
    mock_img.assert_called_once_with(doc, expected_fname)

    mock_figure.assert_called_once_with(figsize=(8, 6))
    mock_fig.add_subplot.assert_called_once()
    mock_fig.add_axes.assert_called_once()
    mock_ax_main.set_extent.assert_called_once()
    mock_ax_main.scatter.assert_called_once()
    mock_ax_main.gridlines.assert_called_once()
    mock_ax_main.coastlines.assert_called_once()
    mock_ax_main.set_title.assert_called_once()
    assert mock_gl.top_labels is False
    assert mock_gl.right_labels is False
    assert mock_gl.bottom_labels is True
    assert mock_gl.left_labels is True
    mock_inset_ax.set_extent.assert_called_once()
    mock_inset_ax.add_feature.assert_called()
    mock_inset_ax.coastlines.assert_called_once()
    mock_inset_ax.plot.assert_called_once()


def test_qc_hist(qc_dataset, tmp_path):
    doc = MagicMock()
    outdir = str(tmp_path) + "/"

    with (
        patch.object(write_report.plt, "savefig") as mock_save,
        patch.object(write_report, "img_rst") as mock_img,
    ):
        write_report.qc_hist(doc, qc_dataset, outdir, "TEMP_QC")

    expected_fname = outdir + "TEMP_QC.png"

    mock_save.assert_called_once_with(expected_fname)
    mock_img.assert_called_once_with(doc, expected_fname)

    #   Make sure it breaks by resetting QC param
    empty_ds = xr.Dataset(
        {"TEMP_QC": xr.DataArray(np.array([])), "TEMP": xr.DataArray(np.array([]))},
        attrs={"dataset_id": "TEST_DS"},
    )

    with pytest.raises(ValueError):
        write_report.qc_hist(doc, empty_ds, outdir, "TEMP_QC")


def test_qc_hist_all_nan(qc_dataset, tmp_path):
    #   Repeat from before, but now test for when they are all NaN
    doc = MagicMock()
    outdir = str(tmp_path) + "/"

    #   Reset TEMP and QC to skip plotting and just do text
    qc_dataset["TEMP"] = xr.DataArray(
        np.full(5, np.nan), attrs=qc_dataset["TEMP"].attrs
    )
    qc_dataset["TEMP_QC"] = xr.DataArray(
        np.full(5, np.nan), attrs=qc_dataset["TEMP_QC"].attrs
    )

    #   More MagicMock objects to check on the actual figure params
    mock_fig = MagicMock()
    mock_ax0 = MagicMock()
    mock_ax1 = MagicMock()
    mock_ax1.containers = [MagicMock()]

    with (
        patch.object(
            write_report.plt, "subplots", return_value=(mock_fig, [mock_ax0, mock_ax1])
        ),
        patch.object(write_report.plt, "savefig"),
        patch.object(write_report, "img_rst"),
    ):
        write_report.qc_hist(doc, qc_dataset, outdir, "TEMP_QC")

    #   Should write text instead of plotting
    mock_ax0.text.assert_called_once()
    mock_ax0.set_title.assert_called_once()
    assert not mock_ax0.plot.called


def test_make_plots(qc_dataset, tmp_path):
    doc = MagicMock()
    outdir = str(tmp_path) + "/"

    with (
        patch.object(write_report, "inset_geo") as mock_inset,
        patch.object(write_report, "qc_hist") as mock_hist,
    ):
        write_report.make_plots(doc, qc_dataset, outdir)

    doc.h2.assert_called_once_with("Plots")
    mock_inset.assert_called_once()

    #   Can add more if qc_dataset is modified in tests
    mock_hist.assert_called_once_with(doc, qc_dataset, outdir, "TEMP_QC")


def test_write_data_report(tmp_path, qc_dataset):
    #   Test the whole step with Sphinx enabled to write out
    rst_file = tmp_path / "glider_test_run_check.rst"  # _check.rst Expected from the CC
    rst_file.write_text("")

    context = {
        "global_parameters": {
            "out_directory": str(tmp_path) + "/",
            "log_file": "run.log",
            "filename_core": "glider_test_run",
        },
        "data": qc_dataset,
    }
    (tmp_path / "run.log").write_text(
        "2026-02-17 12:00:00 - INFO - toolbox.pipeline - Test message\n"
    )

    step = write_report.WriteDataReport.__new__(write_report.WriteDataReport)
    step.context = context
    step.parameters = {"fname": "report.rst", "title": "Test Report", "build": True}
    step.log = MagicMock()
    step.log_warn = MagicMock()

    with (
        patch.object(write_report, "make_plots"),
        patch.object(write_report, "write_conf_py") as mock_conf,
        patch.object(write_report, "run_sphinx") as mock_sphinx,
    ):
        step.run()

    mock_conf.assert_called_once()
    mock_sphinx.assert_called_once()


def test_write_data_report_no_build(tmp_path, qc_dataset):
    #   Repeat above, no build
    rst_file = tmp_path / "glider_test_run_check.rst"  # _check.rst Expected from the CC
    rst_file.write_text("")

    context = {
        "global_parameters": {
            "out_directory": str(tmp_path) + "/",
            "log_file": "run.log",
            "filename_core": "glider_test_run",
        },
        "data": qc_dataset,
    }
    (tmp_path / "run.log").write_text(
        "2026-02-17 12:00:00 - INFO - toolbox.pipeline - Test message\n"
    )

    step = write_report.WriteDataReport.__new__(write_report.WriteDataReport)
    step.context = context
    step.parameters = {
        "fname": "report.rst",
        "title": "Test Report",
        "build": False,
    }

    step.log = MagicMock()
    step.log_warn = MagicMock()

    with (
        patch.object(write_report, "make_plots"),
        patch.object(write_report, "write_conf_py") as mock_conf,
        patch.object(write_report, "run_sphinx") as mock_sphinx,
    ):
        step.run()

    mock_conf.assert_not_called()
    mock_sphinx.assert_not_called()


def test_write_data_report_with_cc(tmp_path, qc_dataset):
    #   Since last tests were written, "filename_core" is automatically added to the global context when
    #   data are loaded and it serves as the default if the user doesn't specify title, fname
    context = {
        "global_parameters": {
            "out_directory": str(tmp_path) + "/",
            "log_file": "run.log",
            "filename_core": "test_filename",
            "cc_file": "cc.json",
        },
        "data": qc_dataset,
    }

    (tmp_path / "run.log").write_text("log line\n")

    step = write_report.WriteDataReport.__new__(write_report.WriteDataReport)
    step.context = context
    step.parameters = {
        "title": None,
        "fname": None,
        "build": False,
    }

    step.log = MagicMock()
    step.log_warn = MagicMock()

    with (
        patch.object(write_report, "RstCloth") as mock_rst,
        patch.object(write_report, "make_plots"),
        patch.object(write_report, "write_conf_py") as mock_conf,
        patch.object(write_report, "add_cc") as mock_cc,
    ):
        doc = MagicMock()
        mock_rst.return_value = doc
        step.run()

    mock_cc.assert_called_once()

    #   Check defaults
    doc.h1.assert_called_once_with(
        "Data report test filename"
    )  #   Underscores should have been removed
    assert (
        tmp_path / "test_filename.rst"
    ).exists()  #   Should default to filename_core


def test_write_data_report_missing_dataset_id(tmp_path, qc_dataset):
    #   Test again without dataset_id
    qc_dataset = qc_dataset.copy()
    qc_dataset.attrs.pop("dataset_id", None)

    rst_file = tmp_path / "glider_test_run_check.rst"  # _check.rst Expected from the CC
    rst_file.write_text("")

    context = {
        "global_parameters": {
            "out_directory": str(tmp_path) + "/",
            "log_file": "run.log",
            "filename_core": "glider_test_run",  # Manually add this since we aren't loading a dataset.
        },
        "data": qc_dataset,
    }
    (tmp_path / "run.log").write_text(
        "2026-02-17 12:00:00 - INFO - toolbox.pipeline - Test message\n"
    )

    step = write_report.WriteDataReport.__new__(write_report.WriteDataReport)
    step.context = context
    step.parameters = {
        "fname": "report.rst",
        "title": "Test Report",
        "build": True,
    }

    step.log = MagicMock()
    step.log_warn = MagicMock()

    with (
        patch.object(write_report, "make_plots"),
        patch.object(write_report, "write_conf_py") as mock_conf,
        patch.object(write_report, "run_sphinx"),
        patch.object(write_report, "current_info", return_value={"user": "tester"}),
    ):
        step.run()

    step.log_warn.assert_any_call(
        "Dataset ID missing from OG1 file. Reporting with unk platform information."
    )
    _, kwargs = mock_conf.call_args
    assert kwargs["subtitle"] == "Dataset ID: unknown dataset ID"
    assert (
        qc_dataset.attrs["dataset_id"] == "unknown dataset ID"
    )  #   This also should have changed
