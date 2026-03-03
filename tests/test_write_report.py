"""Tests the step 'Write Report'"""

#   Test module import
from toolbox.steps.custom import write_report
import pytest
from unittest.mock import patch, MagicMock  #   Patch for OS, MagicMock for .rst stream object and function calls

#   Other imports
from datetime import datetime, timezone
import xarray as xr
import json
import numpy as np


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
    )   #   Confirm it still works w/o optional subtitle being defined
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
        args = mock_run.call_args[0][0] #   first positional arg in subprocess.run
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
        build_dir_arg = called_args[-2] #   source_dir, build_dir, -q
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
""" #   Blank line should get ignored

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


def test_qc_section(qc_dataset):
    doc = MagicMock()   #   Fake object that records what is done to it (representing the file)

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
    with patch.object(write_report, "build_qc_dict", return_value={"dummy": "data"}), \
         patch.object(write_report, "flatten_qc_dict", return_value=[]):
        write_report.qc_section(doc, qc_dataset)

    doc.paragraph.assert_called_once_with("No QC tests found.")
    doc.newline.assert_called_once()    #   At the top
    doc.table.assert_not_called()


def test_info_page(tmp_path):
    doc = MagicMock()
    params = {"out_directory": "/tmp/", "log_file": "run.log"}
    glatters = {"glider_serial": "77",
                "dataset_id": "delayed_SEA077_M21",
                "start_date": "20230316T1019",
                "platform_vocabulary": "long NERC link",}

    with patch.object(write_report, "current_info", return_value={"user": "user_name"}):
        write_report.run_info_page(doc, params, glatters)

    assert doc.table.call_count == 2    #   2 for each time run_info_page is called
    doc.table_list.assert_called_once() #   "platform_vocabulary" should make it happen

    headers_arg = doc.table_list.call_args[1]["headers"]
    assert headers_arg == ["", "Glider information"]

    doc.h2.assert_called_once_with("Pipeline run information")


def test_img_rst():
    doc = MagicMock()
    write_report.img_rst(doc, "/some/output/dir/TEMP_QC.png")
    doc.directive.assert_called_once_with(
        name="image", arg="TEMP_QC.*", fields=None
    )
    assert doc.newline.call_count == 2


def test_basic_geo(qc_dataset, tmp_path):
    #   Confirm that the file gets made and img_rst is called
    doc = MagicMock()
    outdir = str(tmp_path) + "/"
    ext = ".png"
    g_extent = [10, 11, 57, 58]

    with (
        patch.object(write_report.plt, "savefig") as mock_save,
        patch.object(write_report, "img_rst") as mock_img,
    ):
        write_report.basic_geo(doc, qc_dataset, g_extent, ext, outdir)

    expected_fname = outdir + "geographic.png"
    mock_save.assert_called_once_with(expected_fname)
    mock_img.assert_called_once_with(doc, expected_fname)


def test_inset_geo(qc_dataset, tmp_path):
    doc = MagicMock()
    outdir = str(tmp_path) + "/"

    with (
        patch.object(write_report.plt, "savefig") as mock_save,
        patch.object(write_report.plt, "close") as mock_close,
        patch.object(write_report, "img_rst") as mock_img,
    ):
        write_report.inset_geo(doc, qc_dataset, outdir=outdir)

    expected_fname = outdir + "geographic.png"

    mock_save.assert_called_once_with(expected_fname)
    mock_close.assert_called_once()
    mock_img.assert_called_once_with(doc, expected_fname)


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
        {
            "TEMP_QC": xr.DataArray(np.array([])),
            "TEMP": xr.DataArray(np.array([]))
        },
        attrs={"dataset_id": "TEST_DS"}
    )

    with pytest.raises(ValueError):
        write_report.qc_hist(doc, empty_ds, outdir, "TEMP_QC")
    

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
    context = {
        "global_parameters": {
            "out_directory": str(tmp_path) + "/",
            "log_file": "run.log",
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