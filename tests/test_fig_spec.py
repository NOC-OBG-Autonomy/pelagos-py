import warnings

import matplotlib

matplotlib.use("Agg")

from pelagos_py.utils import fig_spec  # noqa: E402


def test_legend_skips_axes_without_labels():
    fig, axes = fig_spec.new_fig()
    ax = axes[0][0]
    ax.plot([0, 1], [0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig_spec.legend(ax)
    assert ax.get_legend() is None


def test_legend_drawn_for_labelled_series():
    fig, axes = fig_spec.new_fig()
    ax = axes[0][0]
    ax.plot([0, 1], [0, 1], label="TEMP")
    fig_spec.legend(ax)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["TEMP"]
