import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from pelagos_py.steps.base_step import BaseStep
from pelagos_py.utils import diagnostic_capture


class FigStep(BaseStep):
    step_name = "Fig Step"
    parameter_schema = {}
    diagnostic_figures = {"a": ("A", True), "b": ("B", False), "c": ("C", None)}

    def draw_figure(self, name):
        self.drawn.append(name)
        return plt.figure()

    def run(self):
        self.drawn = []
        if self.diagnostics:
            self.generate_diagnostics()
        return self.context


def drawn(diagnostics):
    step = FigStep("Fig Step", diagnostics=diagnostics)
    with diagnostic_capture.force_headless_backend():  # no Tk window, plt.show is a no-op
        step.run()
    plt.close("all")
    return step.drawn


def test_true_draws_the_defaults():
    assert drawn(True) == ["a"]


def test_all_draws_everything():
    assert drawn("all") == ["a", "b"]


def test_list_selects_by_name():
    assert drawn(["b", "c"]) == ["b", "c"]  # 'c' is only reachable by name


def test_false_draws_nothing():
    assert drawn(False) == []


def test_unknown_name_halts():
    with pytest.raises(SystemExit):
        drawn(["nope"])


def test_capture_names_files_by_figure(tmp_path):
    images = []
    with diagnostic_capture.force_headless_backend(), \
            diagnostic_capture.capture_figures(str(tmp_path), "Fig Step", 3, images):
        FigStep("Fig Step", diagnostics="all").run()
    diagnostic_capture.wait_for_saves()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["03_Fig_Step_a.png", "03_Fig_Step_b.png"]
