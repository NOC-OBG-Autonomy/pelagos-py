<!-- omit in toc -->
# Contributing to pelagos-py

First off, thanks for taking the time to contribute!

All types of contributions are encouraged and valued. Please make sure to read the relevant section of this document before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all involved.

<!-- omit in toc -->
## Table of Contents

- [I Have a Question](#i-have-a-question)
- [Development Workflow](#development-workflow)
- [Reporting Bugs](#reporting-bugs)
- [Code Quality and Testing](#code-quality-and-testing)
- [Formatting and Linting](#formatting-and-linting)
- [Testing](#testing)
- [Documentation](#documentation)
- [Common Commands](#common-commands)
- [Legal Notice and Attribution](#legal-notice-and-attribution)

## I Have a Question

Before you ask a question, it is best to search for existing [Issues](https://github.com/NOC-OBG-Autonomy/pelagos-py/issues) that might help you. In case you have found a suitable issue and still need clarification, you can write your question in this issue. It is also advisable to search the internet for answers first.

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open an [Issue](https://github.com/NOC-OBG-Autonomy/pelagos-py/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project, Python version and details of your Python environment (pip/conda and versions of dependencies).

We will then take care of the issue as soon as possible.

## Development Workflow

To maintain code quality and stability, we use a workflow based on pull requests to `main`. All new features and bug fixes should be submitted to the `main` branch.

1. **Fork the repository**: Create your own copy of the project on GitHub.
2. **Clone your fork**: Download the code to your local machine.
3. **Create a branch**: Work on a new branch specifically for your changes.
4. **Submit a Pull Request**: Direct your Pull Request (PR) from your fork to the `main` branch of the original repository.

See [Common Commands](#common-commands) below for the exact git commands.

## Reporting Bugs

<!-- omit in toc -->
### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more information. Therefore, we ask you to investigate carefully, collect information and describe the issue in detail in your report. Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment components/versions.
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](https://github.com/NOC-OBG-Autonomy/pelagos-py/issues?q=label%3Abug).
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Version of Python and package manager (conda/pip etc), depending on what seems relevant.
  - Possibly your input, the output and any error messages you are seeing.
  - Can you reliably reproduce the issue?

<!-- omit in toc -->
### How Do I Submit a Good Bug Report?

We use GitHub issues to track bugs and errors. If you run into an issue with the project:

- Open an [Issue](https://github.com/NOC-OBG-Autonomy/pelagos-py/issues/new). (Since we can't be sure at this point whether it is a bug or not, we ask you not to talk about a bug yet and not to label the issue.)
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the *reproduction steps* that someone else can follow to recreate the issue on their own. This usually includes your code. For good bug reports you should isolate the problem and create a reduced test case.
- Provide the information you collected in the previous section.

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps. If there are no reproduction steps or no obvious way to reproduce the issue, the team will ask you for those steps. Bugs will not be addressed until they are reproducible.

## Code Quality and Testing

To ensure pelagos-py remains reliable, we enforce the following requirements:

- **Automated checks**: Once a PR is submitted, the tests and the formatting/lint checks run automatically. Code must pass them to be considered for merging.
- **Review Process**: A human reviewer must approve the changes before they are merged into `main`.

The workflow automatically updates the version number upon a successful PR to `main`. Any manually created release is then automatically published to PyPI.

## Formatting and Linting

Code is formatted and linted with [ruff](https://docs.astral.sh/ruff/); its settings are in `pyproject.toml`. CI checks every PR with a pinned ruff version, so run both before pushing:

```bash
ruff format .
ruff check .
```

The dashboard's JavaScript and HTML are formatted with [Prettier](https://prettier.io/) (settings in `.prettierrc.json`; CSS is left as written):

```bash
npx prettier@3.3.3 --write "src/pelagos_py/dashboard/static/**/*.{js,html}"
```

## Testing

We use [pytest](https://docs.pytest.org/) for testing. The tests live in the `tests/` directory at the root of the repository, one file per step or feature.

**Installing the test dependencies**

The testing tools (`pytest` and `pytest-cov`) are an optional dependency group in `pyproject.toml`. Install pelagos-py in editable mode with the `test` extra from the root of the repository:

```bash
pip install -e ".[test]"
```

The `-e` flag means changes to the source are picked up without reinstalling. The quotes around `".[test]"` stop some shells (such as `zsh`) treating the square brackets as a glob pattern.

**Running the tests**

From the root of the repository:

```bash
pytest                                                # full suite
pytest tests/test_impossible_date_qc.py               # one file
pytest tests/test_impossible_date_qc.py::test_dates -v  # one test, verbose
pytest --cov=pelagos_py                               # with a coverage report
```

## Documentation

**Building the docs**

The documentation is built with Sphinx from the `docs/` folder:

```bash
pip install -e .
pip install -r docs/requirements.txt
cd docs
make figures   # draws the user-guide figures (downloads the demo file the first time)
make html      # builds the site into docs/_build/html
```

The user-guide figures are not committed: `make figures` draws them from the demo file, and the docs CI does the same before publishing.

**API reference**

The API reference is generated automatically from the source by [sphinx-autoapi](https://sphinx-autoapi.readthedocs.io/), which reads the code statically (it never imports the package). The rule for what appears is simple:

- **Has a docstring → it is shown.** Any class, method or function with a docstring is added to the API reference.
- **No docstring → it is hidden.** Undocumented members (and imported names, `__init__`, etc.) are left out automatically, keeping the pages clean.

So to document something, write a docstring; to keep something out of the docs, leave its docstring off. We use NumPy-style docstrings (Parameters, Returns, Examples sections).

If a member *has* a docstring but you still want to hide it from the docs, add `:meta private:` anywhere in that docstring:

```python
def helper(self):
    """Do the thing.

    :meta private:
    """
```

> **Note:** A runtime marker such as a `@nodoc` decorator will **not** work, because sphinx-autoapi parses the source without running it and never sees attributes set at runtime. Use the docstring rule (or `:meta private:`) instead.

## Common Commands

**Clone the repository**

```bash
git clone https://github.com/NOC-OBG-Autonomy/pelagos-py.git
```

**Create a feature branch**

Always create a new branch for your work to keep your fork organised:

```bash
git checkout -b my-new-feature
```

**Stay updated with main**

Before submitting a PR, ensure your local code is up to date with the latest changes from the official `main` branch:

```bash
git fetch upstream
git merge upstream/main
```

**Submitting changes**

Push your changes to your fork and then use the GitHub web interface to open a Pull Request. Ensure the **base branch** is set to `main`.

```bash
git add .
git commit -m "Brief description of changes"
git push origin my-new-feature
```

## Legal Notice and Attribution

When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute will be provided under the project's Apache-2.0 license.

This guide is based on [contributing.md](https://contributing.md/generator).
