Development
===========

We welcome contributions to pelagos_py. To maintain code quality and stability, we use a specific workflow involving pull requests to ``main``.

If you have a bug report, feature request, or question, please open an issue on the `GitHub issues page <https://github.com/NOC-OBG-Autonomy/pelagos-py/issues>`_ — this is also a great way to contribute.

Development Workflow
--------------------

All new features and bug fixes should be submitted to the ``main`` branch.

1. **Fork the repository**: Create your own copy of the project on GitHub.
2. **Clone your fork**: Download the code to your local machine.
3. **Create a branch**: Work on a new branch specifically for your changes.
4. **Submit a Pull Request**: Direct your Pull Request (PR) from your fork to the ``main`` branch of the original repository.

Code Quality and Testing
------------------------

To ensure pelagos_py remains reliable, we enforce the following requirements:

* **Automated Testing**: Once a PR is submitted, automated tests will run. Code must pass these tests to be considered for merging.
* **Review Process**: A human reviewer must approve the changes before they are merged into ``main``.

Testing
-------

We use `pytest <https://docs.pytest.org/>`_ for testing. The tests live in the
``tests/`` directory at the root of the repository.

**Installing the test dependencies**

The testing tools (``pytest`` and ``pytest-cov``) are declared as an optional
dependency group in ``pyproject.toml`` rather than being installed by default.
To install pelagos_py together with everything needed to run the tests, do an
*editable* install with the ``test`` extra from the root of the repository:

.. code-block:: bash

   pip install -e ".[test]"

The ``-e`` flag installs the project in editable mode, so changes you make to
the source are picked up without reinstalling. The ``[test]`` extra pulls in the
testing dependencies. The quotes around ``".[test]"`` are required by some
shells (such as ``zsh``) to stop the square brackets being interpreted as a
glob pattern.

**Running the full test suite**

From the root of the repository, simply run:

.. code-block:: bash

   pytest

pytest discovers the configuration in ``pyproject.toml`` and collects every test
under ``tests/`` automatically. You should see output similar to:

.. code-block:: text

   ========================= test session starts =========================
   platform darwin -- Python 3.14.3, pytest-9.0.3, pluggy-1.6.0
   rootdir: /path/to/pelagos-py
   configfile: pyproject.toml
   collected 41 items

   .........................................                       [100%]

   ========================= 41 passed in 5.87s ==========================

**Running a specific test**

To run a single test file, pass its path to pytest:

.. code-block:: bash

   pytest tests/test_impossible_date_qc.py

You can narrow it down further to a single test function (or a single
parametrised case) using the ``::`` separator, and add ``-v`` for verbose
output that lists each test by name:

.. code-block:: bash

   pytest tests/test_impossible_date_qc.py::test_dates -v

For example:

.. code-block:: text

   ========================= test session starts =========================
   collected 3 items

   tests/test_impossible_date_qc.py::test_dates[times0-expected_flags0] PASSED [ 33%]
   tests/test_impossible_date_qc.py::test_dates[times1-expected_flags1] PASSED [ 66%]
   tests/test_impossible_date_qc.py::test_dates[times2-expected_flags2] PASSED [100%]

   ========================== 3 passed in 4.48s ==========================

**Measuring coverage**

Because ``pytest-cov`` is installed with the ``test`` extra, you can also produce
a coverage report for the ``pelagos_py`` package:

.. code-block:: bash

   pytest --cov=pelagos_py

Documentation
-------------

The API reference is generated automatically from the source by
`sphinx-autoapi <https://sphinx-autoapi.readthedocs.io/>`_, which reads the
code statically (it never imports the package). The rule for what appears is
simple:

* **Has a docstring → it is shown.** Any class, method or function with a
  docstring is added to the API reference.
* **No docstring → it is hidden.** Undocumented members (and imported names,
  ``__init__``, etc.) are left out automatically, keeping the pages clean.

So to document something, write a docstring; to keep something out of the docs,
leave its docstring off. We use NumPy-style docstrings (Parameters, Returns,
Examples sections).

If a member *has* a docstring but you still want to hide it from the docs, add
``:meta private:`` anywhere in that docstring:

.. code-block:: python

   def helper(self):
       """Do the thing.

       :meta private:
       """

.. note::

   A runtime marker such as a ``@nodoc`` decorator will **not** work, because
   sphinx-autoapi parses the source without running it and never sees
   attributes set at runtime. Use the docstring rule (or ``:meta private:``)
   instead.

Common Commands
---------------

**Clone the repository**

.. code-block:: bash

   git clone https://github.com/NOC-OBG-Autonomy/pelagos-py.git

**Create a feature branch**

Always create a new branch for your work to keep your fork organised:

.. code-block:: bash

   git checkout -b my-new-feature

**Stay updated with main**

Before submitting a PR, ensure your local code is up to date with the latest changes from the official ``main`` branch:

.. code-block:: bash

   git fetch upstream
   git merge upstream/main

**Submitting changes**

Push your changes to your fork and then use the GitHub web interface to open a Pull Request. Ensure the **base branch** is set to ``main``.

.. code-block:: bash

   git add .
   git commit -m "Brief description of changes"
   git push origin my-new-feature
