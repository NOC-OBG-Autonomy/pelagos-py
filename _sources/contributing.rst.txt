Contributing
============

We welcome contributions to the NOC Autonomy Toolbox. To maintain code quality and stability, we use a specific workflow involving a development branch.

Development Workflow
--------------------

All new features and bug fixes should be submitted to the ``dev`` branch rather than ``main``.

1. **Fork the repository**: Create your own copy of the project on GitHub.
2. **Clone your fork**: Download the code to your local machine.
3. **Create a branch**: Work on a new branch specifically for your changes.
4. **Submit a Pull Request**: Direct your Pull Request (PR) from your fork to the ``dev`` branch of the original repository.

Code Quality and Testing
------------------------

To ensure the toolbox remains reliable, we enforce the following requirements:

* **Automated Testing**: Once a PR is submitted, automated tests will run. Code must pass these tests to be considered for the ``dev`` branch.
* **Review Process**: Code in the ``dev`` branch is further reviewed and tested before it is merged into the stable ``main`` branch.

Common Commands
---------------

**Clone the development branch**

To start working specifically on the latest development code, clone the ``dev`` branch directly:

.. code-block:: bash

   git clone -b dev https://github.com/NOC-OBG-Autonomy/pelagos-py.git

**Create a feature branch**

Always create a new branch for your work to keep your fork organised:

.. code-block:: bash

   git checkout -b my-new-feature

**Stay updated with the dev branch**

Before submitting a PR, ensure your local code is up to date with the latest changes from the official ``dev`` branch:

.. code-block:: bash

   git fetch upstream
   git merge upstream/dev

**Submitting changes**

Push your changes to your fork and then use the GitHub web interface to open a Pull Request. Ensure the **base branch** is set to ``dev``.

.. code-block:: bash

   git add .
   git commit -m "Brief description of changes"
   git push origin my-new-feature