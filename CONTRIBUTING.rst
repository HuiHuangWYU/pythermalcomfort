============
Contributing
============

Contributions are welcome and greatly appreciated!
Every bit helps, and credit will always be given.

Bug Reports
===========

When `reporting a bug <https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/issues>`_, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug (minimal reproducible example preferred).
* Relevant error messages and a shortened traceback.

Documentation Improvements
==========================

pythermalcomfort can always use more documentation: docs, docstrings, examples,
and tutorials are all valuable.

Issues, Features and Feedback
==============================

The best way to send feedback is to open an `issue <https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/issues>`_.

If you are proposing a feature:

* Explain in detail how it would work and why it's useful.
* Keep the scope narrow so it is easier to review and implement.
* Consider opening a discussion issue first for larger changes.

Contributing Code
=================

Setup
-----

Fork the repository on GitHub and clone your fork locally:

.. code-block:: bash

    git clone git@github.com:your-username/pythermalcomfort.git
    cd pythermalcomfort
    git remote add upstream git@github.com:CenterForTheBuiltEnvironment/pythermalcomfort.git
    git fetch upstream

Create a feature branch:

.. code-block:: bash

    git checkout -b Feature/short-description   # new feature
    git checkout -b Fix/short-description        # bug fix
    git checkout -b Documentation/doc-name       # docs only

Branch naming
-------------

* ``Feature/your-feature-name``
* ``Fix/your-bug-name``
* ``Documentation/doc-name``

Quality checks
--------------

Run these before every commit and before opening a PR:

.. code-block:: bash

    # format
    ruff format ./pythermalcomfort ./tests

    # lint with auto-fix
    ruff check --fix ./pythermalcomfort ./tests

    # format docstrings
    docformatter -r -i --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort

    # run full test suite
    pytest tests/

    # or via tox (tests Python 3.10–3.13)
    tox

Pull request checklist
-----------------------

* Clear summary of the change and motivation.
* Tests for new behaviour and updates for any affected tests.
* Documentation updates (docstrings or ``docs/``).
* CHANGELOG entry (if applicable).
* Add yourself to ``AUTHORS.rst`` (optional).
* All CI checks pass.

Developer guides
================

Detailed step-by-step guides are available for the two main contribution types:

* **Adding a thermal comfort function** — see :doc:`functions`
* **Adding a Matplotlib plot class** — see :doc:`plots`

If you are unsure which guide applies, open an issue first.

Tips
====

* Open an issue before starting larger features to discuss scope and design.
* Keep PRs focused and small where possible.
* Include tests and documentation for all public API changes.

License
=======

pythermalcomfort is released under the MIT License.
