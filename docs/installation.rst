.. highlight:: shell

============
Installation
============

..
    Stable release
    --------------

    To install XRDPipeline, run this command in your terminal:

    .. code-block:: console

        $ pip install xrdpipeline

    This is the preferred method to install XRDPipeline, as it will always install the most recent stable release.

This software was designed for Python version 3.10.8. Set up and start a virtual environment in that version, then run:

``git clone https://github.com/AdvancedPhotonSource/XRDdatapipeline.git``

..
        The cookiecutter template applied by Miaoqi should let this be pip-installable, but it needs to be registered with pypi first.

To install dependencies, next run:

``python -m pip install -r requirements.txt``

These commands should install this module and all of its prerequisites.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

.. 
    From sources
    ------------

    The sources for XRDPipeline can be downloaded from the `Github repo`_.

    You can either clone the public repository:

    .. code-block:: console

        $ git clone git://github.com/AdvancedPhotonSource/XRDdatapipeline

    Or download the `tarball`_:

    .. code-block:: console

        $ curl -OJL https://github.com/AdvancedPhotonSource/XRDdatapipeline/tarball/master

    Once you have a copy of the source, you can install it with:

    .. code-block:: console

        $ python setup.py install


    .. _Github repo: https://github.com/AdvancedPhotonSource/XRDdatapipeline
    .. _tarball: https://github.com/AdvancedPhotonSource/XRDdatapipeline/tarball/master
