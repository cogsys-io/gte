#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Top-level package for Ground Truth Establisher (GTE)."""

from .gte import GTE  # noqa: F401
from .aux_log import Log0  # noqa: F401
from .aux_pandas import df9  # noqa: F401
from .aux_pandas import ddf  # noqa: F401
from .aux_pandas import rdf  # noqa: F401

# fmt: off
from . import _version
__version__ = _version.get_versions()['version']
__version_dict__ = _version.get_versions()
# fmt: on

__author__ = """cogsys.io"""
__email__ = "cogsys@cogsys.io"


def get_module_version():
    """Get module version."""
    return __version__


# end
