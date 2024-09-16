#!/usr/bin/env python3

import numpy as np
import pandas as pd
from contextlib import ExitStack
from IPython.core.display import display

from sklearn.datasets import load_wine

ds9 = load_wine()
df9 = pd.DataFrame(
    data=np.c_[ds9["data"], ds9["target"]],
    columns=ds9["feature_names"] + ["target"],
)


def _context_pandas(
    max_columns=222,
    max_colwidth=44,
    width=2222,
    max_rows=44,
    min_rows=33,
):
    """Apply custom context to dataframe representation (ExitStack)."""
    return [
        pd.option_context("display.max_columns", max_columns),
        pd.option_context("display.max_colwidth", max_colwidth),
        pd.option_context("display.width", width),
        pd.option_context("display.max_rows", max_rows),
        pd.option_context("display.min_rows", min_rows),
    ]


def ddf(df0, **opt):
    """Display DF using custom formatting context.

    Examples
    --------
    >>> import pandas as pd
    >>> from gte import ddf, df9
    >>> ddf(df9)

    """
    with ExitStack() as stack:
        _ = [stack.enter_context(cont) for cont in _context_pandas(**opt)]
        display(df0)


def rdf(df0, **opt):
    """Get DF repr using custom formatting context.

    Examples
    --------
    >>> import pandas as pd
    >>> from gte import rdf, df9
    >>> print(rdf(df9))

    """
    with ExitStack() as stack:
        _ = [stack.enter_context(cont) for cont in _context_pandas(**opt)]
        return str(df0)
