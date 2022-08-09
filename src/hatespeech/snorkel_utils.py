"""Snorkel utility functions."""

import itertools as it
from functools import partial
from typing import List, Tuple, Union

import more_itertools as mit
import numpy as np
import pandas as pd
from snorkel.labeling.apply.core import ApplierMetadata, BaseLFApplier
from snorkel.labeling.apply.pandas import rows_to_triplets
from snorkel.labeling.lf import LabelingFunction
from tqdm.auto import tqdm


class ImprovedPandasLFApplier(BaseLFApplier):
    """LF applier for a Pandas DataFrame.

    Examples:
        >>> from snorkel.labeling import labeling_function
        >>> @labeling_function()
        ... def is_big_num(x):
        ...     return 1 if x.num > 42 else 0
        >>> applier = PandasLFApplier([is_big_num])
        >>> applier.apply(pd.DataFrame(dict(num=[10, 100], text=["hello", "hi"])))
        array([[0], [1]])
    """

    def apply(
        self,
        df: pd.DataFrame,
        batch_size: int = 128,
        progress_bar: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ApplierMetadata]]:
        """Label Pandas DataFrame of data points with LFs.

        Args:
            df (Pandas DataFrame):
                Pandas DataFrame containing data points to be labeled by LFs
            batch_size (int, optional):
                Number of data points to label at once. Defaults to 1.
            progress_bar (bool, optional)
                Display a progress bar. Defaults to True.

        Returns:
            NumPy array.
                The NumPy array is a matrix of labels emitted by LFs.
        """
        # Initialise the function which is to be applied to the data
        apply_fn = partial(batched_apply_lfs_to_data_point, lfs=self._lfs)

        # Split up the dataframe into batches
        batches = mit.chunked(df, n=batch_size)
        num_batches = len(df) // batch_size
        if len(df) % batch_size != 0:
            num_batches += 1
        # batches = np.split(df, np.arange(batch_size, len(df), batch_size))

        # Apply the function to the dataframe
        itr = batches
        if progress_bar:
            itr = tqdm(itr, total=num_batches, desc="Applying LFs to batches")
        labels = list(it.chain(*[apply_fn(batch) for batch in itr]))

        # Add row indices to the labels
        triplets = rows_to_triplets(labels)

        # Convert the labels to a NumPy array
        label_arr = self._numpy_from_row_data(triplets)

        # Return the labels
        return label_arr


def batched_apply_lfs_to_data_point(
    batch: Union[pd.DataFrame, pd.Series],
    lfs: List[LabelingFunction],
) -> List[List[Tuple[int, int]]]:
    """Label a batch of data points with a set of LFs.

    Args:
        batch (Pandas DataFrame or Series):
            Data points to label.
        lfs (list of LabelingFunction):
            Set of LFs to label ``x`` with.

    Returns:
        list of list of pairs of int:
            A list for every row in `batch`, each of which consisting of (LF index,
            label) tuples.
    """
    # Apply all the labelling functions to the batch
    label_values = {lf_idx: lf(batch) for lf_idx, lf in enumerate(lfs)}

    # Organise the labels into a list of lists of pairs of int
    return [
        [(lf_idx, label_value[row_idx]) for lf_idx, label_value in label_values.items()]
        for row_idx in range(len(batch))
    ]
