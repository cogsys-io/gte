#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ground Truth Establisher (GTE) Module.

Great stuff!

"""

import random
import numpy as np
import pandas as pd
import mne
from mne.simulation import SourceSimulator
from mne.beamformer import Beamformer

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Callable
from typing import get_origin, get_args

from .aux_log import Log0

logZ = Log0()
log0 = logZ.logger


@dataclass
class GTE:
    """
    Ground Truth Establisher (GTE) class for data gathering and simulations.

    This class is designed to handle data gathering and perform simulations,
    primarily using MNE-Python. It provides functionality to manage subject
    directories, individual subject data, and MNE objects for simulations.

    Attributes
    ----------
    _subjects_dir : Path or None
        Private attribute to store the subjects directory path.
    _subject : str or None
        Private attribute to store the current subject identifier.
    _info0 : mne.Info or None
        Private attribute to store the MNE Info object.
    _montage0 : mne.channels.DigMontage or None
        Private attribute to store the MNE DigMontage object.
    _trans0 : mne.transforms.Transform or None
        Private attribute to store the MNE Transform object.
    _src0 : mne.SourceSpaces or None
        Private attribute to store the MNE SourceSpaces object.
    _bem_model0 : list of mne.bem.ConductorModel or None
        Private attribute to store the MNE BEM model.
    _bem_solution0 : mne.bem.ConductorModel or None
        Private attribute to store the MNE BEM solution.
    _fwd0 : mne.Forward or None
        Private attribute to store the MNE Forward solution.
    _annot0 : str or None
        Private attribute to store the FreeSurfer's annot (surface labels).
    _labels0 : list of mne.Label
        FreeSurfer parcellation labels
    _label0_names : list of str
        FreeSurfer parcellation label names
    _singularity_events : np.ndarray
        Singularity events data. TODO Detailed description.
    _singularity_event_IDs : Dict[str, int]
        Singularity events data. TODO Detailed description.
    _singularity_events_desc : Dict[int, str]
        Singularity events data. TODO Detailed description.
    _singularity_events_df : pd.DataFrame
        Singularity events data. TODO Detailed description.
    _experimental_events : np.ndarray
        Experimental events data. TODO Detailed description.
    _experimental_event_IDs : Dict[str, int]
        Experimental events data. TODO Detailed description.
    _experimental_events_desc : Dict[int, str]
        Experimental events data. TODO Detailed description.
    _experimental_events_df : pd.DataFrame
        Experimental events data. TODO Detailed description.

    Properties
    ----------
    subjects_dir : Path or None
        Path to the subjects directory.
    subject : str or None
        Identifier for the current subject.
    subject_dir : Path or None
        Path to the current subject's directory.
    info0 : mne.Info or None
        MNE Info object loaded from a .fif file.
    montage0 : mne.channels.DigMontage or None
        MNE DigMontage object loaded from a -montage.fif file.
    trans0 : mne.transforms.Transform or None
        MNE Transform object loaded from a -trans.fif file.
    src0 : mne.SourceSpaces or None
        MNE SourceSpaces object loaded from a -src.fif file.
    bem_model0 : list of mne.bem.ConductorModel or None
        MNE BEM model loaded from a -bem-model.fif file.
    bem_solution0 : mne.bem.ConductorModel or None
        MNE BEM solution loaded from a -bem-solution.fif file.
    fwd0 : mne.Forward or None
        MNE Forward solution loaded from a -fwd.fif file.
    annot0 : str or None
        FreeSurfer's annot (surface labels).
    labels0 : list of mne.Label
        FreeSurfer parcellation labels.
    label0_names : list of str
        FreeSurfer parcellation label names.
    labels2 : list of mne.Label
        Vertices selected as activity/noise sources.
    label2_names : list of str
        Label names for vertices selected as activity/noise sources.
    singularity_events : np.ndarray
        Singularity events data. TODO Detailed description.
    singularity_event_IDs : Dict[str, int]
        Singularity events data. TODO Detailed description.
    singularity_events_desc : Dict[int, str]
        Singularity events data. TODO Detailed description.
    singularity_events_df : pd.DataFrame
        Singularity events data. TODO Detailed description.
    experimental_events : np.ndarray
        Experimental events data. TODO Detailed description.
    experimental_event_IDs : Dict[str, int]
        Experimental events data. TODO Detailed description.
    experimental_events_desc : Dict[int, str]
        Experimental events data. TODO Detailed description.
    experimental_events_df : pd.DataFrame
        Experimental events data. TODO Detailed description.


    """

    _subjects_dir: Optional[Path] = field(default=None, init=False)
    _subject: Optional[str] = field(default=None, init=False)
    _info0: Optional[mne.Info] = field(default=None, init=False)
    _montage0: Optional[mne.channels.DigMontage] = field(default=None, init=False)
    _trans0: Optional[mne.transforms.Transform] = field(default=None, init=False)
    _src0: Optional[mne.SourceSpaces] = field(default=None, init=False)
    _bem_model0: Optional[List[Dict]] = field(default=None, init=False)

    _bem_solution0: Optional[mne.bem.ConductorModel] = field(default=None, init=False)
    _fwd0: Optional[mne.Forward] = field(default=None, init=False)
    _genuine_noise_cov0: Optional[mne.Covariance] = field(default=None, init=False)

    _annot0: Optional[str] = field(default=None, init=False)
    _labels0: Optional[List[mne.Label]] = field(default=None, init=False)
    _labels2: Optional[List[mne.Label]] = field(default=None, init=False)
    _labels3: Optional[List[mne.Label]] = field(default=None, init=False)
    _labels4: Optional[List[mne.Label]] = field(default=None, init=False)
    _label0_names: Optional[List[str]] = field(default=None, init=False)
    _label2_names: Optional[List[str]] = field(default=None, init=False)
    _label3_names: Optional[List[str]] = field(default=None, init=False)
    _label4_names: Optional[List[str]] = field(default=None, init=False)

    _singularity_events: Optional[np.ndarray] = field(default=None, init=False)
    _singularity_event_IDs: Optional[Dict[str, int]] = field(default=None, init=False)
    _singularity_events_desc: Optional[Dict[int, str]] = field(default=None, init=False)
    _singularity_events_df: Optional[pd.DataFrame] = field(default=None, init=False)

    _experimental_events: Optional[np.ndarray] = field(default=None, init=False)
    _experimental_event_IDs: Optional[Dict[str, int]] = field(default=None, init=False)
    _experimental_events_desc: Optional[Dict[int, str]] = field(
        default=None, init=False
    )
    _experimental_events_df: Optional[pd.DataFrame] = field(default=None, init=False)
    _activ0: Optional[Dict[str, Dict[str, Dict[str, float]]]] = field(
        default=None, init=False
    )
    _activ0_labels: Optional[List[str]] = field(default=None, init=False)
    _activ0_events: Optional[List[str]] = field(default=None, init=False)
    _activ0_trial_samp_total: Optional[int] = field(default=None, init=False)
    _activ0_trial_times: Optional[np.ndarray] = field(default=None, init=False)
    _source_simulator: Optional[SourceSimulator] = field(default=None, init=False)
    _activ0_stc: Optional[mne.SourceEstimate] = field(default=None, init=False)
    _activ0_raw: Optional[mne.io.Raw] = field(default=None, init=False)
    _activ2_raw: Optional[mne.io.Raw] = field(default=None, init=False)
    _activ2_epochs: Optional[mne.Epochs] = field(default=None, init=False)
    _activ2_evoked: Optional[Dict[str, mne.Evoked]] = field(default=None, init=False)
    _activ2_data_cov: Optional[mne.Covariance] = field(default=None, init=False)
    _activ2_noise_cov: Optional[mne.Covariance] = field(default=None, init=False)
    _activ2_common_cov: Optional[mne.Covariance] = field(default=None, init=False)
    _bf_filters: Optional[Dict[str, Beamformer]] = field(default=None, init=False)
    _stcs: Optional[Dict[str, mne.SourceEstimate]] = field(default=None, init=False)

    def __post_init__(self):
        """Post-initialization method to set up default values if needed."""
        pass

    @staticmethod
    def _is_valid_dir(path: Path) -> bool:
        """
        Check if a path is a valid directory or a symlink to a valid directory.

        Parameters
        ----------
        path : Path
            The path to check.

        Returns
        -------
        bool
            True if the path is a valid directory or a symlink to a valid directory,
            False otherwise.
        """
        return path.exists() and (
            path.is_dir() or (path.is_symlink() and path.resolve().is_dir())
        )

    @property
    def subjects_dir(self) -> Optional[Path]:
        """
        Get the subjects directory path.

        Returns
        -------
        Path or None
            The path to the subjects directory if set, otherwise None.
        """
        return self._subjects_dir

    @subjects_dir.setter
    def subjects_dir(self, value: Optional[Path]):
        """
        Set the subjects directory path.

        Parameters
        ----------
        value : Path or None
            The path to set as the subjects directory. If None, the subjects_dir
            will be set to None.

        Raises
        ------
        ValueError
            If the provided path does not exist, is not a directory, or is not
            a symlink to a directory.
        """
        if value is not None:
            path = Path(value).expanduser().resolve()
            if not self._is_valid_dir(path):
                raise ValueError(
                    f"The provided subjects_dir '{path}' does not exist, is not a directory, or is not a symlink to a directory."
                )
            self._subjects_dir = path
        else:
            self._subjects_dir = None

    @property
    def subject(self) -> Optional[str]:
        """
        Get the current subject identifier.

        Returns
        -------
        str or None
            The identifier of the current subject if set, otherwise None.
        """
        return self._subject

    @subject.setter
    def subject(self, value: Optional[str]):
        """
        Set the current subject identifier.

        Parameters
        ----------
        value : str or None
            The identifier to set as the current subject. If None, the current
            subject will be unset.

        Raises
        ------
        ValueError
            If subjects_dir is not set when trying to set a subject, or if the
            subject directory does not exist, is not a directory, or is not a
            symlink to a directory under subjects_dir.
        """
        if value is not None:
            if self._subjects_dir is None:
                raise ValueError("subjects_dir must be set before setting a subject")
            subject_path = (self._subjects_dir / value).resolve()
            if not self._is_valid_dir(subject_path):
                raise ValueError(
                    f"The subject directory '{subject_path}' does not exist, is not a directory, or is not a symlink to a directory."
                )
        self._subject = value

    @property
    def subject_dir(self) -> Optional[Path]:
        """
        Get the directory path for the current subject.

        Returns
        -------
        Path or None
            The path to the current subject's directory if both subjects_dir
            and subject are set, otherwise None.
        """
        if self._subjects_dir is not None and self._subject is not None:
            return (self._subjects_dir / self._subject).resolve()
        return None

    @property
    def info0(self) -> Optional[mne.Info]:
        """
        Get the MNE Info object.

        Returns
        -------
        mne.Info or None
            The MNE Info object if loaded, otherwise None.
        """
        return self._info0

    @info0.setter
    def info0(self, fif_file: Union[str, Path]):
        """
        Load MNE Info from a .fif file and set it as the current info.

        Parameters
        ----------
        fif_file : str or Path
            Path to the .fif file to load the MNE Info from.

        Raises
        ------
        ValueError
            If the file does not exist or is not a .fif file.
        RuntimeError
            If there's an error while reading the .fif file.
        """
        fif_path = self._get_mne_file_path(fif_file, "-info.fif")
        try:
            self._info0 = mne.io.read_info(fif_path)
        except Exception as e:
            raise RuntimeError(f"Error reading .fif file: {str(e)}")

    @property
    def montage0(self) -> Optional[mne.channels.DigMontage]:
        """
        Get the MNE DigMontage object.

        Returns
        -------
        mne.channels.DigMontage or None
            The MNE DigMontage object if loaded, otherwise None.
        """
        return self._montage0

    @montage0.setter
    def montage0(self, fif_file: Union[str, Path]):
        """
        Load MNE DigMontage from a -montage.fif file and set it as the current montage.

        Parameters
        ----------
        fif_file : str or Path
            Path to the -montage.fif file to load the MNE DigMontage from.

        Raises
        ------
        ValueError
            If the file does not exist or is not a -montage.fif file.
        RuntimeError
            If there's an error while reading the .fif file.
        """
        fif_path = self._get_mne_file_path(fif_file, "-montage.fif")
        try:
            self._montage0 = mne.channels.read_dig_fif(fif_path)
        except Exception as e:
            raise RuntimeError(f"Error reading -montage.fif file: {str(e)}")

    @property
    def trans0(self) -> Optional[mne.transforms.Transform]:
        """
        Get the MNE Transform object.

        Returns
        -------
        mne.transforms.Transform or None
            The MNE Transform object if loaded, otherwise None.
        """
        return self._trans0

    @trans0.setter
    def trans0(self, fif_file: Union[str, Path]):
        """
        Load MNE Transform from a -trans.fif file and set it as the current trans.

        Parameters
        ----------
        fif_file : str or Path
            Path to the -trans.fif file to load the MNE Transform from.

        Raises
        ------
        ValueError
            If the file does not exist or is not a .fif file.
        RuntimeError
            If there's an error while reading the .fif file.
        """
        fif_path = self._get_mne_file_path(fif_file, "-trans.fif")
        try:
            self._trans0 = mne.read_trans(fif_path)
        except Exception as e:
            raise RuntimeError(f"Error reading -trans.fif file: {str(e)}")

    @property
    def src0(self) -> Optional[mne.SourceSpaces]:
        """
        Get the MNE SourceSpaces object.

        Returns
        -------
        mne.SourceSpaces or None
            The MNE SourceSpaces object if loaded, otherwise None.
        """
        return self._src0

    @src0.setter
    def src0(self, fif_file: Union[str, Path]):
        """
        Load MNE SourceSpaces from a -src.fif file and set it as the current src.

        Parameters
        ----------
        fif_file : str or Path
            Path to the -src.fif file to load the MNE SourceSpaces from.

        Raises
        ------
        ValueError
            If the file does not exist or is not a .fif file.
        RuntimeError
            If there's an error while reading the .fif file.
        """
        fif_path = self._get_mne_file_path(fif_file, "-src.fif")
        try:
            self._src0 = mne.read_source_spaces(fif_path)
        except Exception as e:
            raise RuntimeError(f"Error reading -src.fif file: {str(e)}")

    @property
    def bem_model0(self) -> Optional[List[Dict]]:
        """
        Get the MNE BEM model.

        Returns
        -------
        list of mne.bem.ConductorModel or None
            The MNE BEM model if loaded, otherwise None.
        """
        return self._bem_model0

    @bem_model0.setter
    def bem_model0(self, fif_file: Union[str, Path]):
        """
        Load MNE BEM model from a -bem-model.fif file and set it as the current bem_model.

        Parameters
        ----------
        fif_file : str or Path
            Path to the -bem-model.fif file to load the MNE BEM model from.

        Raises
        ------
        ValueError
            If the file does not exist or is not a .fif file.
        RuntimeError
            If there's an error while reading the .fif file.
        """
        fif_path = self._get_mne_file_path(fif_file, "-bem-model.fif")
        try:
            self._bem_model0 = mne.read_bem_surfaces(fif_path)
        except Exception as e:
            raise RuntimeError(f"Error reading -bem-model.fif file: {str(e)}")

    @property
    def bem_solution0(self) -> Optional[mne.bem.ConductorModel]:
        """
        Get the MNE BEM solution.

        Returns
        -------
        mne.bem.ConductorModel or None
            The MNE BEM solution if loaded, otherwise None.
        """
        return self._bem_solution0

    @bem_solution0.setter
    def bem_solution0(self, fif_file: Union[str, Path]):
        """
        Load MNE BEM solution from a -bem-solution.fif file and set it as the current bem_solution.

        Parameters
        ----------
        fif_file : str or Path
            Path to the -bem-solution.fif file to load the MNE BEM solution from.

        Raises
        ------
        ValueError
            If the file does not exist or is not a .fif file.
        RuntimeError
            If there's an error while reading the .fif file.
        """
        fif_path = self._get_mne_file_path(fif_file, "-bem-solution.fif")
        try:
            self._bem_solution0 = mne.read_bem_solution(fif_path)
        except Exception as e:
            raise RuntimeError(f"Error reading -bem-solution.fif file: {str(e)}")

    @property
    def fwd0(self) -> Optional[mne.Forward]:
        """
        Get the MNE Forward solution.

        Returns
        -------
        mne.Forward or None
            The MNE Forward solution if loaded, otherwise None.
        """
        return self._fwd0

    @fwd0.setter
    def fwd0(self, fif_file: Union[str, Path]):
        """
        Load MNE Forward solution from a -fwd.fif file and set it as the current fwd.

        Parameters
        ----------
        fif_file : str or Path
            Path to the -fwd.fif file to load the MNE Forward solution from.

        Raises
        ------
        ValueError
            If the file does not exist or is not a .fif file.
        RuntimeError
            If there's an error while reading the .fif file.
        """
        fif_path = self._get_mne_file_path(fif_file, "-fwd.fif")
        try:
            self._fwd0 = mne.read_forward_solution(fif_path)
        except Exception as e:
            raise RuntimeError(f"Error reading -fwd.fif file: {str(e)}")

    @property
    def genuine_noise_cov0(self) -> Optional[mne.Covariance]:
        """
        Get the real noise covariance matrix.

        Returns
        -------
        mne.Covariance or None
            The MNE Covariance object representing the real noise covariance matrix if loaded,
            otherwise None.

        See Also
        --------
        genuine_noise_cov0.setter : Method to load the real noise covariance matrix from a file.
        """
        return self._genuine_noise_cov0

    @genuine_noise_cov0.setter
    def genuine_noise_cov0(self, fif_file: Union[str, Path]):
        """
        Load the real noise covariance matrix.

        Load the real noise covariance matrix from a .fif file and set it as the current
        real noise covariance.

        Parameters
        ----------
        fif_file : str or Path
            Path to the .fif file to load the real noise covariance matrix from.
            The file name should follow the pattern:
            "<subject>-<timestamp>-real-noise-cov.fif"
            e.g., "phantomica-20240913T000907-real-noise-cov.fif"

        Raises
        ------
        ValueError
            If the file does not exist or does not match the expected naming pattern.
        RuntimeError
            If there's an error while reading the .fif file.

        Notes
        -----
        This method uses the `_get_mne_file_path` helper method to resolve the file path
        and ensure it exists and has the correct suffix.
        """
        fif_path = self._get_mne_file_path(fif_file, "-noise-cov.fif")
        try:
            self._genuine_noise_cov0 = mne.read_cov(fif_path)
            log0.info(f"Real noise covariance matrix loaded from {fif_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading real noise covariance file: {str(e)}")

    @property
    def annot0(self) -> Optional[str]:
        """
        Get the annotation string.

        Returns
        -------
        str
            The current annotation string.
        """
        return self._annot0

    @annot0.setter
    def annot0(self, value: str):
        """
        Set the annotation string.

        Parameters
        ----------
        value : str
            The annotation string to set.
        """
        self._annot0 = value

    @property
    def labels0(self) -> Optional[List[mne.Label]]:
        """
        Get the annotation labels.

        Returns
        -------
        Optional[List[mne.Label]]
            The current annotation labels, or None if not set.
        """
        return self._labels0

    @property
    def label0_names(self) -> Optional[List[str]]:
        """
        Get the annotation label names.

        Returns
        -------
        Optional[List[mne.Label]]
            The current annotation labels, or None if not set.
        """
        return self._label0_names

    @property
    def labels2(self) -> Optional[List[mne.Label]]:
        """
        Get the processed annotation labels.

        Returns
        -------
        Optional[List[mne.Label]]
            The processed annotation labels, or None if not set.
        """
        return self._labels2

    @property
    def label2_names(self) -> Optional[List[str]]:
        """
        Get the annotation label names.

        Returns
        -------
        Optional[List[mne.Label]]
            The current annotation labels, or None if not set.
        """
        return self._label2_names

    @property
    def labels3(self) -> Optional[List[mne.Label]]:
        """
        Get the annotation labels.

        Returns
        -------
        Optional[List[mne.Label]]
            The current annotation labels, or None if not set.
        """
        return self._labels3

    @property
    def label3_names(self) -> Optional[List[str]]:
        """
        Get the annotation label names.

        Returns
        -------
        Optional[List[mne.Label]]
            The current annotation labels, or None if not set.
        """
        return self._label3_names

    @property
    def labels4(self) -> Optional[List[mne.Label]]:
        """
        Get the annotation labels.

        Returns
        -------
        Optional[List[mne.Label]]
            The current annotation labels, or None if not set.
        """
        return self._labels4

    @property
    def label4_names(self) -> Optional[List[str]]:
        """
        Get the annotation label names.

        Returns
        -------
        Optional[List[mne.Label]]
            The current annotation labels, or None if not set.
        """
        return self._label4_names

    def read_labels_from_annot(
        self, regexp: Optional[str] = None, sort: bool = False, verbose: bool = False
    ) -> None:
        """
        Read annotation labels from FreeSurfer parcellation based on the GTE.annot0 property.

        Parameters
        ----------
        regexp : Optional[str], default None
            Regular expression to filter labels.
        sort : bool, default False
            If True, sort the labels by name.
        verbose : bool, default False
            If True, print additional information.

        Raises
        ------
        ValueError
            If subjects_dir, subject, or annot0 is not set.
        """
        if self.subjects_dir is None or self.subject is None or not self.annot0:
            raise ValueError(
                "subjects_dir, subject, and annot0 must be set before reading labels."
            )

        try:
            self._labels0 = mne.read_labels_from_annot(
                subject=self.subject,
                parc=self.annot0,
                subjects_dir=self.subjects_dir,
                regexp=regexp,
                sort=sort,
                verbose=verbose,
            )
            self._label0_names = [label.name for label in self._labels0]
            log0.info(
                f"Successfully acquired {len(self._labels0)} labels from annotation {self.annot0}"
            )
        except Exception as e:
            log0.error(f"Error reading labels from annotation: {str(e)}")
            raise

    def process_labels0(
        self, location: str = "center", extent: float = 0.0, verbose: bool = False
    ) -> None:
        """
        Process labels0 to create labels2 based on specified parameters.

        Parameters
        ----------
        location : str, default "center"
            The location within each label to select. Options: "center" or "random".
        extent : float, default 5.0
            The extent of the selection in mm.
        verbose : bool, default False
            If True, print progress information.

        Raises
        ------
        ValueError
            If labels0 is not set or if location is invalid.
        """
        if self._labels0 is None:
            raise ValueError("labels0 must be set before processing labels.")

        if location not in ["center", "random"]:
            raise ValueError("location must be either 'center' or 'random'.")

        self._labels2 = []
        total = len(self._labels0)
        leadz = len(str(total))

        for idx0, label0 in enumerate(self._labels0):
            # Using mne.label.select_sources to get labels
            # containing limited/restricted number of vertices
            label2 = mne.label.select_sources(
                subject=self.subject,
                label=label0,
                location=location,
                extent=extent,
                subjects_dir=self.subjects_dir,
            )
            # CONSIDER: in labels4 we use name property so changing it here is
            # currently not recommended, in future we may consider
            # implementation based on "".endswith() "".startswith() or
            # "".contains(), but it may also be more error prone
            #
            # label2.name = f"source-label-{location}-{label0.name}"
            label2.name = label0.name

            if verbose:
                print(
                    f"[{idx0+1:0{leadz}d}/{total:0{leadz}d}] "
                    f"Processing: {label0.name} [{len(label2)}/{len(label0)}]"
                )

            self._labels2.append(label2)

        self._label2_names = [label.name for label in self._labels2]

        log0.info(
            f"Successfully processed {len(self._labels2)} labels "
            f"with location '{location}' and extent {extent}"
        )

    def make_dummy_events(
        self,
        event_labels: Union[int, List[str]],
        event_repets: int,
        event_interv: int,
        event_begins: int,
    ) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str], pd.DataFrame]:
        """
        Generate dummy events for simulation or testing purposes.

        Parameters
        ----------
        event_labels : int or list of str
            If int, number of unique events to generate. Event labels will be created
            automatically as "Ev01", "Ev02", etc.
            If list of str, custom labels for events. Must be unique.
        event_repets : int
            Number of repetitions for each event.
        event_interv : int
            Interval between events in samples.
        event_begins : int
            Sample number at which the first event begins.

        Returns
        -------
        events : numpy.ndarray
            2D array with columns:
            - Event onset (in samples)
            - Signal value of the immediately preceding sample
            - Event code
        event_IDs : dict
            Mapping of event labels to their corresponding values.
        event_desc : dict
            Mapping of event values to their corresponding labels.
        events_df : pandas.DataFrame
            DataFrame containing event information with columns:
            - sample_num: Event onset (in samples)
            - preceding_val: Signal value of the immediately preceding sample
            - event_code: Numeric code for the event
            - event_labels: String label for the event

        Notes
        -----
        The method generates dummy events that can be used for simulating
        experiment data or testing event-related functionalities. The events
        are randomly shuffled to simulate a realistic experimental scenario.

        The 'preceding_val' column is set to zero for all events, simulating
        a scenario where events are detected at the rising edge of a trigger signal.

        Raises
        ------
        TypeError
            If event_labels is neither an integer nor a list of strings.
        AssertionError
            If event_labels is a list of strings with non-unique elements.
        """
        if isinstance(event_labels, int):
            num_events = event_labels
            leadz = len(str(num_events + 1)) + 1
            event_labels = [f"Ev{ii:0{leadz}d}" for ii in range(1, num_events + 1)]
        elif isinstance(event_labels, list):
            if all(isinstance(item, str) for item in event_labels):
                if len(event_labels) != len(set(event_labels)):
                    raise ValueError("'event_labels' must contain only unique strings")
                num_events = len(event_labels)
        else:
            raise TypeError(
                "'event_labels' must be either an integer or a list of strings"
            )

        event_values = list(range(1, num_events + 1))
        events_total = num_events * event_repets
        log0.debug(f"event_values = {event_values}")
        log0.debug(f"event_labels = {event_labels}")
        log0.debug(f"event_repets = {event_repets}")
        log0.debug(f"events_total = {events_total}")

        event_samp = np.arange(
            event_begins, event_begins + event_interv * events_total, event_interv
        )
        event_prec = np.zeros(events_total, dtype=int)
        event_code = np.repeat(event_values, event_repets)
        np.random.shuffle(event_code)

        events = np.column_stack((event_samp, event_prec, event_code))
        event_IDs = {key: val for key, val in zip(event_labels, event_values)}
        event_desc = {val: key for key, val in zip(event_labels, event_values)}

        events_df = pd.DataFrame(
            {
                "sample_num": event_samp,
                "preceding_val": event_prec,
                "event_code": event_code,
            }
        )
        events_df["event_labels"] = events_df.event_code.map(event_desc)

        return events, event_IDs, event_desc, events_df

    def make_singularity_events(
        self,
        event_labels: Union[int, List[str]] = ["singularity"],
        event_repets: int = 1,
        event_interv: int = 1000,
        event_begins: int = 5000,
    ):
        """
        Generate singularity event(s) and store them in the corresponding properties.

        This method creates singularity events using the `make_dummy_events` method
        and stores the results in the class properties. By default, it generates
        a single "singularity" event.

        Parameters
        ----------
        event_labels : List[str], optional
            Labels for the singularity events. Default is ["singularity"].
        event_repets : int, optional
            Number of repetitions for each event. Default is 1.
        event_interv : int, optional
            Interval between events in samples. Default is 1000.
        event_begins : int, optional
            Sample number at which the first event begins. Default is 5000.

        Returns
        -------
        None
            The method doesn't return any value, but updates the following
            class attributes:
            - _singularity_events : numpy.ndarray
            - _singularity_event_IDs : Dict[str, int]
            - _singularity_events_desc : Dict[int, str]
            - _singularity_events_df : pandas.DataFrame

        See Also
        --------
        make_dummy_events : The underlying method used to generate events.

        Notes
        -----
        This method is primarily used to generate a single singularity event,
        but can be customized to generate multiple events if needed.

        Examples
        --------
        >>> gte = GTE()
        >>> gte.make_singularity_events()  # Uses default parameters
        >>> gte.make_singularity_events(event_labels=["start", "end"], event_repets=2)
        """
        (
            self._singularity_events,
            self._singularity_event_IDs,
            self._singularity_events_desc,
            self._singularity_events_df,
        ) = self.make_dummy_events(
            event_labels=event_labels,
            event_repets=event_repets,
            event_interv=event_interv,
            event_begins=event_begins,
        )
        log0.info("Singularity events generated and stored.")

    def make_experimental_events(
        self,
        event_labels: Union[int, List[str]] = ["Ev01", "Ev02"],
        event_repets: int = 100,
        event_interv: int = 2000,
        event_begins: int = 5000,
    ):
        """
        Generate experimental events and store them in the corresponding properties.

        This method creates experimental events using the `make_dummy_events` method
        and stores the results in the class properties. By default, it generates
        two types of events ("Ev1" and "Ev2") repeated 100 times each.

        Parameters
        ----------
        event_labels : List[str], optional
            Labels for the experimental events. Default is ["Ev1", "Ev2"].
        event_repets : int, optional
            Number of repetitions for each event. Default is 100.
        event_interv : int, optional
            Interval between events in samples. Default is 2000.
        event_begins : int, optional
            Sample number at which the first event begins. Default is 5000.

        Returns
        -------
        None
            The method doesn't return any value, but updates the following
            class attributes:
            - _experimental_events : numpy.ndarray
            - _experimental_event_IDs : Dict[str, int]
            - _experimental_events_desc : Dict[int, str]
            - _experimental_events_df : pandas.DataFrame

        See Also
        --------
        make_dummy_events : The underlying method used to generate events.

        Notes
        -----
        This method is designed to generate multiple experimental events,
        simulating a typical experimental paradigm with repeated trials.

        Examples
        --------
        >>> gte = GTE()
        >>> gte.make_experimental_events()  # Uses default parameters
        >>> gte.make_experimental_events(
        ...     event_labels=["cond1", "cond2", "cond3"],
        ...     event_repets=50,
        ...     event_interv=1500
        ... )
        """
        (
            self._experimental_events,
            self._experimental_event_IDs,
            self._experimental_events_desc,
            self._experimental_events_df,
        ) = self.make_dummy_events(
            event_labels=event_labels,
            event_repets=event_repets,
            event_interv=event_interv,
            event_begins=event_begins,
        )
        log0.info("Experimental events generated and stored.")

    @property
    def singularity_events(self) -> Optional[np.ndarray]:
        """
        Get the singularity events array.

        Returns
        -------
        Optional[np.ndarray]
            A 2D numpy array containing singularity event information, or None if not set.
            Each row represents an event with columns:
            - Event onset (in samples)
            - Signal value of the immediately preceding sample
            - Event code

        See Also
        --------
        make_singularity_events : Method to generate singularity events.

        Notes
        -----
        This property is populated by calling the `make_singularity_events` method.
        """
        return self._singularity_events

    @property
    def singularity_event_IDs(self) -> Optional[Dict[str, int]]:
        """
        Get the singularity event IDs.

        Returns
        -------
        Optional[Dict[str, int]]
            A dictionary mapping singularity event labels to their numeric codes,
            or None if not set.

        See Also
        --------
        make_singularity_events : Method to generate singularity events.

        Notes
        -----
        This property is populated by calling the `make_singularity_events` method.
        """
        return self._singularity_event_IDs

    @property
    def singularity_events_desc(self) -> Optional[Dict[int, str]]:
        """
        Get the singularity events description.

        Returns
        -------
        Optional[Dict[int, str]]
            A dictionary mapping numeric event codes to their corresponding labels
            for singularity events, or None if not set.

        See Also
        --------
        make_singularity_events : Method to generate singularity events.

        Notes
        -----
        This property is populated by calling the `make_singularity_events` method.
        """
        return self._singularity_events_desc

    @property
    def singularity_events_df(self) -> Optional[pd.DataFrame]:
        """
        Get the singularity events DataFrame.

        Returns
        -------
        Optional[pd.DataFrame]
            A pandas DataFrame containing detailed information about singularity events,
            or None if not set. Columns include:
            - sample_num: Event onset (in samples)
            - preceding_val: Signal value of the immediately preceding sample
            - event_code: Numeric code for the event
            - event_labels: String label for the event

        See Also
        --------
        make_singularity_events : Method to generate singularity events.

        Notes
        -----
        This property is populated by calling the `make_singularity_events` method.
        """
        return self._singularity_events_df

    @property
    def experimental_events(self) -> Optional[np.ndarray]:
        """
        Get the experimental events array.

        Returns
        -------
        Optional[np.ndarray]
            A 2D numpy array containing experimental event information, or None if not set.
            Each row represents an event with columns:
            - Event onset (in samples)
            - Signal value of the immediately preceding sample
            - Event code

        See Also
        --------
        make_experimental_events : Method to generate experimental events.

        Notes
        -----
        This property is populated by calling the `make_experimental_events` method.
        """
        return self._experimental_events

    @property
    def experimental_event_IDs(self) -> Optional[Dict[str, int]]:
        """
        Get the experimental event IDs.

        Returns
        -------
        Optional[Dict[str, int]]
            A dictionary mapping experimental event labels to their numeric codes,
            or None if not set.

        See Also
        --------
        make_experimental_events : Method to generate experimental events.

        Notes
        -----
        This property is populated by calling the `make_experimental_events` method.
        """
        return self._experimental_event_IDs

    @property
    def experimental_events_desc(self) -> Optional[Dict[int, str]]:
        """
        Get the experimental events description.

        Returns
        -------
        Optional[Dict[int, str]]
            A dictionary mapping numeric event codes to their corresponding labels
            for experimental events, or None if not set.

        See Also
        --------
        make_experimental_events : Method to generate experimental events.

        Notes
        -----
        This property is populated by calling the `make_experimental_events` method.
        """
        return self._experimental_events_desc

    @property
    def experimental_events_df(self) -> Optional[pd.DataFrame]:
        """
        Get the experimental events DataFrame.

        Returns
        -------
        Optional[pd.DataFrame]
            A pandas DataFrame containing detailed information about experimental events,
            or None if not set. Columns include:
            - sample_num: Event onset (in samples)
            - preceding_val: Signal value of the immediately preceding sample
            - event_code: Numeric code for the event
            - event_labels: String label for the event

        See Also
        --------
        make_experimental_events : Method to generate experimental events.

        Notes
        -----
        This property is populated by calling the `make_experimental_events` method.
        """
        return self._experimental_events_df

    def generate_waveform_basic(
        self,
        times: np.ndarray,
        latency: float = 0.25,
        duration: float = 0.20,
        amplitude: float = 1,
        scale: float = 1e-9,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Generate source time courses for simulated evoked responses.

        This method creates a time series that represents an evoked response
        in neuroscience data, particularly useful for MEG/EEG simulations.
        It combines a sinusoidal wave (representing oscillatory activity)
        with a Gaussian envelope (representing the overall shape of the evoked response).

        Parameters
        ----------
        times : np.ndarray
            Time points at which to evaluate the function.
        latency : float, optional
            The center time of the evoked response (in same units as 'times').
            Default is 0.25.
        duration : float, optional
            The duration of the evoked response (impacts the width of the Gaussian).
            Default is 0.20.
        amplitude : float, optional
            The amplitude of the waveform. Default is 1.
        scale : float, optional
            Scaling factor for the amplitude. Default is 1e-9 (nanoamperes).
        rng : np.random.Generator, optional
            Random number generator for adding jitter. If None, np.random is used.

        Returns
        -------
        np.ndarray
            The generated time series of the evoked response.

        Notes
        -----
        - The function uses a fixed frequency in the beta band (15 Hz).
        - The amplitude is scaled to be in the order of nanoamperes (1e-9).
        - A small random jitter is added to the peak of the Gaussian for variability.
        """
        # Define the oscillation frequency (e.g., beta band)
        f = 15  # Hz
        # Generate the sinusoidal component (oscillatory activity)
        sinusoid = np.sin(2 * np.pi * f * (times - latency))
        # Standard deviation for the Gaussian envelope
        sigma = 0.100 * duration
        # Generate the Gaussian envelope
        # - Adding a small random jitter to the peak time
        if rng is not None:
            jitter = (sigma / 4.0) * rng.random()
        else:
            jitter = (sigma / 4.0) * np.random.rand()

        # Example boundary check (CONSIDER implementing clip)
        # jitter = min(max(jitter, -times.max()), times.max())
        # jitter = np.clip(jitter, -sigma, sigma)

        gf = np.exp(-((times - latency - jitter) ** 2) / (2 * (sigma**2)))

        # Combine sinusoid and Gaussian, and scale to nanoamperes
        wf0 = sinusoid * gf * scale * amplitude
        return wf0

    @property
    def activ0(self) -> Optional[Dict[str, Dict[str, Dict[str, float]]]]:
        """
        Get the predefined activations.

        Returns
        -------
        Optional[Dict[str, Dict[str, Dict[str, float]]]]
            A nested dictionary of predefined activations, or None if not set.
        """
        return self._activ0

    @property
    def activ0_labels(self) -> Optional[List[str]]:
        """
        Get the sorted list of unique activation labels (brain regions).

        Returns
        -------
        List[str]
            Sorted list of unique activation labels.
        """
        return self._activ0_labels

    @property
    def activ0_events(self) -> Optional[List[str]]:
        """
        Get the list of activation events.

        Returns
        -------
        List[str]
            List of activation events.
        """
        return self._activ0_events

    def set_predefined_activations(
        self,
        activations: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        event_repets: int = 100,
        event_interv: int = 2000,
        event_begins: int = 5000,
    ) -> None:
        """
        Set predefined activations for different events and brain regions.

        Parameters
        ----------
        activations : Optional[Dict[str, Dict[str, Dict[str, float]]]], optional
            A nested dictionary of activations. If None, a default set of activations is used.
            The structure is:
            {
                "Event1": {
                    "Region1": {"lat": float, "dur": float, "amp": float},
                    "Region2": {"lat": float, "dur": float, "amp": float},
                    ...
                },
                "Event2": {
                    ...
                },
                ...
            }
            Where:
            - "lat" is latency
            - "dur" is duration
            - "amp" is amplitude

        Returns
        -------
        None

        Notes
        -----
        This method sets the `activ0`, `activ0_labels`, and `activ0_events` properties of the class.
        """
        if activations is None:
            activations = {
                "Ev01": {
                    "frontalpole_1-lh": dict(lat=0.35, dur=0.20, amp=300),
                    "superiorfrontal_1-lh": dict(lat=0.25, dur=0.40, amp=320),
                    "rostralmiddlefrontal_7-lh": dict(lat=0.45, dur=0.20, amp=320),
                    "superiorparietal_3-rh": dict(lat=0.25, dur=0.60, amp=320),
                },
                "Ev02": {
                    "superiorfrontal_1-lh": dict(lat=0.35, dur=0.20, amp=320),
                    "superiorparietal_1-rh": dict(lat=0.45, dur=0.20, amp=440),
                    "superiortemporal_1-lh": dict(lat=0.25, dur=0.40, amp=320),
                    "precentral_13-lh": dict(lat=0.35, dur=0.20, amp=320),
                    "lateraloccipital_2-rh": dict(lat=0.45, dur=0.30, amp=440),
                    "lateraloccipital_4-lh": dict(lat=0.25, dur=0.20, amp=320),
                },
            }

        self._activ0 = activations
        self._activ0_labels = sorted(
            set(key for inner_dict in activations.values() for key in inner_dict)
        )
        self._activ0_events = list(activations.keys())

        self.make_experimental_events(
            event_labels=self._activ0_events,
            event_repets=event_repets,
            event_interv=event_interv,
            event_begins=event_begins,
        )

        self._labels4 = [
            label for label in self._labels2 if label.name in self._activ0_labels
        ]
        self._label4_names = self._activ0_labels

    def set_randomized_activations(
        self,
        num_labels: int,
        num_labels_per_event: int,
        event_labels: Union[int, List[str]],
        event_repets: int = 100,
        event_interv: int = 2000,
        event_begins: int = 5000,
        allow_label_repetition: bool = True,
    ):
        """
        Set randomized activations.

        Set randomized activations by sampling labels and assigning them to events,
        and update experimental events to match.

        This method first validates all inputs, then generates experimental events,
        and finally creates random activations for these events.

        Parameters
        ----------
        num_labels : int
            Total number of unique labels to (pre-)sample from self._label0_names.
        num_labels_per_event : int
            Number of labels to assign to each event.
        event_labels : int or List[str]
            If int, the number of unique events to generate. Event labels will be created
            automatically as "Ev001", "Ev002", etc.
            If List[str], custom labels for events. Must be unique.
        event_repets : int, optional
            Number of repetitions for each event. Default is 100.
        event_interv : int, optional
            Interval between events in samples. Default is 2000.
        event_begins : int, optional
            Sample number at which the first event begins. Default is 5000.
        allow_label_repetition : bool, optional
            If True, allows the same label to be used across different events.
            If False, ensures each label is used only once. Default is True.

        Returns
        -------
        None

        Notes
        -----
        This method updates the following class attributes:
        - _activ0 : Dict containing the randomized activations
        - _activ0_labels : List of unique activation labels
        - _activ0_events : List of event names
        - Experimental event attributes (_experimental_events, _experimental_event_IDs, etc.)

        Raises
        ------
        ValueError
            - If there are not enough labels in self._label0_names.
            - If the number of labels per event * number of events > total number of labels
              (when allow_label_repetition is False).
            - If event_labels is neither an integer nor a list of strings.

        Examples
        --------
        >>> gte = GTE()
        >>> gte.set_randomized_activations(
        ...     num_labels=20,
        ...     num_labels_per_event=3,
        ...     event_labels=["Event1", "Event2", "Event3"],
        ...     event_repets=50,
        ...     allow_label_repetition=True
        ... )
        """
        # FIXME IMPORTANT
        """
        raise NotImplementedError(
            "The 'set_randomized_activations' method is not implemented yet. "
            "Stay tuned for future updates!"
        )
        """
        # Validate inputs
        # Also this code is by necessity repetition from the 'make_dummy_events'
        # This is because we do not want to run it and update events before this
        # function execution is guaranteed to succeed.
        if isinstance(event_labels, int):
            num_events = event_labels
            leadz = len(str(num_events + 1)) + 1
            event_names = [f"Ev{ii:0{leadz}d}" for ii in range(1, num_events + 1)]
        elif isinstance(event_labels, list):
            if all(isinstance(label, str) for label in event_labels):
                event_names = event_labels
                num_events = len(event_names)
            else:
                raise ValueError("If 'event_labels' are strings they must be unique")
        else:
            raise ValueError(
                "'event_labels' must be either an integer or a list of strings"
            )

        if len(self._label0_names) < num_labels:
            raise ValueError(
                f"Not enough labels to sample from. "
                f"Required: {num_labels}, Available: {len(self._label0_names)}"
            )

        if allow_label_repetition:
            sampled_labels = random.sample(self._label0_names, num_labels)
        else:
            if num_labels_per_event * num_events > num_labels:
                raise ValueError(
                    "Too many labels per event specified. "
                    f"Max possible: {num_labels // num_events} "
                    "when allow_label_repetition is False"
                )
            # Prevent label reuse across events
            sampled_labels = random.sample(
                self._label0_names, num_labels_per_event * num_events
            )
            label_iterator = iter(sampled_labels)

        # Generate experimental events
        self.make_experimental_events(
            event_labels=event_names,
            event_repets=event_repets,
            event_interv=event_interv,
            event_begins=event_begins,
        )

        # Sample labels

        # Initialize the activations dictionary
        activations = {event: {} for event in event_names}

        # Assign labels to events
        for event in event_names:
            if allow_label_repetition:
                event_labels = random.choices(sampled_labels, k=num_labels_per_event)
            else:
                # event_labels = random.sample(sampled_labels, num_labels_per_event) # FIXED
                event_labels = [
                    next(label_iterator) for _ in range(num_labels_per_event)
                ]
            for label in event_labels:
                activations[event][label] = {
                    "lat": round(random.uniform(0.1, 0.4), 2),
                    "dur": round(random.uniform(0.1, 0.4), 2),
                    "amp": round(random.uniform(200, 500), 2),
                }

        # Update class properties
        self._activ0 = activations
        self._activ0_labels = sorted(
            set(label for event in activations.values() for label in event)
        )
        self._activ0_events = event_names
        self._labels4 = [
            label for label in self._labels2 if label.name in self._activ0_labels
        ]
        self._label4_names = self._activ0_labels

        log0.info(
            f"Randomized activations set for {num_events} events with {num_labels} total labels."
        )

    def activations_to_dataframe(self) -> pd.DataFrame:
        """
        Convert the predefined activations to a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame representation of the activations.
            The DataFrame has the following columns:

            - Event : str
                The name of the event.
            - Region : str
                The name of the brain region.
            - Latency : float
                The latency of the activation in seconds.
            - Duration : float
                The duration of the activation in seconds.
            - Amplitude : float
                The amplitude of the activation.

        Raises
        ------
        ValueError
            If activations have not been set (activ0 is None).

        Notes
        -----
        This method assumes that the activations have been previously set
        using the `set_predefined_activations` method or a similar mechanism.

        Examples
        --------
        >>> gte = GTE()
        >>> gte.set_predefined_activations(...)
        >>> df = gte.activations_to_dataframe()
        >>> print(df.head())

        """
        if self._activ0 is None:
            raise ValueError(
                "Activations have not been set. Use set_predefined_activations() or "
                "set_randomized_activations() first."
            )

        data = []
        for event, regions in self._activ0.items():
            for region, params in regions.items():
                data.append(
                    {
                        "Event": event,
                        "Region": region,
                        "Latency": params["lat"],
                        "Duration": params["dur"],
                        "Amplitude": params["amp"],
                    }
                )

        return pd.DataFrame(data)

    @property
    def activ0_trial_samp_total(self) -> int:
        """
        Get the number of samples for the activation trial.

        Returns
        -------
        int
            The number of samples for the activation trial.
        """
        if self._activ0_trial_samp_total is None:
            self._activ0_trial_samp_total = 1000  # Default value
        return self._activ0_trial_samp_total

    @activ0_trial_samp_total.setter
    def activ0_trial_samp_total(self, value: int):
        """
        Set the number of samples for the activation trial.

        Parameters
        ----------
        value : int
            The number of samples to set for the activation trial.
        """
        self._activ0_trial_samp_total = value

    @property
    def activ0_trial_times(self) -> np.ndarray:
        """
        Get the times array.

        Returns
        -------
        np.ndarray
            A 1D array of time points based on the number of samples and sampling frequency.
        """
        if self._info0 is None:
            raise ValueError(
                "The _info0 property must be set before accessing activ0_trial_times."
            )

        return (
            np.arange(self._activ0_trial_samp_total, dtype=np.float64)
            / self._info0["sfreq"]
        )

    @activ0_trial_times.setter
    def activ0_trial_times(self, value: np.ndarray):
        """
        Set the times array. This should be calculated based on the activation trial sample count and info0.

        Parameters
        ----------
        value : np.ndarray
            A new times array to set.
        """
        # self._activ0_trial_times = value
        raise NotImplementedError(
            "The 'activ0_trial_times' SETTER method is not implemented yet."
        )

    @property
    def source_simulator(self) -> Optional[SourceSimulator]:
        """
        Get the SourceSimulator object.

        Returns
        -------
        Optional[mne.simulation.SourceSimulator]
            The SourceSimulator object if initialized, otherwise None.
        """
        return self._source_simulator

    def initialize_source_simulator(self):
        """
        Initialize and store a SourceSimulator object based on _src0 and _info0.

        This method creates a SourceSimulator object using the source space (_src0)
        and the sampling frequency from the measurement info (_info0). The created
        SourceSimulator is stored in the _source_simulator attribute.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If _src0 or _info0 is not set.

        Notes
        -----
        This method should be called after _src0 and _info0 have been properly set.
        The SourceSimulator object can be accessed via the source_simulator property
        after initialization.

        Examples
        --------
        >>> gte = GTE()
        >>> gte.set_src0(...)  # Set the source space
        >>> gte.set_info0(...)  # Set the measurement info
        >>> gte.initialize_source_simulator()
        >>> sim = gte.source_simulator
        """
        if self._src0 is None:
            raise ValueError(
                "Source space (_src0) must be set before initializing SourceSimulator."
            )
        if self._info0 is None:
            raise ValueError(
                "Measurement info (_info0) must be set before initializing SourceSimulator."
            )

        src = self._src0
        tstep = 1 / self._info0["sfreq"]

        try:
            self._source_simulator = SourceSimulator(src, tstep=tstep)
            log0.info(f"SourceSimulator initialized with tstep={tstep:.6f}s")
        except Exception as e:
            log0.error(f"Failed to initialize SourceSimulator: {str(e)}")
            raise

    def add_data_to_source_simulator(self):
        """
        Add activation data to the source simulator using the internal attributes.

        This method uses `_activ0`, `_experimental_event_IDs`, and `_experimental_events` attributes
        to add data to the source simulator.

        Raises
        ------
        AssertionError
            If the label is not found in the labels list.
        ValueError
            If the source simulator is not initialized or if required data is missing.
        """
        if self._source_simulator is None:
            raise ValueError(
                "Source simulator is not initialized. Call initialize_source_simulator first."
            )

        if self._activ0 is None:
            raise ValueError(
                "Activation data (_activ0) is not set. Set activations before calling this method."
            )

        if self._experimental_event_IDs is None or self._experimental_events is None:
            raise ValueError(
                "Experimental events (_experimental_event_IDs or _experimental_events) are not set."
            )

        for act_idx, (act_key, act_val) in enumerate(self._activ0.items()):
            # Get the event code corresponding to the activation key
            act_code = self._experimental_event_IDs.get(act_key)
            if act_code is None:
                raise ValueError(
                    f"Activation key '{act_key}' not found in experimental_event_IDs."
                )
            # Filter events for the current activation key
            tmp_events = self._experimental_events[
                np.where(self._experimental_events[:, 2] == act_code)[0], :
            ]
            log0.warning(f"{act_idx}: {act_key} [{act_code}] {tmp_events.shape}")

            # Loop through each label and add data to the source simulator
            for lab_idx, (lab_name, lab_params) in enumerate(act_val.items()):
                log0.warning(f"- {lab_idx}: {lab_name} --- {lab_params}")

                # Find the corresponding label in the labels4 (self._labels4)
                tmp_label = [label for label in self._labels4 if label.name == lab_name]
                if len(tmp_label) != 1:
                    raise ValueError(f"PROBLEM with label {lab_name!r} selection!")
                tmp_label = tmp_label[0]
                log0.warning(f"  - {tmp_label = }")

                # Extract activation parameters
                tmp_lat = lab_params["lat"]
                tmp_dur = lab_params["dur"]
                tmp_amp = lab_params["amp"]
                log0.warning(f"  - {tmp_lat = }, {tmp_dur = }, {tmp_amp = }")

                # Generate the waveform for this label's activation
                tmp_wf = self.generate_waveform_basic(
                    times=self.activ0_trial_times,
                    latency=tmp_lat,
                    duration=tmp_dur,
                    amplitude=tmp_amp,
                )

                # Add data to the source simulator
                self._source_simulator.add_data(tmp_label, tmp_wf, tmp_events)
                log0.warning(
                    f"  - data added source waveform to source simulator for {lab_name}"
                )

    @property
    def activ0_stc(self) -> Optional[mne.SourceEstimate]:
        """
        Get the cached source time course (STC).

        Returns
        -------
        stc : mne.SourceEstimate
            The cached source estimate (STC) stored in the class.

        Raises
        ------
        ValueError
            If the activ0_stc has not been set yet.
        """
        if self._activ0_stc is None:
            raise ValueError(
                "activ0_stc has not been extracted yet. Use extract_activ0_stc to set it."
            )
        return self._activ0_stc

    def extract_activ0_stc(self):
        """
        Set the source time course (STC) from the source simulator and cache it.

        This method runs the source simulator's `get_stc()` method and stores the
        result in `_activ0_stc` for future use.

        Raises
        ------
        ValueError
            If the source simulator is not initialized or if no data is available in the simulator.
        """
        if self._source_simulator is None:
            raise ValueError(
                "Source simulator is not initialized. Call initialize_source_simulator first."
            )

        self._activ0_stc = self._source_simulator.get_stc()

    @property
    def activ0_raw(self) -> Optional[mne.io.Raw]:
        """
        Get the cached time course (Raw).

        Returns
        -------
        stc : mne.io.Raw
            The cached time course (Raw) stored in the class.

        Raises
        ------
        ValueError
            If the activ0_raw has not been set yet.
        """
        if self._activ0_raw is None:
            raise ValueError(
                "activ0_raw has not been extracted yet. Use extract_activ0_raw to set it."
            )
        return self._activ0_raw

    @property
    def activ2_raw(self) -> Optional[mne.io.Raw]:
        """
        Get the cached time course (Raw) with noise added.

        Returns
        -------
        stc : mne.io.Raw
            The cached time course (Raw) stored in the class.

        Raises
        ------
        ValueError
            If the activ2_raw has not been set yet.
        """
        if self._activ2_raw is None:
            raise ValueError(
                "activ2_raw has not been extracted yet. Use extract_activ2_raw to set it."
            )
        return self._activ2_raw

    def extract_activ0_raw(self):
        """
        Set the time course (Raw) from the source simulator and cache it.

        This method runs the source simulator's `get_stc()` method and stores the
        result in `_activ0_raw` for future use.

        Raises
        ------
        ValueError
            If the source simulator is not initialized or if no data is available in the simulator.
        """
        if self._source_simulator is None:
            raise ValueError(
                "Source simulator is not initialized. Call initialize_source_simulator first."
            )
        if self._info0 is None:
            raise ValueError(
                "Measurement info (_info0) must be set before simulating raw data."
            )
        if self._fwd0 is None:
            raise ValueError(
                "Forward solution (_fwd0) must be set before simulating raw data."
            )

        self._activ0_raw = mne.simulation.simulate_raw(
            info=self._info0,
            stc=self._source_simulator,  # an instance of mne.SourceEstimate or mne.simulation.SourceSimulator
            src=None,  # Can be None if forward is provided.
            bem=None,  # Can be None if forward is provided.
            forward=self._fwd0,
            verbose=True,
            n_jobs=12,
        )
        events_annot4x0 = mne.annotations_from_events(
            events=self._experimental_events,
            sfreq=self._activ0_raw.info["sfreq"],
            event_desc=self.experimental_events_desc,
        )
        self._activ0_raw.set_annotations(events_annot4x0)
        self._activ0_raw.set_eeg_reference(projection=True)

    def extract_activ2_raw(self):
        """
        Set the time course (Raw) from the source simulator and cache it.

        This method runs the source simulator's `get_stc()` method and stores the
        result in `_activ0_raw` for future use.

        Raises
        ------
        ValueError
            If the source simulator is not initialized or if no data is available in the simulator.
        """
        if self._activ0_raw is None:
            raise ValueError(
                "Clean _activ0_raw data is not initialized. Call extract_activ0_raw first."
            )

        self._activ2_raw = self._activ0_raw.copy()
        mne.simulation.add_noise(
            self._activ2_raw, cov=self._genuine_noise_cov0, random_state=0
        )
        mne.simulation.add_eog(self._activ2_raw, random_state=0)
        if len(mne.pick_types(self._activ2_raw.info, meg=True)) > 0:
            mne.simulation.add_ecg(self._activ2_raw, random_state=0)

    @property
    def activ2_epochs(self) -> Optional[mne.Epochs]:
        """
        Get the cached time course (Epochs).

        Returns
        -------
        stc : mne.Epochs
            The cached time course (Epochs) stored in the class.

        Raises
        ------
        ValueError
            If the activ2_epochs has not been set yet.
        """
        if self._activ2_epochs is None:
            raise ValueError(
                "activ2_epochs has not been extracted yet. Use extract_activ2_epochs_and_evoked to set it."
            )
        return self._activ2_epochs

    @property
    def activ2_evoked(self) -> Optional[Dict[str, mne.Evoked]]:
        """
        Get the cached time course (Evoked).

        Returns
        -------
        stc : mne.Evoked
            The cached time course (Evoked) stored in the class.

        Raises
        ------
        ValueError
            If the activ2_evoked has not been set yet.
        """
        if self._activ2_evoked is None:
            raise ValueError(
                "activ2_evoked has not been extracted yet. Use extract_activ2_epochs_and_evoked to set it."
            )
        return self._activ2_evoked

    def extract_activ2_epochs_and_evoked(
        self,
        tmin: float = -0.4,  # Added as parameter
        tmax: float = 1.2,  # Added as parameter
        baseline: Tuple[Optional[float], Optional[float]] = (
            None,
            0,
        ),
    ):
        """
        Extract epochs.

        Raises
        ------
        ValueError
            If the raw data is not initialized or if no data is available in the simulator.
        """
        if self._activ2_raw is None:
            raise ValueError(
                "Clean _activ2_raw data is not initialized. Call extract_activ2_raw first."
            )

        self._activ2_epochs = mne.Epochs(
            self._activ2_raw,
            self._experimental_events,
            self._experimental_event_IDs,
            tmin=tmin,  # Use parameter
            tmax=tmax,  # Use parameter
            baseline=baseline,  # Use parameter
        )

        self._activ2_evoked = {}
        for ev in self._experimental_event_IDs.keys():
            self._activ2_evoked[ev] = self._activ2_epochs[ev].average()

    @property
    def activ2_data_cov(self) -> Optional[mne.Covariance]:
        """
        Get the data covariance.

        Returns
        -------
        Optional[mne.Covariance]
            The data covariance if set, otherwise None.
        """
        return self._activ2_data_cov

    @property
    def activ2_noise_cov(self) -> Optional[mne.Covariance]:
        """
        Get the noise covariance.

        Returns
        -------
        Optional[mne.Covariance]
            The noise covariance if set, otherwise None.
        """
        return self._activ2_noise_cov

    @property
    def activ2_common_cov(self) -> Optional[mne.Covariance]:
        """
        Get the common covariance.

        Returns
        -------
        Optional[mne.Covariance]
            The common covariance if set, otherwise None.
        """
        return self._activ2_common_cov

    def compute_covariances(
        self,
        data_tmin: float = 0.01,
        data_tmax: float = 0.60,
        noise_tmin: Optional[float] = None,
        noise_tmax: float = 0,  # This value is reasonably set to the onset of trial/stimuli
        method: str = "empirical",
    ):
        """
        Compute and set data, noise, and common covariances.

        This method computes the data covariance, noise covariance, and common covariance
        using the specified time ranges and method.

        Parameters
        ----------
        data_tmin : float, optional
            Start time for data covariance computation. Default is 0.01.
        data_tmax : float, optional
            End time for data covariance computation. Default is 0.60.
        noise_tmin : float or None, optional
            Start time for noise covariance computation. Default is None.
        noise_tmax : float, optional
            End time for noise covariance computation. Default is 0.
        method : str, optional
            Method to compute covariance. Default is "empirical".

        Raises
        ------
        ValueError
            If _activ2_epochs is not set.

        Notes
        -----
        This method sets _activ2_data_cov, _activ2_noise_cov, and _activ2_common_cov.
        """
        if self._activ2_epochs is None:
            raise ValueError("_activ2_epochs must be set before computing covariances.")

        self._activ2_data_cov = mne.compute_covariance(
            self._activ2_epochs, tmin=data_tmin, tmax=data_tmax, method=method
        )

        self._activ2_noise_cov = mne.compute_covariance(
            self._activ2_epochs, tmin=noise_tmin, tmax=noise_tmax, method=method
        )

        self._activ2_common_cov = self._activ2_data_cov + self._activ2_noise_cov

        log0.info("Data, noise, and common covariances computed and set.")

    @property
    def bf_filters(self) -> Optional[Dict[str, Beamformer]]:
        """
        Get the LCMV beamformer filters.

        Returns
        -------
        Optional[Dict[str, mne.beamformer.Beamformer]]
            Dictionary of LCMV beamformer filters for each condition, or None if not computed.
            Each filter is an instance of mne.beamformer.Beamformer.
        """
        return self._bf_filters

    @property
    def stcs(self) -> Optional[Dict[str, mne.SourceEstimate]]:
        """
        Get the source estimates computed by applying LCMV beamformer filters.

        Returns
        -------
        Optional[Dict[str, mne.SourceEstimate]]
            Dictionary of source estimates for each condition, or None if not computed.
        """
        return self._stcs

    def compute_lcmv_bf_filters(
        self,
        pick_ori: str = "vector",
        weight_norm: str = "unit-noise-gain-invariant",
        reg: float = 0.05,
        make_lcmv_func: Callable = mne.beamformer.make_lcmv,
    ):
        """
        Compute LCMV beamformer filters for each condition in _activ2_evoked.

        Parameters
        ----------
        pick_ori : str, optional
            Orientation selection strategy. Default is "vector".
        weight_norm : str, optional
            Type of weight normalization. Default is "unit-noise-gain-invariant".
        reg : float, optional
            Regularization parameter. Default is 0.05.

        Raises
        ------
        ValueError
            If required attributes (__activ2_evoked, _fwd0, _activ2_data_cov, _activ2_noise_cov) are not set.

        Notes
        -----
        This method computes LCMV beamformer filters (mne.beamformer.Beamformer objects)
        for each condition and stores them in the _bf_filters attribute.
        The default parameters are set to:
        - pick_ori: "vector" for vector beamformer
        - weight_norm: "unit-noise-gain-invariant" for unit-noise gain beamformer
        """
        if (
            self._activ2_evoked is None
            or self._fwd0 is None
            or self._activ2_data_cov is None
            or self._activ2_noise_cov is None
        ):
            raise ValueError(
                "Evoked data, forward solution, and covariances must be set before computing filters."
            )

        self._bf_filters = {}
        for key, val in self._activ2_evoked.items():
            self._bf_filters[key] = make_lcmv_func(
                info=val.info,
                forward=self._fwd0,
                data_cov=self._activ2_data_cov,
                noise_cov=self._activ2_noise_cov,
                pick_ori=pick_ori,
                weight_norm=weight_norm,
                reg=reg,
                rank=None,
            )

        log0.info(
            f"LCMV beamformer filters computed for {len(self._bf_filters)} conditions "
            f"with pick_ori='{pick_ori}' and weight_norm='{weight_norm}'."
        )

    def apply_lcmv_bf_filters(
        self, apply_lcmv_func: Callable = mne.beamformer.apply_lcmv
    ):
        """
        Apply LCMV beamformer filters to the evoked data and store the results.

        This method applies the previously computed LCMV beamformer filters
        to the evoked data to create source estimates, and stores them in the _stcs attribute.

        Raises
        ------
        ValueError
            If filters or evoked data are not set.

        Notes
        -----
        The resulting source estimates can be accessed via the stcs property after calling this method.
        """
        if self._bf_filters is None or self._activ2_evoked is None:
            raise ValueError(
                "Filters and evoked data must be set before applying filters."
            )

        self._stcs = {}
        for key, bf_filter in self._bf_filters.items():
            self._stcs[key] = apply_lcmv_func(self._activ2_evoked[key], bf_filter)

        log0.info(
            f"LCMV beamformer applied to {len(self._stcs)} conditions. "
            "Source estimates stored in stcs property."
        )

    def list_properties(self) -> pd.DataFrame:
        """
        List all properties of the GTE dataclass object with their declared and actual types.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns:
            - 'Name': The name of the property (str)
            - 'Declared Type': The type declared in the class definition (str)
            - 'Actual Type': The type of the current value of the property (str)
            - 'Is Optional': Whether the property is declared as Optional (bool)

        Notes
        -----
        This method uses the dataclasses.fields() function to introspect
        the class structure. It includes both regular fields and properties.
        The 'Actual Type' column shows 'None' for properties with no current value.

        Examples
        --------
        >>> gte = GTE()
        >>> gte.subjects_dir = Path("/path/to/subjects")
        >>> properties_df = gte.list_properties()
        >>> print(properties_df)
        """
        properties = []
        for f in fields(self):
            value = getattr(self, f.name)
            declared_type = f.type
            actual_type = type(value).__name__ if value is not None else "None"
            is_optional = False

            # Check if the declared type is Optional
            if get_origin(declared_type) is Union:
                args = get_args(declared_type)
                if type(None) in args:
                    is_optional = True
                    # Remove NoneType from args to get the actual type
                    other_args = [arg for arg in args if arg is not type(None)]
                    declared_type = (
                        other_args[0]
                        if len(other_args) == 1
                        else Union[tuple(other_args)]
                    )

            properties.append(
                {
                    "Name": f.name,
                    "Declared Type": str(declared_type),
                    "Actual Type": actual_type,
                    "Is Optional": is_optional,
                }
            )

        return pd.DataFrame(properties)

    def publish(self):
        """
        Publish. Important method for future implementation.

        Raises
        ------
        NotImplementedError
            This method is not implemented yet.
        """
        raise NotImplementedError(
            "The 'publish' method is not implemented yet. Stay tuned for future updates!"
        )

    def _get_mne_file_path(self, fif_file: Union[str, Path], suffix: str) -> Path:
        """
        Provide helper method to get the full path for MNE files.

        Parameters
        ----------
        fif_file : str or Path
            The filename or path provided by the user.
        suffix : str
            The expected suffix for the file.

        Returns
        -------
        Path
            The full path to the MNE file.

        Raises
        ------
        ValueError
            If the subjects_dir or subject is not set, or if the file does not exist.
        """
        if self.subjects_dir is None or self.subject is None:
            raise ValueError(
                "Both subjects_dir and subject must be set before loading MNE files."
            )

        if isinstance(fif_file, str):
            fif_path = Path(fif_file)
        else:
            fif_path = fif_file

        if not fif_path.is_absolute():
            fif_path = self.subjects_dir / self.subject / "aux" / "mne" / fif_path

        if not fif_path.exists():
            raise ValueError(f"The file '{fif_path}' does not exist.")
        if not fif_path.name.endswith(suffix):
            log0.warning(
                f"The file '{fif_path}' does not end with '{suffix}' in its name. "
                "Please make sure you know what you are doing."
            )

        return fif_path
