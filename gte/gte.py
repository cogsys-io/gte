#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ground Truth Establisher (GTE) Module.

Great stuff!

"""
import numpy as np
import pandas as pd
import nibabel as nib
import pathlib
import mne

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple

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
    _bem_model0: Optional[list] = field(default=None, init=False)
    _bem_solution0: Optional[mne.bem.ConductorModel] = field(default=None, init=False)
    _fwd0: Optional[mne.Forward] = field(default=None, init=False)

    _annot0: Optional[str] = field(default=None, init=False)
    _labels0: Optional[List[mne.Label]] = field(default=None, init=False)
    _label0_names: Optional[str] = field(default=None, init=False)
    _labels2: Optional[List[mne.Label]] = field(default=None, init=False)
    _label2_names: Optional[str] = field(default=None, init=False)

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
    def bem_model0(self) -> Optional[list]:
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
    def annot0(self) -> str:
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
            label2 = mne.label.select_sources(
                subject=self.subject,
                label=label0,
                location=location,
                extent=extent,
                subjects_dir=self.subjects_dir,
            )
            label2.name = f"source-label-{location}-{label0.name}"

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
            automatically as "Ev001", "Ev002", etc.
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
        event_id : dict
            Mapping of event labels to their corresponding values.
        event_desc : dict
            Mapping of event values to their corresponding labels.
        df : pandas.DataFrame
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
            n_events = event_labels
            leadz = len(str(n_events + 1))
            event_labels = [f"Ev{ii:0{leadz}d}" for ii in range(1, n_events + 1)]
        elif isinstance(event_labels, list) and all(
            isinstance(item, str) for item in event_labels
        ):
            assert len(event_labels) == len(
                set(event_labels)
            ), "'event_labels' must contain only unique strings"
            n_events = len(event_labels)
        else:
            raise TypeError(
                "'event_labels' must be either an integer or a list of strings"
            )

        event_values = list(range(1, n_events + 1))
        events_total = n_events * event_repets
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
        event_id = {key: val for key, val in zip(event_labels, event_values)}
        event_desc = {val: key for key, val in zip(event_labels, event_values)}

        df = pd.DataFrame(
            {
                "sample_num": event_samp,
                "preceding_val": event_prec,
                "event_code": event_code,
            }
        )
        df["event_labels"] = df.event_code.map(event_desc)

        return events, event_id, event_desc, df

    def make_singularity_events(
        self,
        event_labels: List[str] = ["singularity"],
        event_repets: int = 1,
        event_interv: int = 1000,
        event_begins: int = 5000,
    ):
        """
        Generate singularity events and store them in the corresponding properties.

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
        event_labels: List[str] = ["Ev1", "Ev2"],
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
