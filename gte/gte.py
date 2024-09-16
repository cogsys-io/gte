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
from typing import Optional, Union

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
    _montage0 : mne.channels.DigMontage or None
        Private attribute to store the MNE DigMontage object.

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
    montage0 : mne.channels.DigMontage or None
        MNE DigMontage object loaded from a -montage.fif file.
    """

    _subjects_dir: Optional[Path] = field(default=None, init=False)
    _subject: Optional[str] = field(default=None, init=False)
    _info0: Optional[mne.Info] = field(default=None, init=False)
    _trans0: Optional[mne.transforms.Transform] = field(default=None, init=False)
    _src0: Optional[mne.SourceSpaces] = field(default=None, init=False)
    _bem_model0: Optional[list] = field(default=None, init=False)
    _bem_solution0: Optional[mne.bem.ConductorModel] = field(default=None, init=False)
    _fwd0: Optional[mne.Forward] = field(default=None, init=False)
    _montage0: Optional[mne.channels.DigMontage] = field(default=None, init=False)

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
        fif_path = self._get_mne_file_path(fif_file, ".fif")
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
