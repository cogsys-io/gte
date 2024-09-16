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


@dataclass
class GTE:
    """
    Ground Truth Establisher (GTE) class for data gathering and simulations.

    This class is designed to handle data gathering and perform simulations,
    primarily using MNE-Python. It provides functionality to manage subject
    directories, individual subject data, and MNE Info objects for simulations.

    Attributes
    ----------
    _subjects_dir : Path or None
        Private attribute to store the subjects directory path.
    _subject : str or None
        Private attribute to store the current subject identifier.
    _info0 : mne.Info or None
        Private attribute to store the MNE Info object.

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
    """

    _subjects_dir: Optional[Path] = field(default=None, init=False)
    _subject: Optional[str] = field(default=None, init=False)
    _info0: Optional[mne.Info] = field(default=None, init=False)

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
        fif_path = Path(fif_file).expanduser().resolve()
        if not fif_path.exists():
            raise ValueError(f"The file '{fif_path}' does not exist.")
        if fif_path.suffix.lower() != ".fif":
            raise ValueError(f"The file '{fif_path}' is not a .fif file.")

        try:
            self._info0 = mne.io.read_info(fif_path)
        except Exception as e:
            raise RuntimeError(f"Error reading .fif file: {str(e)}")

    # Additional methods for simulations can be added here
