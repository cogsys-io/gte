#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ground Truth Establisher (GTE) Module.

Great stuff!

"""
import numpy as np
import pandas as pd
import nibabel as nib
import pathlib

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GTE:
    """
    Ground Truth Establisher (GTE) class for data gathering and simulations.

    This class is designed to handle data gathering and perform simulations,
    primarily using MNE-Python. It provides functionality to manage subject
    directories and individual subject data for simulations.

    Attributes
    ----------
    _subjects_dir : Path or None
        Private attribute to store the subjects directory path.
    _subject : str or None
        Private attribute to store the current subject identifier.

    Properties
    ----------
    subjects_dir : Path or None
        Path to the subjects directory.
    subject : str or None
        Identifier for the current subject.
    subject_dir : Path or None
        Path to the current subject's directory.
    """

    _subjects_dir: Optional[Path] = field(default=None, init=False)
    _subject: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        """
        Post-initialization method to set up default values if needed.
        """
        pass  # We don't need to do anything here now, as we're not setting defaults

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

    # Additional methods for simulations can be added here
