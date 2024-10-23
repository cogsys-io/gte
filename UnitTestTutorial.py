#!/usr/bin/env ipython
# -*- coding: utf-8 -*-


# Development

import os
""" DANGER ZONE:
os.environ['QT_NO_GLIB'] = '1'
os.environ['SESSION_MANAGER'] = ''
"""

devel_mode = False
devel_mode = True
if devel_mode:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# Imports

import mne
import gte
import pathlib
import numpy as np
import srsly

import matplotlib
from matplotlib import pyplot as plt

from gte import GTE
from gte import Log0
from gte import GTE  # noqa: F401
from gte import ddf  # noqa: F401
from gte import rdf  # noqa: F401

# Init

""" OPTIONS:
get_ipython().run_line_magic("matplotlib", "qt")
get_ipython().run_line_magic("matplotlib", "inline")
"""
get_ipython().run_line_magic("matplotlib", "qt")

logZ = Log0()
log0 = logZ.logger

# Init

gte = GTE()
log0.warning(f"{gte = }")
None

# Init: =gte.subjects_dir= and =gte.subject=

gte.subjects_dir = pathlib.Path().home()/"mount/data/subjects/"
log0.info(f"{gte.subjects_dir = }")

gte.subject = "phantomica-20240913T000907"
log0.info(f"{gte.subject = }")

# Init: =gte.info0=

gte.info0 = gte.subjects_dir/gte.subject/f"aux/mne/{gte.subject}-basic-info.fif"
gte.info0 = f"{gte.subject}-basic-info.fif"
log0.info(f"{gte.info0 = }")

# Init: =gte.trans0=

gte.trans0 = f"{gte.subject}-head-mri-trans.fif"
log0.info(f"{gte.trans0 = }")

# Init: =gte.montage0=

gte.montage0 = f"{gte.subject}-basic-montage.fif"
log0.info(f"{gte.montage0 = }")

# Init: =gte.src0=

""" OPTIONS:
gte.src0 = f"{gte.subject}-src-ico5.fif"
gte.src0 = f"{gte.subject}-src-oct6.fif"
"""
gte.src0 = f"{gte.subject}-src-oct6.fif"
log0.info(f"{gte.src0 = }")

# Init: =gte.bem_model0=

""" OPTIONS:
gte.bem_model0 = f"{gte.subject}-bem-model-ico3.fif"
gte.bem_model0 = f"{gte.subject}-bem-model-ico4.fif"
gte.bem_model0 = f"{gte.subject}-bem-model-ico5.fif"
"""
gte.bem_model0 = f"{gte.subject}-bem-model-ico4.fif"
log0.info(f"{type(gte.bem_model0) = }")
log0.info(f"{len(gte.bem_model0) = }")
log0.info(f"{type(gte.bem_model0[0]) = }")
log0.info(f"{len(gte.bem_model0[0]) = }")

# Init: =gte.bem_solution0=

""" OPTIONS:
gte.bem_solution0 = f"{gte.subject}-bem-solution-ico3.fif"
gte.bem_solution0 = f"{gte.subject}-bem-solution-ico4.fif"
gte.bem_solution0 = f"{gte.subject}-bem-solution-ico5.h5"
"""
gte.bem_solution0 = f"{gte.subject}-bem-solution-ico4.fif"
log0.info(f"{gte.bem_solution0 = }")

# Init: =gte.fwd0=

""" OPTIONS:
gte.fwd0 = f"{gte.subject}-fwd-src-ico5-bem-solution-ico3.fif"
gte.fwd0 = f"{gte.subject}-fwd-src-ico5-bem-solution-ico4.fif"
gte.fwd0 = f"{gte.subject}-fwd-src-ico5-bem-solution-ico5.fif"
gte.fwd0 = f"{gte.subject}-fwd-src-oct6-bem-solution-ico3.fif"
gte.fwd0 = f"{gte.subject}-fwd-src-oct6-bem-solution-ico4.fif"
gte.fwd0 = f"{gte.subject}-fwd-src-oct6-bem-solution-ico5.fif"
"""
gte.fwd0 = f"{gte.subject}-fwd-src-oct6-bem-solution-ico4.fif"
log0.info(f"{gte.fwd0 = }")

# Init: =gte.genuine_noise_cov0=

gte.genuine_noise_cov0 = f"{gte.subject}-real-noise-cov.fif"

log0.info(f"{gte.genuine_noise_cov0 = }")
gte.genuine_noise_cov0.plot(gte.info0)

# Init: =gte.annot0=

""" OPTIONS:
gte.annot0 = "aparc"
gte.annot0 = "HCPMMP1"
gte.annot0 = "aparc_sub"
gte.annot0 = "aparc.a2009s"
gte.annot0 = "aparc_sub_fix"
"""
gte.annot0 = "aparc_sub_fix"
log0.info(f"{gte.annot0 = }")

# Init: =gte.labels0=

gte.read_labels_from_annot()
log0.warning(f"{len(gte.labels0) = }")
log0.warning(f"{gte.labels0 = }")
log0.warning(f"{gte.label0_names = }")

# Init: =gte.labels2=

verbose = True
verbose = False
gte.process_labels0(verbose=verbose)

log0.warning(f"{gte.labels2 = }")
log0.warning(f"{gte.label2_names = }")

# Check WaveForm Generation

rng = np.random.RandomState(7)

n_samp = 1000
times = np.arange(n_samp, dtype=np.float64) / gte.info0["sfreq"]

tmp_lat = 0.35
tmp_lat = 0.45
tmp_lat = 0.25

tmp_amp = 2
tmp_amp = 3
tmp_amp = 1

tmp_dur = 0.45
tmp_dur = 0.35
tmp_dur = 0.25

tmp_wf = GTE().generate_waveform_basic(times=times, latency=tmp_lat, duration=tmp_dur, amplitude=tmp_amp)
log0.warning(f"{type(tmp_wf) = }")
log0.warning(f"{tmp_wf.shape = }")
log0.warning(f"{times.shape = }")
log0.warning(f"{times[:20] = }")
"""
plt.close('all')
"""
plt.plot(times, tmp_wf)

# Check Events Generation

""" OPTIONS:
event_labels = ["Ev01", "Ev02"]
event_labels = 3
"""
event_labels = ["Ev01", "Ev02"]

temp_events, temp_event_IDs, temp_event_desc, temp_events_df = GTE().make_dummy_events(
    event_labels=event_labels,
    event_repets=100,
    event_interv=2000,
    event_begins=5000,
)

ddf(temp_events_df.head(n=4))
ddf(temp_events_df.tail(n=4))

# Init: Get Singularity Event(s)

gte.make_singularity_events()

ddf(gte.singularity_events_df)

# Init: Get Experimental Events

""" OPTIONS:
event_labels = ["Ev01", "Ev02"]
event_labels = 3
"""
event_labels = 3
gte.make_experimental_events(event_labels = event_labels)

print(f"{gte.experimental_events_df.shape = }")
ddf(gte.experimental_events_df.event_labels.value_counts().sort_index())
ddf(gte.experimental_events_df.head(n=4))
ddf(gte.experimental_events_df.tail(n=4))

# Randomized Activations

gte.set_randomized_activations(
    num_labels=9,
    num_labels_per_event=2,
    event_labels=2,
)
print(srsly.yaml_dumps(gte.activ0))
print(srsly.yaml_dumps(gte.activ0_labels))
print(srsly.yaml_dumps(gte.activ0_events))
ddf(gte.activations_to_dataframe())
print("\n=========== UPDATED EVENTS 4 ===========\n")
print(f"{gte.experimental_events_df.shape = }")
ddf(gte.experimental_events_df.event_labels.value_counts().sort_index())
ddf(gte.experimental_events_df.head(n=4))
ddf(gte.experimental_events_df.tail(n=4))
print("\n=========== UPDATED LABELS 4 ===========\n")
print(f"{len(gte.labels4) = }")
print(f"{gte.labels4 = }")
print(f"{gte.label4_names = }")

# Predefined Activations

gte.set_predefined_activations()
print(srsly.yaml_dumps(gte.activ0))
print(srsly.yaml_dumps(gte.activ0_labels))
print(srsly.yaml_dumps(gte.activ0_events))
ddf(gte.activations_to_dataframe())
print("\n=========== UPDATED EVENTS ===========\n")
print(f"{gte.experimental_events_df.shape = }")
ddf(gte.experimental_events_df.event_labels.value_counts().sort_index())
ddf(gte.experimental_events_df.head(n=4))
ddf(gte.experimental_events_df.tail(n=4))
print("\n=========== UPDATED LABELS 4 ===========\n")
print(f"{len(gte.labels4) = }")
print(f"{gte.labels4 = }")
print(f"{gte.label4_names = }")

# Check Activity Labels

gte.labels2[0].name

# Number of Samples in Trial

# print(f"{gte.activ0_trial_num_samp = }")
print(f"{gte.activ0_trial_samp_total = }")

# Times

print(f"{len(gte.activ0_trial_times) = }")
print(f"{gte.activ0_trial_times[:5] = }")

# Initialize Source Simulator

gte.initialize_source_simulator()
print(gte.source_simulator)

# Add data to Source Simulator

gte.add_data_to_source_simulator()

# Get Source Time Course

gte.extract_activ0_stc()
gte.activ0_stc

# Check Source Time Course

gte.activ0_stc

# Generate Raw Data

gte.extract_activ0_raw()

# Plot Raw

if False:
  gte.activ0_raw.plot(duration=10.0, start=0.0)

# Add Noise to Raw

gte.extract_activ2_raw()  # TODO add noise parameters

# Plot Noisy Raw

gte.activ2_raw.plot(duration=10.0, start=0.0)

# Add Evoked and Epoched Data

gte.extract_activ2_epochs_and_evoked()

# Plot Epochs

print(gte.activ2_epochs.event_id.keys())
gte.activ2_epochs.plot()

# Plot evoked

ev = list(gte.activ2_epochs.event_id.keys())[0]
gte.activ2_evoked[ev].plot(spatial_colors=True)
gte.activ2_evoked[ev].plot_image()

# Compute Covariances

gte.compute_covariances(
    data_tmin = 0.01,
    data_tmax = 0.60,
    noise_tmin = None,
    noise_tmax = 0,
    method = "empirical")

# Plot Covariances

gte.activ2_data_cov.plot(gte.activ2_epochs.info)
gte.activ2_noise_cov.plot(gte.activ2_epochs.info)
gte.activ2_common_cov.plot(gte.activ2_epochs.info)

# Filters

gte.compute_lcmv_bf_filters()

# Apply LCMV Filters

gte.apply_lcmv_bf_filters()

# Plot Activity

idx0 = 0
idx0 = 1
ev = list(gte.activ2_epochs.event_id.keys())[idx0]

log0.warning(f"{gte.stcs[ev].shape = }")
gte.stcs[ev].plot(
    # hemi="rh",
    hemi="split",
    subjects_dir=gte.subjects_dir,
    subject=gte.subject,
    views=["lat", "med"],
    time_label="LCMV source power in the 12-30 Hz frequency band",
)

# Final Checkups

ddf(gte.list_properties())

# Publish
# #+header: :async yes
# #+header: :eval yes
# #+header: :eval no
# #+header: :eval query

gte.publish()

# MNE Plot: Source Space

gte.src0.plot(
    head=False,
    brain=None,
    skull=False,
    trans=gte.trans0,
    subjects_dir=gte.subjects_dir)

# MNE Plot: Alignment

"""
mne.viz.close_all_3d_figures()
"""
fig = mne.viz.create_3d_figure(size=(600, 400), bgcolor=(0.00, 0.00, 0.00))
src = None
src = gte.src0   # get source positions
fwd = gte.fwd0
fwd = None  # get no quivers
surfaces = ["white", "head"]
mne.viz.plot_alignment(
    info=gte.info0,
    trans=gte.trans0,
    subject=gte.subject,
    subjects_dir=gte.subjects_dir,
    surfaces=surfaces,
    coord_frame="mri",
    meg=(),
    eeg=dict(original=0.2, projected=0.8),
    fwd=fwd,
    dig=False,
    ecog=False,
    src=src,
    bem=gte.bem_model0,
    mri_fiducials=True,
    seeg=False,
    fnirs=False,
    show_axes=True,
    dbs=False,
    fig=fig,
    interaction="terrain",
    sensor_colors="magenta",
    verbose=True)

# MNE Plot: Brain, Head, Montage and Sources

hemi = "lh"
hemi = "rh"
hemi = "split"
hemi = "both"

surf = "inflated"
surf = "pial"
surf = "white"

cortex = "high_contrast"
cortex = "low_contrast"
cortex = "classic"

Brain = mne.viz.get_brain_class()
brain = Brain(
    subject=gte.subject,
    hemi=hemi,
    surf=surf,
    cortex=cortex,
    subjects_dir=gte.subjects_dir,
    alpha=0.4,
    size=(800, 600),
)
brain.add_annotation(gte.annot0, borders=False, alpha=1.0)
brain.add_sensors(info=gte.info0, trans=gte.trans0, eeg=dict(original=0.2, projected=1.0))
brain.add_forward(fwd=gte.fwd0, trans=gte.trans0)
brain.add_head(dense=True, color="white", alpha=0.15)
type(brain)
