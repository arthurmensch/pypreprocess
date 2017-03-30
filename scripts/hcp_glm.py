"""
Run GLM on HCP data using nistats.
"""
# Author: Elvis Dohmatob, Arthur Mensch

import os
import re
from copy import copy
from os.path import join

import nibabel
import numpy as np
import pandas as pd
from hcp_builder.utils import get_data_dirs
from nilearn._utils import check_niimg
from nilearn.image import new_img_like
from nistats.first_level_model import FirstLevelModel
from nistats.second_level_model import SecondLevelModel
from sklearn.externals.joblib import Memory, Parallel, delayed

# regex for contrasts
CON_REAL_REGX = ("set fmri\(con_real(?P<con_num>\d+?)\.(?P<ev_num>\d+?)\)"
            " (?P<con_val>\S+)")

# regex for "Number of EVs"
NUM_EV_REGX = """set fmri\(evs_orig\) (?P<evs_orig>\d+)
set fmri\(evs_real\) (?P<evs_real>\d+)
set fmri\(evs_vox\) (?P<evs_vox>\d+)"""

# regex for "Number of contrasts"
NUM_CON_REGX = """set fmri\(ncon_orig\) (?P<ncon>\d+)
set fmri\(ncon_real\) (?P<ncon_real>\d+)"""

# regex for "# EV %i title"
EV_TITLE_REGX = """set fmri\(evtitle\d+?\) \"(?P<evtitle>.+)\""""

# regex for "Title for contrast_real %i"
CON_TITLE_REGX = """set fmri\(conname_real\.\d+?\) \"(?P<conname_real>.+)\""""

# regex for "Basic waveform shape (EV %i)"
# 0 : Square
# 1 : Sinusoid
# 2 : Custom (1 entry per volume)
# 3 : Custom (3 column format)
# 4 : Interaction
# 10 : Empty (all zeros)
EV_SHAPE_REGX = """set fmri\(shape\d+\) (?P<shape>[0|1|3])"""

# regex for "Custom EV file (EV %i)"
EV_CUSTOM_FILE_REGX = """set fmri\(custom\d+?\) \"(?P<custom>.+)\""""


def _get_abspath_relative_to_file(filename, ref_filename):
    """
    Returns the absolute path of a given filename relative to a reference
    filename (ref_filename).

    """

    # we only handle files
    assert os.path.isfile(ref_filename)

    old_cwd = os.getcwd()  # save CWD
    os.chdir(os.path.dirname(ref_filename))  # we're in context now
    abspath = os.path.abspath(filename)  # bing0!
    os.chdir(old_cwd)  # restore CWD

    return abspath


def read_fsl_design_file(design_filename):
    """
    Scrapes an FSL design file for the list of contrasts.

    Returns
    -------
    conditions: list of n_conditions strings
        condition (EV) titles

    timing_files: list of n_condtions strings
        absolute paths of files containing timing info for each condition_id

    contrast_ids: list of n_contrasts strings
        contrast titles

    contrasts: 2D array of shape (n_contrasts, n_conditions)
        array of contrasts, one line per contrast_id; one column per
        condition_id

    Raises
    ------
    AssertionError or IndexError if design_filename is corrupt (not in
    official FSL format)

    """

    # read design file
    design_conf = open(design_filename, 'r').read()

    # scrape n_conditions and n_contrasts
    n_conditions_orig = int(re.search(NUM_EV_REGX,
                                      design_conf).group("evs_orig"))
    n_conditions = int(re.search(NUM_EV_REGX, design_conf).group("evs_real"))
    n_contrasts = int(re.search(NUM_CON_REGX, design_conf).group("ncon_real"))

    # initialize 2D array of contrasts
    contrasts = np.zeros((n_contrasts, n_contrasts))

    # lookup EV titles
    conditions = [item.group("evtitle") for item in re.finditer(
                  EV_TITLE_REGX, design_conf)]
    assert len(conditions) == n_conditions_orig

    # lookup contrast titles
    contrast_ids = [item.group("conname_real")for item in re.finditer(
                    CON_TITLE_REGX, design_conf)]
    assert len(contrast_ids) == n_contrasts

    # # lookup EV (condition) shapes
    # condition_shapes = [int(item.group("shape")) for item in re.finditer(
    #         EV_SHAPE_REGX, design_conf)]
    # print(condition_shapes)

    # lookup EV (condition) custom files
    timing_files = [_get_abspath_relative_to_file(item.group("custom"),
                                                  design_filename)
                    for item in re.finditer(EV_CUSTOM_FILE_REGX, design_conf)]

    # lookup the contrast values
    count = 0
    for item in re.finditer(CON_REAL_REGX, design_conf):
        count += 1
        value = float(item.group('con_val'))

        i = int(item.group('con_num')) - 1
        j = int(item.group('ev_num')) - 1

        # roll-call
        assert 0 <= i < n_contrasts, item.group()
        assert 0 <= j < n_conditions, item.group()

        contrasts[i, j] = value

    # roll-call
    assert count == n_contrasts * n_conditions, count

    return conditions, timing_files, list(zip(contrast_ids, contrasts))


def make_paradigm_from_timing_files(timing_files, trial_types=None):
    if not trial_types is None:
        assert len(trial_types) == len(timing_files)

    onsets = []
    durations = []
    amplitudes = []
    curated_trial_types = []
    count = 0
    for timing_file in timing_files:
        timing = np.loadtxt(timing_file)
        if timing.ndim == 1:
            timing = timing[np.newaxis, :]

        if trial_types is None:
            trial_type = os.path.basename(timing_file).lower(
                ).split('.')[0]
        else:
            trial_type = trial_types[count]
        curated_trial_types += [trial_type] * timing.shape[0]

        count += 1

        if timing.shape[1] == 3:
            onsets += list(timing[..., 0])
            durations += list(timing[..., 1])
            amplitudes += list(timing[..., 2])
        elif timing.shape[1] == 2:
            onsets += list(timing[..., 0])
            durations += list(timing[..., 1])
            amplitudes = durations + list(np.ones(len(timing)))
        elif timing.shape[1] == 1:
            onsets += list(timing[..., 0])
            durations += list(np.zeros(len(timing)))
            amplitudes = durations + list(np.ones(len(timing)))
        else:
            raise TypeError(
                "Timing info must either be 1D array of onsets of 2D "
                "array with 2 or 3 columns: the first column is for "
                "the onsets, the second for the durations, and the "
                "third --if present-- if for the amplitudes; got %s" % timing)

    return pd.DataFrame({'trial_type': curated_trial_types,
                         'onset': onsets,
                         'duration': durations,
                         'modulation': amplitudes})


def run_glm(subject, task):
    hrf_model = "spm + derivative"
    drift_model = "polynomial"
    drift_order = 2
    subject = str(subject)
    subject_data_dir = join(get_data_dirs()[0], subject,
                            'MNINonLinear', 'Results')
    output_dir = join(get_data_dirs()[0], 'nistats', subject)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    memory = Memory(os.path.join(output_dir, "cache_dir", subject))
    directions = ['RL', 'LR']
    models = {}
    model_contrasts = {}
    # the actual GLM stuff
    for direction in directions:
        fmri_file = os.path.join(subject_data_dir,
                                 "tfMRI_%s_%s/tfMRI_%s_%s.nii.gz" % (
                                     task, direction, task, direction))
        mask_file = check_niimg(join(subject_data_dir,
                                     "tfMRI_%s_%s/tfMRI_%s_%s_SBRef.nii.gz" % (
                                         task, direction, task, direction)))
        mask_file = new_img_like(mask_file,
                                 mask_file.get_data() != 0, copy_header=False)
        design_file = os.path.join(subject_data_dir,
                                   "tfMRI_%s_%s/tfMRI_%s_%s_hp200_s4_level1.fsf"
                                   % (task, direction, task, direction))
        # read the experimental setup
        print("Reading experimental setup from %s ..." % design_file)
        trial_types, timing_files, contrasts = read_fsl_design_file(
            design_file)

        # Pad contrast with 1, for subject id
        for i, (contrast_name, contrast_val) in enumerate(contrasts):
            contrast_val = np.hstack([contrast_val, np.zeros(1)])
            contrasts[i] = (contrast_name, contrast_val)

        model_contrasts[direction] = contrasts

        # fix timing filenames as we load the fsl file one directory
        # higher than expected
        timing_files = [tf.replace("EVs", "tfMRI_%s_%s/EVs" % (
            task, direction)) for tf in timing_files]

        # make design matrix
        print("Constructing design matrix for direction %s ..." % direction)
        events = make_paradigm_from_timing_files(timing_files,
                                                 trial_types=trial_types)
        # convert contrasts to dict
        order1_model = FirstLevelModel(memory=memory, mask=mask_file,
                                       smoothing_fwhm=4,
                                       standardize=True,
                                       signal_scaling=False,
                                       t_r=.72,
                                       hrf_model=hrf_model,
                                       drift_model=drift_model,
                                       drift_order=drift_order,
                                       subject_id=direction)
        order1_model.fit(fmri_file, events)
        models[direction] = order1_model
    contrasts = copy(model_contrasts['RL'])
    order2_model = SecondLevelModel(memory=memory,
                                    smoothing_fwhm=4)
    order2_model.fit([models['RL'], models['LR']],
                     first_level_conditions=contrasts)
    models['level2'] = order2_model

    # Padd with an extra 0 to contruct level 2 contrast
    for i, (contrast_name, contrast_val) in enumerate(contrasts):
        contrast_val = np.hstack([contrast_val, np.zeros(1)])
        contrasts[i] = (contrast_name, contrast_val)
    model_contrasts['level2'] = contrasts

    for model_type in models:
        model = models[model_type]
        contrasts = model_contrasts[model_type]
        model_output_dir = join(output_dir, model_type)
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        for contrast_name, contrast_val in contrasts:
            print("\tContrast: %s" % contrast_name)
            # save computed mask
            mask_path = os.path.join(model_output_dir, "mask.nii.gz")
            print("Saving mask image to %s ..." % mask_path)
            model.masker_.mask_img_.to_filename(mask_path)
            z_map = model.compute_contrast(
                contrast_val, output_type='z_score')
            eff_map = model.compute_contrast(
                contrast_val, output_type='effect_size')
            # store stat maps to disk
            for map_type, out_map in zip(['z', 'effects'], [z_map, eff_map]):
                map_dir = os.path.join(model_output_dir, '%s_maps' % map_type)
                if not os.path.exists(map_dir):
                    os.makedirs(map_dir)
                map_path = os.path.join(map_dir, '%s_%s.nii.gz' % (map_type,
                                                                   contrast_name))
                print("\t\tWriting %s ..." % map_path)
                nibabel.save(out_map, map_path)

    print("Done (subject %s)" % subject)


if __name__ == "__main__":
    # get subjects to process

    subjects = [100307]
    tasks = ['EMOTION']

    Parallel(n_jobs=1)(delayed(run_glm)(
        subject_dir, task) for subject_dir in subjects
                       for task in tasks)
