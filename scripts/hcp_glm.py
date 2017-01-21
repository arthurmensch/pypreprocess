"""
Run GLM on HCP data using nistats.
"""
# Author: Elvis Dohmatob

import os
import time
import glob
import numpy as np
import nibabel
from nilearn.image import index_img
from sklearn.externals.joblib import Memory, Parallel, delayed
from pypreprocess.external.nistats.glm import FirstLevelGLM
from pypreprocess.reporting.base_reporter import ProgressReport
from pypreprocess.reporting.glm_reporter import generate_subject_stats_report
from pypreprocess.fsl_to_nistats import (read_fsl_design_file,
                                         make_dmtx_from_timing_files)

# from pypreprocess.reporting.glm_reporter import group_one_sample_t_test

# config
n_subjects = int(os.environ.get("N_SUBJECTS", 900))
root = os.environ.get("ROOT", "/")
output_dir = os.path.join(root, "home/arthur/output")
tr = .72
hrf_model = "spm + derivative"
drift_model = "Cosine"
hfcut = 200.
# cons = [
#     "0BK-2BK", "PLACE-AVG", "FACE-AVG", "TOOL-AVG", "BODY-AVG",
#     "LH-RH", "LF-RF", "T-AVG",
#     "FACES-SHAPES",
#     "TOM-RANDOM",
#     "MATH-STORY"]
cons = ["LH-RH", "RH-LH", "RF-LF", "LF-RF", "T-AVG"]


def do_subject_glm(subject_dir, task, directions=None,
                   report=True):
    subject_id = os.path.basename(subject_dir)
    stats_start_time = time.ctime()
    if directions is None:
        directions = ['LR', 'RL']
    subject_output_base_dir = os.path.join(output_dir, subject_id)
    memory = Memory(os.path.join(output_dir, "cache_dir", subject_id))
    # the actual GLM stuff
    for direction in directions:
        subject_output_dir = os.path.join(subject_output_base_dir, direction)
        if not os.path.exists(subject_output_dir):
            os.makedirs(subject_output_dir)
        fmri_file = os.path.join(subject_dir,
                                 "MNINonLinear/Results/",
                                 "tfMRI_%s_%s/tfMRI_%s_%s_hp200_s4_level1.feat/"
                                 "tfMRI_%s_%s_hp200_s4.nii.gz" % (
                                     task, direction, task, direction,
                                     task, direction))
        design_file = os.path.join(subject_dir,
                                   "MNINonLinear/Results/tfMRI_%s_%s/",
                                   "tfMRI_%s_%s_hp200_s4_level1.fsf"
                                   % (task, direction, task, direction))

        if not os.path.isfile(design_file):
            print("Can't find design file %s; skipping subject %s" % (
                design_file, subject_id))
            return
        if not os.path.exists(fmri_file):
            print("File %s is missing; skipping subject %s ..." % (fmri_file,
                                                                   subject_id))
            return

        # read the experimental setup
        print("Reading experimental setup from %s ..." % design_file)
        fsl_condition_ids, timing_files, fsl_contrast_ids, contrast_values = \
            read_fsl_design_file(design_file)

        # fix timing filenames
        timing_files = [tf.replace("EVs", "tfMRI_%s_%s/EVs" % (
            task, direction)) for tf in timing_files]

        # make design matrix
        print("Constructing design matrix for direction %s ..." % direction)
        n_scans = nibabel.load(fmri_file).shape[-1]
        design_matrix, paradigm, frametimes = make_dmtx_from_timing_files(
            timing_files, fsl_condition_ids, n_scans=n_scans, tr=tr,
            hrf_model=hrf_model, drift_model=drift_model, period_cut=hfcut)

        # convert contrasts to dict
        contrasts = dict((contrast_id,
                          # append zeros to end of contrast to match design
                          np.hstack((contrast_value, np.zeros(len(
                              design_matrix.columns) - len(contrast_value)))))

                         for contrast_id, contrast_value in zip(
            fsl_contrast_ids, contrast_values))

        print(('Fitting a "Fixed Effect" GLM for merging LR and RL '
               'phase-encoding directions for subject %s (%s task)...' % (
                   subject_id, task)))
        fmri_glm = FirstLevelGLM(memory=memory, smoothing_fwhm=0, )
        fmri_glm.fit(fmri_file, design_matrix)

        # save computed mask
        mask_path = os.path.join(subject_output_dir, "mask.nii")
        print("Saving mask image to %s ..." % mask_path)
        fmri_glm.masker_.mask_img_.to_filename(mask_path)

        # do the actual model fit
        z_maps = {}
        effects_maps = {}
        map_dirs = {}
        for contrast_id, contrast_val in contrasts.items():
            print("\tcontrast id: %s" % contrast_id)
            z_map, eff_map = fmri_glm.transform(
                contrast_val, contrast_name=contrast_id, output_z=True,
                output_effects=True)

            # store stat maps to disk
            for map_type, out_map in zip(['z', 'effects'],
                                         [z_map, eff_map]):
                map_dir = os.path.join(
                    subject_output_dir, '%s_maps' % map_type)
                map_dirs[map_type] = map_dir
                if not os.path.exists(map_dir):
                    os.makedirs(map_dir)
                map_path = os.path.join(map_dir, '%s_%s.nii' % (map_type,
                                                                contrast_id))
                print("\t\tWriting %s ..." % map_path)
                nibabel.save(out_map, map_path)

                # collect zmaps for contrasts we're interested in
                if map_type == 'z':
                    z_maps[contrast_id] = map_path

                if map_type == 'effects':
                    effects_maps[contrast_id] = map_path

        if report:
            stats_report_filename = os.path.join(subject_output_dir, "reports",
                                                 "report_stats.html")
            generate_subject_stats_report(
                stats_report_filename, contrasts, z_maps,
                fmri_glm.masker_.mask_img_, threshold=2.3, cluster_th=15,
                design_matrices=design_matrix, TR=tr, subject_id=subject_id,
                start_time=stats_start_time, n_scans=n_scans, paradigm=paradigm,
                frametimes=frametimes, drift_model=drift_model, hfcut=hfcut,
                title="GLM for subject %s" % subject_id, hrf_model=hrf_model)
            ProgressReport().finish_dir(subject_output_dir)
            print("Statistic report written to %s\r\n" % stats_report_filename)

    print("Done (subject %s)" % subject_id)
    # return dict(subject_id=subject_id, mask=mask_path,
    #             effects_maps=effects_maps, z_maps=z_maps,
    #             contrasts=contrasts)


if __name__ == "__main__":
    # get subjects to process
    subject_dirs = sorted(glob.glob(os.path.join(
        root, "home/arthur/data/modl_data/HCP/??????")))[:n_subjects]
    n_jobs = 1

    for task in ["EMOTION"]:
        # run first-level GLM
        first_levels = Parallel(n_jobs=n_jobs)(delayed(do_subject_glm)(
            subject_dir, task, cons, smoothing_fwhm=0, report=False)
                                               for subject_dir in subject_dirs)
        # first_levels = [x for x in first_levels if x is not None]
        # print(task, len(first_levels))

        # # run second-level GLM
        # output_dir = os.path.join(data_dir, "GLM%s" % (
        #     ["DC+SBRef", "DC+LoG"][pipeline]))
        # mem.cache(group_one_sample_t_test)(
        #     [subject_data["mask"] for subject_data in first_levels],
        #     [subject_data["effects_maps"] for subject_data in first_levels],
        #     first_levels[0]["contrasts"],
        #     output_dir)
