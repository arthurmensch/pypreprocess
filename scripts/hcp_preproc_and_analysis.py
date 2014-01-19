"""
:Synopsis: preprocessing and/or analysis of HCP task fMRI data
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

import os
import sys
import re
import glob
import numpy as np
import nibabel
import commands
from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.labs.mask import intersect_masks
from pypreprocess.nipype_preproc_spm_utils import (SubjectData,
                                                   _do_subject_realign,
                                                   _do_subject_smooth,
                                                   do_subject_preproc)
from pypreprocess.io_utils import load_specific_vol
from pypreprocess.fsl_to_nipy import (read_design_fsl_design_file,
                                      make_dmtx_from_timing_files,
                                      _insert_directory_in_file_name)
from pypreprocess.reporting.glm_reporter import generate_subject_stats_report
from pypreprocess.reporting.base_reporter import (ProgressReport,
                                                  pretty_time
                                                  )
from pypreprocess.conf_parser import _generate_preproc_pipeline
from joblib import Parallel, delayed, Memory


def _do_fmri_distortion_correction(subject_data,
                                   # i'm unsure of the readout time,
                                   # but this is constant across both PE
                                   # directions and so can be scaled to 1
                                   # (or any other nonzero float)
                                   readout_time=.01392,
                                   coreg_func_to_anat=True,
                                   segment=False,
                                   normalize=False,
                                   func_write_voxel_sizes=None,
                                   anat_write_voxel_sizes=None,
                                   report=True,
                                   **kwargs
                                   ):
    """
    Function to undistort task fMRI data for a given HCP subject.

    """

    directions = ['LR', 'RL']

    subject_data.sanitize()

    # prepare for smart caching
    mem = Memory(os.path.join(subject_data.output_dir, "cache_dir"))

    acq_params = [[1, 0, 0, readout_time], [-1, 0, 0, readout_time]]
    acq_params_file = os.path.join(subject_data.output_dir,
                                   "b0_acquisition_params.txt")
    np.savetxt(acq_params_file, acq_params, fmt='%f')

    fieldmap_files = [os.path.join(os.path.dirname(subject_data.func[sess]),
                                   "%s_3T_SpinEchoFieldMap_%s.nii.gz" % (
                subject_data.subject_id, directions[sess]))
                      for sess in xrange(subject_data.n_sessions)]
    sbref_files = [sess_func.replace(".nii", "_SBRef.nii")
                   for sess_func in subject_data.func]
    subject_data.design_files = [sess_func.replace(
            ".nii.gz", "_hp200_s4_level1.fsf").replace(
            "%s_3T_" % subject_data.subject_id, "")
                                 for sess_func in subject_data.func]

    for x in [fieldmap_files, sbref_files, subject_data.func]:
        assert len(x) == 2
        for y in x:
            assert os.path.isfile(y), y

    # fslroi
    zeroth_fieldmap_files = []
    for fieldmap_file in fieldmap_files:
        if not os.path.isfile(fieldmap_file):
            print "Can't find fieldmap file %s; skipping subject %s" % (
                fieldmap_file, subject_data.subject_id)
            return

        # peel 0th volume of each fieldmap
        zeroth_fieldmap_file = os.path.join(
            subject_data.output_dir, "0th_%s" % os.path.basename(
                fieldmap_file))
        fslroi_cmd = "fsl5.0-fslroi %s %s 0 1" % (
            fieldmap_file, zeroth_fieldmap_file)
        print "\r\nExecuting '%s' ..." % fslroi_cmd
        print mem.cache(commands.getoutput)(fslroi_cmd)

        zeroth_fieldmap_files.append(zeroth_fieldmap_file)

    # merge the 0th volume of both fieldmaps
    merged_zeroth_fieldmap_file = os.path.join(
        subject_data.output_dir, "merged_with_other_direction_%s" % (
            os.path.basename(zeroth_fieldmap_files[0])))
    fslmerge_cmd = "fsl5.0-fslmerge -t %s %s %s" % (
        merged_zeroth_fieldmap_file, zeroth_fieldmap_files[0],
        zeroth_fieldmap_files[1])
    print "\r\nExecuting '%s' ..." % fslmerge_cmd
    print mem.cache(commands.getoutput)(fslmerge_cmd)

    # do topup (learn distortion model)
    topup_results_basename = os.path.join(subject_data.output_dir,
                                          "topup_results")
    topup_cmd = (
        "fsl5.0-topup --imain=%s --datain=%s --config=b02b0.cnf "
        "--out=%s" % (merged_zeroth_fieldmap_file, acq_params_file,
                      topup_results_basename))
    print "\r\nExecuting '%s' ..." % topup_cmd
    print mem.cache(commands.getoutput)(topup_cmd)

    # apply learn deformations to absorb distortion
    dc_fmri_files = []

    for sess in xrange(2):
        # merge SBRef + task BOLD for current PE direction
        assert len(subject_data.func) == 2, subject_data
        fourD_plus_sbref = os.path.join(
            subject_data.output_dir, "sbref_plus_" + os.path.basename(
                subject_data.func[sess]))
        fslmerge_cmd = "fsl5.0-fslmerge -t %s %s %s" % (
            fourD_plus_sbref, sbref_files[sess], subject_data.func[sess])
        print "\r\nExecuting '%s' ..." % fslmerge_cmd
        print mem.cache(commands.getoutput)(fslmerge_cmd)

        # realign task BOLD to SBRef
        sess_output_dir = subject_data.session_output_dirs[sess]
        rfourD_plus_sbref = _do_subject_realign(SubjectData(
                func=[fourD_plus_sbref], output_dir=subject_data.output_dir,
                n_sessions=1, session_output_dirs=[sess_output_dir]),
                                           report=False).func[0]

        # apply topup to realigned images
        dc_rfourD_plus_sbref = os.path.join(
            subject_data.output_dir, "dc" + os.path.basename(
                rfourD_plus_sbref))
        applytopup_cmd = (
            "fsl5.0-applytopup --imain=%s --verbose --inindex=%i "
            "--topup=%s --out=%s --datain=%s --method=jac" % (
                rfourD_plus_sbref, sess + 1, topup_results_basename,
                dc_rfourD_plus_sbref, acq_params_file))
        print "\r\nExecuting '%s' ..." % applytopup_cmd
        print mem.cache(commands.getoutput)(applytopup_cmd)

        # recover undistorted task BOLD
        dc_rfmri_file = dc_rfourD_plus_sbref.replace("sbref_plus_", "")
        fslroi_cmd = "fsl5.0-fslroi %s %s 1 -1" % (
            dc_rfourD_plus_sbref, dc_rfmri_file)
        print "\r\nExecuting '%s' ..." % fslroi_cmd
        print mem.cache(commands.getoutput)(fslroi_cmd)

        # sanity tricks
        if dc_rfmri_file.endswith(".nii"):
            dc_rfmri_file = dc_rfmri_file + ".gz"

        dc_fmri_files.append(dc_rfmri_file)

    subject_data.func = dc_fmri_files
    if isinstance(subject_data.func, basestring):
        subject_data.func = [subject_data.func]

    # continue preprocessing
    subject_data = do_subject_preproc(
        subject_data,
        coreg_anat_to_func=not coreg_func_to_anat,
        segment=segment,
        normalize=normalize,
        func_write_voxel_sizes=func_write_voxel_sizes,
        anat_write_voxel_sizes=anat_write_voxel_sizes,
        report=report)

    # ok for GLM now
    return subject_data


def run_suject_level1_glm(subject_data,
                          readout_time=.01392,  # seconds
                          tr=.72,
                          hrf_model="Canonical with Derivative",
                          drift_model="Cosine",
                          hfcut=100,
                          regress_motion=True,
                          slicer='ortho',
                          cut_coords=6,
                          threshold=3.,
                          cluster_th=15,
                          fwhm=0.,
                          task_id="MOTOR",
                          **other_preproc_kwargs
                          ):
    """
    Function to do preproc + analysis for a single HCP subject (task fMRI)

    """

    add_regs_files = None
    n_motion_regressions = 6

    if not os.path.exists(subject_data.output_dir):
        os.makedirs(subject_data.output_dir)

    subject_data = _do_fmri_distortion_correction(subject_data, fwhm=fwhm,
                                                  readout_time=readout_time,
                                                  **other_preproc_kwargs)
    # chronometry
    stats_start_time = pretty_time()

    # merged lists
    paradigms = []
    frametimes_list = []
    design_matrices = []
    # fmri_files = []
    n_scans = []
    # for direction, direction_index in zip(['LR', 'RL'], xrange(2)):
    for sess in xrange(subject_data.n_sessions):
        direction = ['LR', 'RL'][sess]
        # glob the design file
        # design_file = os.path.join(# _subject_data_dir, "tfMRI_%s_%s" % (
                # task_id, direction),
        design_file = subject_data.design_files[sess]
                #                    "tfMRI_%s_%s_hp200_s4_level1.fsf" % (
                # task_id, direction))
        if not os.path.isfile(design_file):
            print "Can't find design file %s; skipping subject %s" % (
                design_file, subject_data.subject_id)
            return

        # read the experimental setup
        print "Reading experimental setup from %s ..." % design_file
        fsl_condition_ids, timing_files, fsl_contrast_ids, contrast_values = \
            read_design_fsl_design_file(design_file)
        print "... done.\r\n"

        # fix timing filenames
        timing_files = _insert_directory_in_file_name(
            timing_files, os.path.basename(os.path.dirname(
                    subject_data.func[sess])), 1)

        # make design matrix
        print "Constructing design matrix for direction %s ..." % direction
        _n_scans = nibabel.load(subject_data.func[sess]).shape[-1]
        n_scans.append(_n_scans)
        add_regs_file = add_regs_files[
            sess] if not add_regs_files is None else None
        design_matrix, paradigm, frametimes = make_dmtx_from_timing_files(
            timing_files, fsl_condition_ids, n_scans=_n_scans, tr=tr,
            hrf_model=hrf_model, drift_model=drift_model, hfcut=hfcut,
            add_regs_file=add_regs_file,
            add_reg_names=[
                'Translation along x axis',
                'Translation along yaxis',
                'Translation along z axis',
                'Rotation along x axis',
                'Rotation along y axis',
                'Rotation along z axis',
                'Differential Translation along x axis',
                'Differential Translation along yaxis',
                'Differential Translation along z axis',
                'Differential Rotation along x axis',
                'Differential Rotation along y axis',
                'Differential Rotation along z axis'
                ][:n_motion_regressions] if not add_regs_files is None
            else None,
            )

        print "... done."
        paradigms.append(paradigm)
        frametimes_list.append(frametimes)
        design_matrices.append(design_matrix)

        # convert contrasts to dict
        contrasts = dict((contrast_id,
                          # append zeros to end of contrast to match design
                          np.hstack((contrast_value, np.zeros(len(
                                design_matrix.names) - len(contrast_value)))))

                         for contrast_id, contrast_value in zip(
                fsl_contrast_ids, contrast_values))

        # more interesting contrasts
        if task_id == 'MOTOR':
            contrasts['RH-LH'] = contrasts['RH'] - contrasts['LH']
            contrasts['LH-RH'] = -contrasts['RH-LH']
            contrasts['RF-LF'] = contrasts['RF'] - contrasts['LF']
            contrasts['LF-RF'] = -contrasts['RF-LF']
            contrasts['H'] = contrasts['RH'] + contrasts['LH']
            contrasts['F'] = contrasts['RF'] + contrasts['LF']
            contrasts['H-F'] = contrasts['RH'] + contrasts['LH'] - (
                contrasts['RF'] - contrasts['LF'])
            contrasts['F-H'] = -contrasts['H-F']

    # importat maps
    z_maps = {}
    effects_maps = {}
    # replicate contrasts across sessions
    contrasts = dict((cid, [cval] * 2)
                     for cid, cval in contrasts.iteritems())

    # compute effects
    mask_path = os.path.join(subject_data.output_dir, "mask.nii.gz")
    for contrast_id, contrast_val in contrasts.iteritems():
        for map_type in ['z', 'effects']:
            map_dir = os.path.join(
                subject_data.output_dir, '%s_maps' % map_type)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(
                map_dir, '%s.nii.gz' % contrast_id)

            # collect zmaps for contrasts we're interested in
            if map_type == 'z':
                z_maps[contrast_id] = map_path

            if map_type == 'effects':
                effects_maps[contrast_id] = map_path

    # fit GLM
    print (
        'Fitting a "Fixed Effect" GLM for merging LR and RL '
        'phase-encoding directions for subject %s ...' % (
            subject_data.subject_id))
    fmri_glm = FMRILinearModel(subject_data.func,
                               [design_matrix.matrix
                                for design_matrix in design_matrices],
                               mask='compute'
                               )
    fmri_glm.fit(do_scaling=True, model='ar1')
    print "... done.\r\n"

    # save computed mask
    mask_path = os.path.join(subject_data.output_dir, "mask.nii.gz")
    print "Saving mask image to %s ..." % mask_path
    nibabel.save(fmri_glm.mask, mask_path)
    print "... done.\r\n"

    # compute effects
    for contrast_id, contrast_val in contrasts.iteritems():
        print "\tcontrast id: %s" % contrast_id
        z_map, eff_map = fmri_glm.contrast(
            contrast_val,
            con_id=contrast_id,
            output_z=True,
            output_effects=True
            )

        # store stat maps to disk
        for map_type, out_map in zip(['z', 'effects'],
                                     [z_map, eff_map]):
            map_dir = os.path.join(
                subject_data.output_dir, '%s_maps' % map_type)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(
                map_dir, '%s.nii.gz' % contrast_id)
            print "\t\tWriting %s ..." % map_path
            nibabel.save(out_map, map_path)

            # collect zmaps for contrasts we're interested in
            if map_type == 'z':
                z_maps[contrast_id] = map_path

            if map_type == 'effects':
                effects_maps[contrast_id] = map_path

    # remove repeated contrasts
    contrasts = dict((cid, cval[0]) for cid, cval in contrasts.iteritems())

    # do stats report
    if 0x1:
        anat_img = nibabel.load(subject_data.anat)
        stats_report_filename = os.path.join(subject_data.output_dir,
                                             "reports",
                                             "report_stats.html")
        generate_subject_stats_report(
            stats_report_filename,
            contrasts,
            z_maps,
            nibabel.load(mask_path),
            anat=anat_img.get_data(),
            anat_affine=anat_img.get_affine(),
            threshold=threshold,
            cluster_th=cluster_th,
            slicer=slicer,
            cut_coords=cut_coords,
            design_matrices=design_matrices,
            subject_id=subject_data.subject_id,
            start_time=stats_start_time,
            title="GLM for subject %s" % subject_data.subject_id,

            # additional ``kwargs`` for more informative report
            TR=tr,
            n_scans=n_scans,
            hfcut=hfcut,
            drift_model=drift_model,
            hrf_model=hrf_model,
            paradigm={'LR': paradigms[0].__dict__,
                      'RL': paradigms[1].__dict__},
            frametimes={'LR': frametimes_list[0], 'RL': frametimes_list[1]},
            fwhm=fwhm
            )

        ProgressReport().finish_dir(subject_data.output_dir)
        print "\r\nStatistic report written to %s\r\n" % stats_report_filename

    return contrasts, effects_maps, z_maps, mask_path

if __name__ == '__main__':
    ###########################################################################
    # CONFIGURATION
    n_jobs = int(os.environ.get('N_JOBS', -1))
    task_ids = ['MOTOR',
                'WM',
                'LANGUAGE',
                'EMOTION',
                'GAMBLING',
                'RELATIONAL',
                'SOCIAL']
    slicer = 'ortho'  # slicer of activation maps QA
    threshold = 3.
    cluster_th = 15  # minimum number of voxels in reported clusters

    ####################################
    # read input configuration
    conf_file = sys.argv[1]

    # GO!
    for task_id in task_ids:
        subjects, preproc_params = _generate_preproc_pipeline(conf_file,
                                                              protocol=task_id)
        fwhm = preproc_params.get("fwhm")
        task_output_dir = os.path.join(os.path.dirname(subjects[0].output_dir))
        kwargs = {"regress_motion": True,
                  "slicer": slicer,
                  "threshold": threshold,
                  "cluster_th": cluster_th,
                  "task_id": task_id
                  }
        kwargs.update(preproc_params)
        if n_jobs > 1:
            group_glm_inputs = Parallel(
                n_jobs=n_jobs, verbose=100)(delayed(
                    run_suject_level1_glm)(
                        subject_data,
                        **kwargs) for subject_data in subjects)
        else:
            group_glm_inputs = [run_suject_level1_glm(
                    subject_data,
                    **kwargs) for subject_data in subjects]

        # level 2
        stats_start_time = pretty_time()
        normalize = preproc_params.get("normalize", False)
        if normalize:
            mask_images = [subject_glm_results[3]
                           for subject_glm_results in group_glm_inputs]
            group_mask = nibabel.Nifti1Image(
                intersect_masks(mask_images).astype(np.int8),
                nibabel.load(mask_images[0]).get_affine())
            print "... done.\r\n"
            print "Group GLM"
            contrasts = [
                subject_glm_results
                for subject_glm_results in group_glm_inputs]
            contrasts = group_glm_inputs[0][0]
            sujects_effects_maps = [
                subject_glm_results[1]
                for subject_glm_results in group_glm_inputs]
            group_level_z_maps = {}
            design_matrix = np.ones(len(sujects_effects_maps)
                                    )[:, np.newaxis]  # only the intercept
            for contrast_id in contrasts:
                print "\tcontrast id: %s" % contrast_id

                # effects maps will be the input to the second level GLM
                first_level_image = nibabel.concat_images(
                    [x[contrast_id] for x in sujects_effects_maps])

                # fit 2nd level GLM for given contrast
                group_model = FMRILinearModel(first_level_image,
                                            design_matrix, group_mask)
                group_model.fit(do_scaling=False, model='ols')

                # specify and estimate the contrast
                contrast_val = np.array(([[1.]])
                                        )  # the only possible contrast !
                z_map, = group_model.contrast(
                    contrast_val,
                    con_id='one_sample %s' % contrast_id,
                    output_z=True)

                # save map
                map_dir = os.path.join(task_output_dir, 'z_maps')
                if not os.path.exists(map_dir):
                    os.makedirs(map_dir)
                map_path = os.path.join(map_dir, '2nd_level_%s.nii.gz' % (
                        contrast_id))
                print "\t\tWriting %s ..." % map_path
                nibabel.save(z_map, map_path)

                group_level_z_maps[contrast_id] = map_path

            # do stats report
            stats_report_filename = os.path.join(
                task_output_dir, "reports", "report_stats.html")
            generate_subject_stats_report(
                stats_report_filename,
                contrasts,
                group_level_z_maps,
                group_mask,
                threshold=threshold,
                cluster_th=cluster_th,
                design_matrices=[design_matrix],
                subject_id="sub001",
                start_time=stats_start_time,
                title='Group GLM for HCP fMRI %s task' % task_id,
                slicer=slicer,
                cut_coords=cut_coords
                )

            ProgressReport().finish_dir(task_output_dir)
            print "\r\nStatistic report written to %s\r\n" % (
                stats_report_filename)
