# Setup logger
import logging
import os
import sys
from time import time

import numpy as np
import scipy.signal.windows as windows
from scipy.interpolate import interp1d
from mpi4py import MPI

from meer21cm import MockSimulation
sys.path.append("../specs")
from specs_v2 import *

TAG_TASK = 1
TAG_DONE = 2
TAG_TERMINATE = 3

def _prepare_logging(rank):
    logger = logging.getLogger(f"rank{rank}")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(f"[Rank {rank:02d} %(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

def get_powerspectra(seed, logger):
    logger.debug(f"seed {seed}")
    mock = MockSimulation(
        hp_nside=128,
        nu=nu_arr,
        ra_range = ra_range,
        dec_range = dec_range,
        seed=seed,
        downres_factor_radial=sim_upres_radial,
        downres_factor_transverse=sim_upres_transverse,
        batch_number=1,
        discrete_source_dndz=z_func,
        tracer_bias_2=1.5,
        tracer_bias_1=1.5,
        sigma_v_1= 100, # in velocity units
        sigma_v_2= 100,
        mean_amp_1="average_hi_temp",
        omega_hi=5e-4,
        sigma_beam_ch=sigma_beam_new,
    )
    mock.sigma_beam_ch = dish_beam_sigma(13.5, mock.nu)
    tstart = time()

    mock.W_HI = hit_counts_hp>0
    mock.w_HI = hit_counts_hp
    mock.get_enclosing_box()

    num_gal = int(mock.survey_volume * n_gal)
    mock.num_discrete_source = num_gal
    mock.taper_func = getattr(windows, window_name)
    num_pix = mock.W_HI[:,0].sum()

    # randomly generate frequency dependend noise
    generator = np.random.default_rng(seed=seed+50) # this 50 means nothing

    hi_signal_map = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    noise_realisation = sigma_N(num_pix) * generator.normal(size=mock.W_HI.shape)
    mock.data = hi_signal_map + noise_realisation.value

    mock.propagate_mock_tracer_to_gal_cat()
    mock.trim_map_to_range()
    mock.trim_gal_to_range()

    hi_map = mock.data.copy()
    tot_map = fg_map_beam + hi_map

    # resore window
    mock.grid_scheme = "cic"
    mock.downres_factor_transverse = 1.5
    mock.downres_factor_radial = 3
    mock.get_enclosing_box()

    tgen = time()
    logger.debug(f"Time for map generation {tgen - tstart}")

    #############
    # Clean Map #
    #############
    # compute field from data and weights
    mock.data = hi_map.copy()
    himap_rg, _, _ = mock.grid_data_to_field()
    galmap_rg, _, _ = mock.grid_gal_to_field()
    dndz_box = mock.discrete_source_dndz(mock.box_voxel_redshift)

    mock.field_1 = himap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])

    mock.field_2 = galmap_rg
    mock.weights_field_2 = dndz_box
    mock.weights_grid_2 = ((dndz_box>0)*mock.counts_in_box).astype('float')
    mock.apply_taper_to_field(2, axis=[0, 1, 2])

    phi = mock.auto_power_3d_1
    pgal = mock.auto_power_3d_2
    phixgal = mock.cross_power_3d

    tclean = time()
    logger.debug(f"Time for clean Pk {tclean - tgen}")

    ###############
    # Cleaned map #
    ###############
    cov_tot, _, eival, eigvec = pca_clean(
        tot_map, 1, weights=mock.W_HI, return_analysis=True, mean_center=True,
    )
    res_map, A_mat = pca_clean(
        tot_map,
        n_fg,
        weights=mock.W_HI,
        mean_center=True,
        covariance=cov_tot,
        return_A=True,
        ignore_nan=True,
    )
    R_mat = np.eye(mock.nu.size) - A_mat @ A_mat.T
    R_mat = np.nan_to_num(R_mat) # Use for Noise map, also save
    mock.data = res_map
    himap_rg, _, _ = mock.grid_data_to_field()

    mock.field_1 = himap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])

    mock.field_2 = galmap_rg
    mock.weights_field_2 = dndz_box
    mock.weights_grid_2 = ((dndz_box>0)*mock.counts_in_box).astype('float') # test
    mock.apply_taper_to_field(2, axis=[0, 1, 2])

    phi_cleaned = mock.auto_power_3d_1
    pgal_cleaned = mock.auto_power_3d_2
    phixgal_cleaned = mock.cross_power_3d

    tpca = time()
    logger.debug(f"Time for PCAed Pk {tpca - tclean}")
    #########
    # Noise #
    #########
    mock.data = noise_realisation.value
    himap_rg, _, _ = mock.grid_data_to_field()

    mock.field_1 = himap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])

    mock.field_2 = galmap_rg
    mock.weights_field_2 = dndz_box
    mock.weights_grid_2 = ((dndz_box>0)*mock.counts_in_box).astype('float') # test
    mock.apply_taper_to_field(2, axis=[0, 1, 2])

    pnoise = mock.auto_power_3d_1
    pnoisexgal = mock.cross_power_3d

    # Why?
    noise_pc, _, reshape_back = map_los_matrix_form(noise_realisation.value, -1)
    noise_cleaned = R_mat @ noise_pc
    mock.data = reshape_back(noise_cleaned)

    himap_rg, _, _ = mock.grid_data_to_field()
    mock.field_1 = himap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])

    mock.field_2 = galmap_rg
    mock.weights_field_2 = dndz_box
    mock.weights_grid_2 = ((dndz_box>0)*mock.counts_in_box).astype('float') # test
    mock.apply_taper_to_field(2, axis=[0, 1, 2])

    pnoise_cleaned = mock.auto_power_3d_1
    pnoisexgal_cleaned = mock.cross_power_3d
    logger.info(f"Time for Pks {tpca - tstart}")
    return (
        phi, pgal, phixgal,
        phi_cleaned, pgal_cleaned, phixgal_cleaned,
        pnoise, pnoisexgal, pnoise_cleaned, pnoisexgal_cleaned,
        mock.kmode, R_mat
    )

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initialize logger
    logger = _prepare_logging(rank)

    ##########
    # Tasker #
    ##########
    if rank==0:
        Nreal = 28
        seeds = np.arange(Nreal)

        num_workers_done = 0
        next_task = 0

        logger.info(f"Number of realisations = {Nreal}")
        phi_arr, pgal_arr, phixgal_arr = [], [], []
        phi_cleaned_arr, pgal_cleaned_arr, phixgal_cleaned_arr = [], [], []
        pnoise_arr, pnoisexgal_arr, pnoise_cleaned_arr, pnoisexgal_cleaned_arr = [], [], [], []
        R_mat_arr = []
        while num_workers_done < size - 1:
            status = MPI.Status()
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()

            if tag == TAG_TASK:
                if next_task < Nreal:
                    comm.send(seeds[next_task], dest=source, tag=TAG_TASK)
                    next_task += 1
                else:
                    comm.send(None, dest=source, tag=TAG_TERMINATE)
            elif tag == TAG_DONE:
                phi_arr.append(data[0])
                pgal_arr.append(data[1])
                phixgal_arr.append(data[2])
                phi_cleaned_arr.append(data[3])
                pgal_cleaned_arr.append(data[4])
                phixgal_cleaned_arr.append(data[5])
                pnoise_arr.append(data[6])
                pnoisexgal_arr.append(data[7])
                pnoise_cleaned_arr.append(data[8])
                pnoisexgal_cleaned_arr.append(data[9])
                kmode = data[10]
                R_mat_arr.append(data[11])
            elif tag == TAG_TERMINATE:
                num_workers_done += 1
        phi_arr = np.array(phi_arr)
        pgal_arr = np.array(pgal_arr)
        phixgal_arr = np.array(phixgal_arr)
        phi_cleaned_arr = np.array(phi_cleaned_arr)
        pgal_cleaned_arr = np.array(pgal_cleaned_arr)
        phixgal_cleaned_arr = np.array(phixgal_cleaned_arr)
        pnoise_arr = np.array(pnoise_arr)
        pnoisexgal_arr = np.array(pnoisexgal_arr)
        pnoise_cleaned_arr = np.array(pnoise_cleaned_arr)
        pnoisexgal_cleaned_arr = np.array(pnoisexgal_cleaned_arr)
        R_mat_arr = np.array(R_mat_arr)

        np.savez(
            "../data/power_spectra_hp_with_pca_few.npz",
            kmode=kmode,
            R_mat=R_mat_arr,
            phi=phi_arr,
            pgal=pgal_arr,
            phixgal=phixgal_arr,
            phi_cleaned=phi_cleaned_arr,
            pgal_cleaned=pgal_cleaned_arr,
            phixgal_cleaned=phixgal_cleaned_arr,
            pnoise=pnoise_arr,
            pnoisexgal=pnoisexgal_arr,
            pnoise_cleaned=pnoise_cleaned_arr,
            pnoisexgal_cleaned=pnoisexgal_cleaned_arr,
        )

    ##########
    # Worker #
    ##########
    else:
        logger.debug("initialising")
        tstart = time()
        z_func = interp1d(
            z_cen, z_count / dV_arr, kind="linear", bounds_error=False, fill_value=0
        )

        mock = MockSimulation(
            hp_nside=128,
            nu=nu_arr,
            ra_range = ra_range,
            dec_range = dec_range,
            seed=0,
            downres_factor_radial=sim_upres_radial,
            downres_factor_transverse=sim_upres_transverse,
            batch_number=1,
            discrete_source_dndz=z_func,
            tracer_bias_2=1.5,
            tracer_bias_1=1.5,
            sigma_v_1= 100, # in velocity units
            sigma_v_2= 100,
            mean_amp_1="average_hi_temp",
            omega_hi=5e-4,
            sigma_beam_ch=sigma_beam_new,
        )
        mock.data = fg_map.copy()
        mock.W_HI = np.ones_like(hit_counts_hp>0)
        mock.w_HI = np.ones_like(hit_counts_hp)
        mock.sigma_beam_ch = dish_beam_sigma(13.5, mock.nu)

        mock.data,_ = mock.convolve_data(assign_to_self=False)
        mock.trim_map_to_range()
        fg_map_beam = mock.data.copy()

        del mock

        tinit = time()
        logger.debug(f"time for initialisation {tinit - tstart}")

        while True:
            comm.send(None, dest=0, tag=TAG_TASK)
            status = MPI.Status()
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == TAG_TERMINATE:
                logger.info("Terminating")
                comm.send(None, dest=0, tag=TAG_TERMINATE)
                break

            elif tag == TAG_TASK:
                results = get_powerspectra(task, logger)
                comm.send(
                    results, dest=0, tag=TAG_DONE,
                )


