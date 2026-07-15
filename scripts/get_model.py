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

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)
from specs_v2 import *

TAG_TASK = 1
TAG_DONE = 2
TAG_TERMINATE = 3


def _prepare_logging(rank):
    logger = logging.getLogger(f"rank{rank}")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(f"[Rank {rank:02d} %(levelname)s] %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def get_powerspectra(seed, logger):
    logger.debug(f"seed {seed}")
    mock = MockSimulation(
        hp_nside=128,
        nu=nu_arr,
        ra_range=ra_range,
        dec_range=dec_range,
        seed=seed,
        downres_factor_radial=sim_upres_radial,
        downres_factor_transverse=sim_upres_transverse,
        batch_number=1,
        discrete_source_dndz=z_func,
        tracer_bias_2=1.5,
        tracer_bias_1=1.5,
        sigma_v_1=100,  # in velocity units
        sigma_v_2=100,
        mean_amp_1="average_hi_temp",
        omega_hi=5e-4,
        sigma_beam_ch=sigma_beam_new,
    )
    t1 = time()
    mock.taper_func = getattr(windows, window_name)
    mock.sigma_beam_ch = dish_beam_sigma(13.5, mock.nu)
    num_gal = int(mock.survey_volume * n_gal)
    mock.num_discrete_source = num_gal

    mock.W_HI = hit_counts_hp > 0
    mock.w_HI = hit_counts_hp
    mock.get_enclosing_box()

    hi_signal_map = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    generator = np.random.default_rng(seed=seed + 50)  # this 50 means nothing
    num_pix = mock.W_HI[:, 0].sum()
    noise_realisation = sigma_N(num_pix) * generator.normal(size=mock.W_HI.shape)
    mock.data = hi_signal_map + noise_realisation.value

    mock.propagate_mock_tracer_to_gal_cat()
    mock.trim_map_to_range()
    mock.trim_gal_to_range()

    mock.grid_scheme = "cic"
    mock.downres_factor_transverse = 1.5
    mock.downres_factor_radial = 1
    mock.get_enclosing_box()
    t2 = time()
    logger.debug(f"setup {t2-t1} s")

    #############
    # Clean Map #
    #############
    # compute field from data and weights
    himap_rg, _, _ = mock.grid_data_to_field()
    galmap_rg, _, _ = mock.grid_gal_to_field()
    dndz_box = mock.discrete_source_dndz(mock.box_voxel_redshift)

    mock.field_1 = himap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])

    mock.include_sky_sampling = [True, False]
    mock.compensate = [True, True]
    mock.include_beam = [True, False]

    mock.field_2 = galmap_rg
    mock.weights_field_2 = dndz_box
    mock.weights_grid_2 = ((dndz_box > 0) * mock.counts_in_box).astype("float")
    mock.apply_taper_to_field(2, axis=[0, 1, 2])

    phi = mock.auto_power_3d_1
    phi_mod = mock.auto_power_tracer_1_model
    pgal = mock.auto_power_3d_2
    pgal_mod = mock.auto_power_tracer_2_model
    phixgal = mock.cross_power_3d
    phixgal_mod = mock.cross_power_tracer_model

    t3 = time()
    logger.debug(f"finialising {t3-t2} s")
    return (
        phi,
        pgal,
        phixgal,
        phi_mod,
        pgal_mod,
        phixgal_mod,
        mock.kmode,
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
    if rank == 0:
        Nreal = 21
        seeds = np.arange(0, Nreal)

        num_workers_done = 0
        next_task = 0

        logger.info(f"Number of realisations = {Nreal}")
        phi_arr = []
        pgal_arr = []
        phixgal_arr = []
        phi_mod_arr = []
        pgal_mod_arr = []
        phixgal_mod_arr = []

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
                phi_mod_arr.append(data[3])
                pgal_mod_arr.append(data[4])
                phixgal_mod_arr.append(data[5])
                kmode = data[6]
            elif tag == TAG_TERMINATE:
                num_workers_done += 1
        phi_arr = np.array(phi_arr)
        pgal_arr = np.array(pgal_arr)
        phixgal_arr = np.array(phixgal_arr)
        phi_mod_arr = np.array(phi_mod_arr)
        pgal_mod_arr = np.array(pgal_mod_arr)
        phixgal_mod_arr = np.array(phixgal_mod_arr)

        np.savez(
            "../data/power_spectra_models_simple.npz",
            kmode=kmode,
            phi=phi_arr,
            pgal=pgal_arr,
            phixgal=phixgal_arr,
            phi_mod=phi_mod_arr,
            pgal_mod=pgal_mod_arr,
            phixgal_mod=phixgal_mod_arr,
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
                    results,
                    dest=0,
                    tag=TAG_DONE,
                )
