# Setup logger
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time

import numpy as np
import scipy.signal.windows as windows
from meer21cm import MockSimulation
from scipy.interpolate import interp1d

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)
from specs import *

from mpi4py import MPI

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


def get_powerspectra(mock, seed, logger):
    logger.debug(f"seed {seed}")
    tstart = time()
    mock.seed = seed

    mock.taper_func = getattr(windows, window_name)
    mock.num_discrete_source = int(mock.survey_volume * n_gal)
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)
    mock.downres_factor_transverse = 1 / 2
    mock.downres_factor_radial = 1 / 2
    mock.get_enclosing_box()
    num_pix = mock.W_HI[:, :, 0].sum()

    # randomly generate frequency dependend noise
    generator = np.random.default_rng(seed=seed + 50)  # this 50 means nothing
    noise_realisation = sigma_N(num_pix)[None, None, :] * (
        generator.normal(size=(num_pix_x, num_pix_y, num_ch))
    )

    mock.data = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    mock.propagate_mock_tracer_to_gal_cat()
    mock.trim_map_to_range()
    mock.trim_gal_to_range()

    # resore window
    mock.downres_factor_transverse = 3
    mock.downres_factor_radial = 6
    mock.get_enclosing_box()

    # compute field from data and weights
    mock.grid_scheme = "cic"
    himap_rg, _, _ = mock.grid_data_to_field()
    galmap_rg, _, _ = mock.grid_gal_to_field()
    dndz_box = mock.discrete_source_dndz(mock.box_voxel_redshift)

    mock.field_1 = himap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])

    mock.field_2 = galmap_rg
    mock.weights_field_2 = dndz_box
    mock.weights_grid_2 = ((dndz_box > 0) * mock.counts_in_box).astype("float")  # test
    mock.apply_taper_to_field(2, axis=[0, 1, 2])

    phi_arr = mock.auto_power_3d_1
    pgal_arr = mock.auto_power_3d_2
    phixgal_arr = mock.cross_power_3d
    return phi_arr, pgal_arr, phixgal_arr, mock.k_mode


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initialize logger
    logger = _prepare_logging(rank)

    if rank == 0:
        Nreal = 4
        seeds = [0, 4, 0, 4]

        num_workers_done = 0
        next_task = 0

        logger.info(f"Number of realisations = {Nreal}")
        phi_arr, pgal_arr, phixgal_arr = [], [], []
        pnoise_arr, pnoisexgal_arr = [], []
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
                kmode = data[3]
            elif tag == TAG_TERMINATE:
                num_workers_done += 1

        phi_arr = np.array(phi_arr)
        pgal_arr = np.array(pgal_arr)
        phixgal_arr = np.array(phixgal_arr)
        pnoise_arr = np.array(pnoise_arr)
        pnoisexgal_arr = np.array(pnoisexgal_arr)
        np.savez(
            "../data/test_seed_reset.npz",
            kmode=kmode,
            phi=phi_arr,
            pgal=pgal_arr,
            phixgal=phixgal_arr,
            pnoise=pnoise_arr,
            pnoisexgal=pnoisexgal_arr,
        )

        z_func = interp1d(
            z_cen, z_count / dV_arr, kind="linear", bounds_error=False, fill_value=0
        )
        mock = MockSimulation(
            wproj=wcs,
            num_pix_x=num_pix_x,
            num_pix_y=num_pix_y,
            ra_range=ra_range,
            dec_range=dec_range,
            nu=nu_arr,
            discrete_source_dndz=z_func,
            seed=4,
            tracer_bias_2=1.5,
            tracer_bias_1=1.5,
            mean_amp_1="average_hi_temp",
            omega_hi=5e-4,
            sigma_beam_ch=sigma_beam_new,
            sigma_v_1=100,  # in velocity units
            sigma_v_2=100,
        )
        results = get_powerspectra(mock, 4, logger)
        np.savez(
            "../data/test_between.npz",
            phi_direct=results[0],
            phi_reset=phi_arr[1],
            pgal_direct=results[1],
            pgal_reset=pgal_arr[1],
            phixgal_direct=results[2],
            phixgal_reset=phixgal_arr[1],
        )

    else:
        tstart = time()
        z_func = interp1d(
            z_cen, z_count / dV_arr, kind="linear", bounds_error=False, fill_value=0
        )

        mock = MockSimulation(
            wproj=wcs,
            num_pix_x=num_pix_x,
            num_pix_y=num_pix_y,
            ra_range=ra_range,
            dec_range=dec_range,
            nu=nu_arr,
            discrete_source_dndz=z_func,
            seed=0,
            tracer_bias_2=1.5,
            tracer_bias_1=1.5,
            mean_amp_1="average_hi_temp",
            omega_hi=5e-4,
            sigma_beam_ch=sigma_beam_new,
            sigma_v_1=100,  # in velocity units
            sigma_v_2=100,
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
                results = get_powerspectra(mock, task, logger)
                comm.send(
                    results,
                    dest=0,
                    tag=TAG_DONE,
                )
