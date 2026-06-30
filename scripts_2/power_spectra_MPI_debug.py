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

def get_powerspectra(mock, seed, logger):
    tstart = time()

    mock.data = None
    mock.W_HI = hit_counts_hp>0
    mock.w_HI = hit_counts_hp

    mock.downres_factor_transverse = sim_upres_transverse
    mock.downres_factor_radial = sim_upres_radial
    mock.get_enclosing_box()

    logger.debug(f"seed {seed}")
    mock.seed = seed

    num_gal = int(mock.survey_volume * n_gal)
    mock.num_discrete_source = num_gal
    mock.taper_func = getattr(windows, window_name)
    num_pix = mock.W_HI[:,0].sum()
    logger.debug(num_pix)

    # randomly generate frequency dependend noise
    generator = np.random.default_rng(seed=seed+50) # this 50 means nothing
    noise_realisation = sigma_N(num_pix) * generator.normal(size=mock.W_HI.shape)

    hi_map_raw = mock.mock_tracer_field_1
    hi_map = mock.propagate_mock_field_to_data(hi_map_raw)
    hi_noise_map = hi_map + noise_realisation.value

    mock.data = hi_noise_map
    mock.trim_map_to_range()
    hi_map_post = mock.data.copy()

    # resore window
    mock.grid_scheme = "cic"
    mock.downres_factor_transverse = 1.5
    mock.downres_factor_radial = 3
    mock.get_enclosing_box()
    hi_map_rg, _, _ = mock.grid_data_to_field()

    mock.field_1 = hi_map_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])
    phi = mock.auto_power_3d_1

    return (hi_map_raw, hi_map, hi_noise_map, hi_map_post, hi_map_rg, phi)

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
        Nreal = 2
        seeds = (np.ones(Nreal) * 42).astype(int)

        num_workers_done = 0
        next_task = 0

        logger.info(f"Number of realisations = {Nreal}")

        hi_map_raw = []
        hi_map_rg = []
        phi = []
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
                map_data.append(data)
                raise NotImplementedError
            elif tag == TAG_TERMINATE:
                num_workers_done += 1

        np.savez(
            "../data/map_data_hp_same_seed_nofg.npz",
            hi_map_raw_1 = map_data[0][0],
            hi_map_1 = map_data[0][1],
            hi_noise_map_1 = map_data[0][2],
            hi_map_post_1 = map_data[0][3],
            hi_map_rg_1 = map_data[0][4],
            hi_map_raw_2 = map_data[1][0],
            hi_map_2 = map_data[1][1],
            hi_noise_map_2 = map_data[1][2],
            hi_map_post_2 = map_data[1][3],
            hi_map_rg_2 = map_data[1][4],
            phi_1 = map_data[0][5],
            phi_2 = map_data[1][5],
         )

    ##########
    # Worker #
    ##########
    else:
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
        logger.debug(fg_map_beam.shape)

        tinit = time()
        mock.get_enclosing_box()
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
                    results, dest=0, tag=TAG_DONE,
                )

