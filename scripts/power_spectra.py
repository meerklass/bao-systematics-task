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

from specs import *

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("[%(levelname)s] %(message)s")
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_power(seed):
    logger.debug(f"seed {seed}")

    z_func = interp1d(
        z_cen, z_count / dV_arr, kind="linear", bounds_error=False, fill_value=0
    )
    tstart = time()

    mock = MockSimulation(
        wproj=wcs,
        num_pix_x=num_pix_x,
        num_pix_y=num_pix_y,
        ra_range=ra_range,
        dec_range=dec_range,
        nu=nu_arr,
        discrete_source_dndz=z_func,
        seed=seed,
        tracer_bias_2=1.5,
        tracer_bias_1=1.5,
        mean_amp_1="average_hi_temp",
        omega_hi=5e-4,
        sigma_beam_ch=sigma_beam_new,
        sigma_v_1= 100, # in velocity units
        sigma_v_2= 100,
    )
    tinit = time()
    logger.debug(f"time for initialisation {tinit - tstart}")

    mock.taper_func = getattr(windows, window_name)

    num_gal = int(mock.survey_volume * n_gal)
    mock.num_discrete_source = num_gal
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)

    mock.downres_factor_transverse = 1 / 2
    mock.downres_factor_radial = 1 / 2
    mock.get_enclosing_box()

    num_pix =mock.W_HI[:,:,0].sum()
    # randomly generate frequency dependend noise
    generator = np.random.default_rng(seed=seed+50) # this 50 means nothing
    noise_realisation = sigma_N(num_pix)[None, None, :] * (
        generator.normal(size=(num_pix_x, num_pix_y, num_ch))
    )

    mock.data = (
        mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
        + noise_realisation.value
    )
    mock.propagate_mock_tracer_to_gal_cat()
    mock.trim_map_to_range()
    mock.trim_gal_to_range()

    tdata = time()
    logger.debug(f"time for data generation {tdata - tinit}")

    # resore window
    mock.trim_map_to_range()
    mock.downres_factor_transverse = 3
    mock.downres_factor_radial = 6
    mock.get_enclosing_box()

    # compute field from data and weights
    mock.grid_scheme = "cic"
    himap_rg, _, _ = mock.grid_data_to_field()
    galmap_rg, _, _ = mock.grid_gal_to_field()
    dndz_box = mock.discrete_source_dndz(mock._box_voxel_redshift)

    mock.field_1 = himap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])

    mock.field_2 = galmap_rg
    mock.weights_field_2 = dndz_box
    mock.weights_grid_2 = ((dndz_box>0)*mock.counts_in_box).astype('float') # test
    mock.apply_taper_to_field(2, axis=[0, 1, 2])

    tresampling = time()
    logger.debug(f"time for resampling {tresampling - tdata}")

    phi_3d = mock.auto_power_3d_1
    pgal_3d = mock.auto_power_3d_2
    phixgal_3d = mock.cross_power_3d

    tpower = time()
    logger.debug(f"time for power spectra {tpower - tresampling}")

    mock.data = noise_realisation.value
    himap_rg, _, _ = mock.grid_data_to_field()

    mock.field_1 = himap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])

    mock.field_2 = galmap_rg
    mock.weights_field_2 = dndz_box
    mock.weights_grid_2 = ((dndz_box>0)*mock.counts_in_box).astype('float') # test
    mock.apply_taper_to_field(2, axis=[0, 1, 2])

    pnoise_3d = mock.auto_power_3d_1
    pnoisexgal_3d = mock.cross_power_3d

    tnoise = time()
    logger.debug(f"time for noise spectra {tnoise - tpower}")

    return mock.kmode, phi_3d, pgal_3d, phixgal_3d, pnoise_3d, pnoisexgal_3d

if __name__ == "__main__":

    # run the simulations
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    logger.info(f"Number of cpus used = {n_cpus}")

    scaling = 8
    os.environ["OMP_NUM_THREADS"] = str(scaling)
    squeezed_cpus = int(n_cpus // scaling)

    Nreal = 500
    logger.info(f"Number of realisations = {Nreal}")

    phi_arr, pgal_arr, phixgal_arr = [], [], []
    pnoise_arr, pnoisexgal_arr = [], []

    tstart = time()

    with ProcessPoolExecutor(max_workers=squeezed_cpus) as executor:
        futures = {
            executor.submit(get_power, seed): seed
            for seed in range(Nreal)
        }

        for future in as_completed(futures):
            seed = futures[future]

            kmode, phi, pgal, phixgal, pnoise, pnoisexgal = future.result()
            phi_arr.append(phi)
            pgal_arr.append(pgal)
            phixgal_arr.append(phixgal)
            pnoise_arr.append(pnoise)
            pnoisexgal_arr.append(pnoisexgal)
    logger.info(f"Time for completion with scale={scaling}: {time()-tstart:.2f}s")

    phi_arr = np.array(phi_arr)
    pgal_arr = np.array(pgal_arr)
    phixgal_arr = np.array(phixgal_arr)
    pnoise_arr = np.array(pnoise_arr)
    pnoisexgal_arr = np.array(pnoisexgal_arr)

    np.savez(
        "../data/power_spectra.npz",
        kmode=kmode,
        phi=phi_arr,
        pgal=pgal_arr,
        phixgal=phixgal_arr,
        pnoise=pnoise_arr,
        pnoisexgal=pnoisexgal_arr,
    )
