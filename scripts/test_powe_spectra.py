import sys
import os
from multiprocessing import Pool

import astropy.constants as c
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import scipy.signal.windows as windows

from meer21cm.power import bin_3d_to_cy, bin_3d_to_1d
from meer21cm.telescope import dish_beam_sigma
from meer21cm import MockSimulation
from meer21cm.plot import plot_map
from meer21cm.util import create_wcs, redshift_to_freq
from meer21cm.power import get_shot_noise_galaxy
from meer21cm.grid import shot_noise_correction_from_gridding

from specs import *

# Setup logger
import logging
from time import time
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("[%(levelname)s] %(message)s")
)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

def get_power(seed):
    logger.debug(f"seed {seed}")

    z_func = interp1d(
        z_cen, z_count / dV_arr, kind="linear", bounds_error=False, fill_value=0
    )
    sigma_beam_ch = dish_beam_sigma(13.5, nu_arr)
    mock = MockSimulation(
        wproj=wcs,
        num_pix_x=num_pix_x,
        num_pix_y=num_pix_y,
        ra_range=ra_range,
        dec_range=dec_range,
        nu=nu_arr,
        discrete_source_dndz=z_func,
        seed=seed,
        tracer_bias_2=1.5, # Compute power spectrum in units of b_1 b_2 etc.
        tracer_bias_1=1.5,
        mean_amp_1="average_hi_temp",
        omega_hi=5e-4,
        # sigma_beam_ch=sigma_beam_ch,
        sigma_v_1= 105,
        sigma_v_2= 105,
    )
    comov_dist = Planck18.comoving_distance(mock.z_ch).value
    sigma_beam_new = 1 / comov_dist * sigma_beam_ch
    sigma_beam_new *= sigma_beam_ch.mean() / sigma_beam_new.mean()
    mock.sigma_beam_ch = sigma_beam_new

    mock.taper_func = getattr(windows, window_name)

    num_gal = int(mock.survey_volume * n_gal)
    mock.num_discrete_source = num_gal
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)

    mock.downres_factor_transverse = 1 / 2
    mock.downres_factor_radial = 1 / 2
    mock.get_enclosing_box()

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
    gamap_rg, _, _ = mock.grid_gal_to_field()
    dndz_box = mock.discrete_source_dndz(mock._box_voxel_redshift)

    mock.field_1 = himap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])

    mock.include_sky_sampling = [True, False]
    mock.compensate = [True, True]
    mock.include_beam = [True, False]

    mock.field_2 = gamap_rg
    mock.weights_field_2 = dndz_box
    mock.weights_grid_2 = ((dndz_box>0)*mock.counts_in_box).astype('float')
    mock.apply_taper_to_field(2, axis=[0, 1, 2])

    shot_noise = (
        get_shot_noise_galaxy(
            gamap_rg,
            mock.box_len,
            mock.weights_grid_2,
            mock.weights_field_2,
        )
        * shot_noise_correction_from_gridding(
            mock.box_ndim, mock.grid_scheme
        )
    )
    phi_3d = mock.auto_power_3d_1
    phi_3d_model = mock.auto_power_tracer_1_model

    pgal_3d = mock.auto_power_3d_2 - shot_noise
    phixgal_3d = mock.cross_power_3d

    pgal_3d_model = mock.auto_power_tracer_2_model
    phixgal_3d_model = mock.cross_power_tracer_model

    return (
        mock.kmode,
        phi_3d,
        pgal_3d,
        phixgal_3d,
        phi_3d_model,
        pgal_3d_model,
        phixgal_3d_model,
    )

if __name__ == "__main__":
    # run the simulations
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) # Assumes that Slurm is used for tasking
    logger.info(f"Number of cpus used = {n_cpus}")

    scaling = 2
    os.environ["OMP_NUM_THREADS"] = str(scaling)
    squeezed_cpus = int(n_cpus//scaling)

    Nreal = 32
    logger.info(f"Number of realisations = {Nreal}")

    phi_arr, pgal_arr, phixgal_arr = [], [], []

    tstart = time()
    with Pool(squeezed_cpus) as p:
        for kmode, phi, pgal, phixgal, phi_mod, pgal_mod, phixgal_mod in p.map(get_power, range(Nreal)):
            phi_arr.append(phi)
            pgal_arr.append(pgal)
            phixgal_arr.append(phixgal)
    logger.info(f"Time for compleation with scale={scaling}: {time()-tstart}")

    phi_arr = np.array(phi_arr)
    pgal_arr = np.array(pgal_arr)
    phixgal_arr = np.array(phixgal_arr)

    np.savez(
        "../data/test_powerspectra_nosn.npz",
        kmode=kmode,
        phi=phi_arr,
        pgal=pgal_arr,
        phixgal=phixgal_arr,
        phi_mod=phi_mod,
        pgal_mod=pgal_mod,
        phixgal_mod=phixgal_mod,
    )
