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


NU_MHZ = np.array([565.2928416485901, 578.3080260303688, 585.6832971800434, 591.7570498915401, 606.5075921908893, 616.9197396963124, 626.4642082429501, 631.236442516269, 643.3839479392625, 646.4208242950108, 654.2299349240781, 665.5097613882863, 677.6572668112798, 690.2386117136659, 704.5553145336225, 720.1735357917571, 738.82863340564, 751.8438177874186, 755.3145336225597, 768.763557483731, 791.7570498915402, 802.1691973969631, 820.824295010846, 837.7440347071583, 847.2885032537961, 859.002169197397, 868.5466377440348, 873.7527114967462, 884.1648590021691, 892.407809110629, 911.062906724512, 923.644251626898, 933.1887201735358, 960.9544468546637, 983.5140997830803, 996.9631236442517, 1011.7136659436009, 1030.3687635574838, 1047.288503253796, 1055.0976138828632, 1060.303687635575])

TSYS_OVER_ETA_K = np.array([36.75302245250432, 35.673575129533674, 34.98272884283247, 33.773747841105354, 33.1692573402418, 32.65112262521589, 32.089810017271155, 31.528497409326427, 31.355785837651123, 30.362694300518136, 29.32642487046632, 30.40587219343696, 29.585492227979273, 29.02417962003454, 27.858376511226254, 27.5993091537133, 27.08117443868739, 25.094991364421418, 26.260794473229705, 25.9153713298791, 25.310880829015545, 23.97236614853195, 23.97236614853195, 22.979274611398964, 23.238341968911918, 22.461139896373055, 21.8566493955095, 21.33851468048359, 19.8272884283247, 22.202072538860104, 21.8566493955095, 21.986183074265977, 21.07944732297064, 20.77720207253886, 20.129533678756477, 19.654576856649395, 19.870466321243523, 20.08635578583765, 21.511226252158895, 23.324697754749568, 28.808290155440414])


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

def add_boundary_knots(spline):
    """
    Add knots infinitesimally to the left and right.

    Additional intervals are added to have zero 2nd and 3rd derivatives,
    and to maintain the first derivative from whatever boundary condition
    was selected. The spline is modified in place.
    """
    # determine the slope at the left edge
    leftx = spline.x[0]
    lefty = spline(leftx)
    leftslope = spline(leftx, nu=1)

    # add a new breakpoint just to the left and use the
    # known slope to construct the PPoly coefficients.
    leftxnext = np.nextafter(leftx, leftx - 1)
    leftynext = lefty + leftslope*(leftxnext - leftx)
    leftcoeffs = np.array([0, 0, leftslope, leftynext])
    spline.extend(leftcoeffs[..., None], np.r_[leftxnext])

    # repeat with additional knots to the right
    rightx = spline.x[-1]
    righty = spline(rightx)
    rightslope = spline(rightx,nu=1)
    rightxnext = np.nextafter(rightx, rightx + 1)
    rightynext = righty + rightslope * (rightxnext - rightx)
    rightcoeffs = np.array([0, 0, rightslope, rightynext])
    spline.extend(rightcoeffs[..., None], np.r_[rightxnext])


tsys_inter = CubicSpline(NU_MHZ, TSYS_OVER_ETA_K, bc_type="natural")
add_boundary_knots(tsys_inter)


def sigma_N(ps):
    nu = nu_arr * u.Hz
    dnu = nu_resol * u.Hz

    tsys_over_eta = tsys_inter(nu.to(u.MHz).value) * u.K
    num_pix =ps.W_HI[:,:,0].sum()

    t_tot = 20 * u.hr
    n_dish = 64
    n_feeds = 2
    t_pixel = n_dish * t_tot / num_pix

    return tsys_over_eta / np.sqrt(n_feeds * (dnu * t_pixel).to(1).value)


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
        sigma_v_1= 5, # Rough Halo model quantity, fine for now
        sigma_v_2= 5,
    )

    mock.taper_func = getattr(windows, window_name)
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)
    num_gal = int(mock.survey_volume * n_gal)
    mock.num_discrete_source = num_gal

    mock.downres_factor_transverse = 1 / 2
    mock.downres_factor_radial = 1 / 2
    mock.get_enclosing_box()

    # randomly generate frequency dependend noise
    generator = np.random.default_rng(seed=seed+50)
    noise_realisation = sigma_N(mock)[None, None, :] * (
        generator.normal(size=(num_pix_x, num_pix_y, num_ch))
    )
    mock.data = (
        mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
        + noise_realisation.value
    )
    mock.propagate_mock_tracer_to_gal_cat()
    mock.trim_map_to_range()
    mock.trim_gal_to_range()

    # resore window
    mock.trim_map_to_range()
    mock.downres_factor_transverse = 3
    mock.downres_factor_radial = 6
    mock.get_enclosing_box()
    mock.grid_scheme = "nnb"

    # compute field from data and weights
    himap_rg, _, _ = mock.grid_data_to_field()
    gamap_rg, _, _ = mock.grid_gal_to_field()
    dndz_box = mock.discrete_source_dndz(mock._box_voxel_redshift)

    mock.field_1 = himap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])

    mock.field_2 = gamap_rg
    mock.weights_field_2 = dndz_box
    mock.apply_taper_to_field(2, axis=[0, 1, 2])

    phi_3d = mock.auto_power_3d_1

    shot_noise = get_shot_noise_galaxy(
            gamap_rg,
            mock.box_len,
            mock.weights_grid_2,
            mock.weights_field_2,
        ) * shot_noise_correction_from_gridding(mock.box_ndim, mock.grid_scheme)
    pgal_3d = mock.auto_power_3d_2 - shot_noise
    phixgal_3d = mock.cross_power_3d

    mock.data = noise_realisation.value
    himap_rg, _, _ = mock.grid_data_to_field()

    mock.field_1 = himap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])

    mock.field_2 = gamap_rg
    mock.weights_field_2 = dndz_box
    mock.apply_taper_to_field(2, axis=[0, 1, 2])

    pnoise_3d = mock.auto_power_3d_1
    pnoisexgal_3d = mock.cross_power_3d
    return mock.kmode, phi_3d, pgal_3d, phixgal_3d, pnoise_3d, pnoisexgal_3d

if __name__ == "__main__":
    # run the simulations
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) # Assumes that Slurm is used for tasking
    logger.info(f"Number of cpus used = {n_cpus}")

    scaling = 2
    os.environ["OMP_NUM_THREADS"] = str(scaling)
    squeezed_cpus = int(n_cpus//scaling)

    Nreal = 500
    logger.info(f"Number of realisations = {Nreal}")

    phi_arr, pgal_arr, phixgal_arr = [], [], []
    pnoise_arr, pnoisexgal_arr = [], []

    tstart = time()
    with Pool(squeezed_cpus) as p:
        for kmode, phi, pgal, phixgal, pnoise, pnoisexgal in p.map(get_power, range(Nreal)):
            phi_arr.append(phi)
            pgal_arr.append(pgal)
            phixgal_arr.append(phixgal)
            pnoise_arr.append(pnoise)
            pnoisexgal_arr.append(pnoisexgal)
    logger.info(f"Time for compleation with scale={scaling}: {time()-tstart}")

    phi_arr = np.array(phi_arr)
    pgal_arr = np.array(pgal_arr)
    phixgal_arr = np.array(phixgal_arr)
    pnoise_arr = np.array(pnoise_arr)
    pnoisexgal_arr = np.array(pnoisexgal_arr)

    np.savez(
        "../data/powerspectra.npz",
        kmode=kmode,
        phi=phi_arr,
        pgal=pgal_arr,
        phixgal=phixgal_arr,
        pnoise=pnoise_arr,
        pnoisexgal=pnoisexgal_arr,
    )
