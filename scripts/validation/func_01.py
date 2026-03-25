import sys

sys.path.append("../")

from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal.windows as windows
from astropy.cosmology import Planck18
# project to sky coordinates and back
from meer21cm import MockSimulation
from meer21cm.grid import shot_noise_correction_from_gridding
from meer21cm.plot import plot_map
from meer21cm.power import bin_3d_to_1d, bin_3d_to_cy, get_shot_noise_galaxy
from meer21cm.telescope import dish_beam_sigma
from meer21cm.util import create_wcs, redshift_to_freq
from scipy.interpolate import interp1d

from specs import *


def get_3d_power(seed):
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
        tracer_bias_2=1.5,
        tracer_bias_1=1.5,
        mean_amp_1="average_hi_temp",
        omega_hi=5e-4,
        # sigma_beam_ch=sigma_beam_ch,
        sigma_v_1=100,
        sigma_v_2=100,
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
    mock.downres_factor_transverse = 3
    mock.downres_factor_radial = 6
    mock.get_enclosing_box()
    mock.grid_scheme = "cic"
    himap_rg, _, _ = mock.grid_data_to_field()
    galmap_rg, _, _ = mock.grid_gal_to_field()
    dndz_box = mock.discrete_source_dndz(mock._box_voxel_redshift)
    mock.field_1 = himap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])
    mock.include_sky_sampling = [True, False]
    mock.compensate = [True, True]
    mock.include_beam = [True, False]
    mock.field_2 = galmap_rg
    mock.weights_field_2 = dndz_box
    mock.weights_grid_2 = ((dndz_box>0)*mock.counts_in_box).astype('float')
    # mock.weights_grid_2 = ((dndz_box * mock.counts_in_box) > 0).astype("float")
    mock.apply_taper_to_field(2, axis=[0, 1, 2])
    shot_noise = get_shot_noise_galaxy(
        galmap_rg,
        mock.box_len,
        mock.weights_grid_2,
        mock.weights_field_2,
    ) * shot_noise_correction_from_gridding(mock.box_ndim, mock.grid_scheme)
    pdata3d = mock.auto_power_3d_1
    phimod3d = mock.auto_power_tracer_1_model
    pg3d = mock.auto_power_3d_2 - shot_noise
    pgmod3d = mock.auto_power_tracer_2_model
    pcross3d = mock.cross_power_3d
    pcrossmod3d = mock.cross_power_tracer_model
    print(seed)
    return (
        mock.kmode,
        pdata3d,
        pg3d,
        pcross3d,
        phimod3d,
        pgmod3d,
        pcrossmod3d,
    )


if __name__ == "__main__":
    # run the simulations
    pdata3d_arr = []
    pg3d_arr = []
    pcross3d_arr = []
    with Pool(16) as p:
        for kmode, pdata3d, pg3d, pcross3d, phimod3d, pgmod3d, pcrossmod3d in p.map(
            get_3d_power, range(32)
        ):
            pdata3d_arr.append(pdata3d)
            pg3d_arr.append(pg3d)
            pcross3d_arr.append(pcross3d)

    np.savez(
        f"../../data/power_spectra_func01.npz",
        kmode=kmode,
        phi=np.array(pdata3d_arr),
        pgal=np.array(pg3d_arr),
        phixgal=np.array(pcross3d_arr),
        phi_mod=phimod3d,
        pgal_mod=pgmod3d,
        phixgal_mod=pcrossmod3d,
    )
