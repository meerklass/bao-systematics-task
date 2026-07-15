import os
from multiprocessing import Pool

from astropy.cosmology import Planck18
import astropy.units as u
import healpy as hp
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import scipy.signal.windows as windows

from meer21cm import MockSimulation
from meer21cm.grid import shot_noise_correction_from_gridding
from meer21cm.plot import plot_map
from meer21cm.power import bin_3d_to_cy, bin_3d_to_1d
from meer21cm.power import get_shot_noise_galaxy
from meer21cm.telescope import dish_beam_sigma
from meer21cm.util import create_wcs, redshift_to_freq, freq_to_redshift
from meer21cm.util import pca_clean
from meer21cm.util import _map_los_matrix_form as map_los_matrix_form

from utils import add_boundary_knots

window_name = "blackmanharris"
dndz_file = "../specs/LRGELG_dndz.npz"

z_min = 0.4
z_max = 1.1
nu_min = redshift_to_freq(z_max)
nu_max = redshift_to_freq(z_min)

dndz_data = np.load(dndz_file)
z_bin = dndz_data["z_bin"]
z_count = dndz_data["z_count"]
z_cen = (z_bin[:-1] + z_bin[1:]) / 2
dV_arr = Planck18.differential_comoving_volume(z_cen)

# DESI DR2
n_gal = (1052151 + 1613562 + 1802770) / (4.9 + 7.6 + 9.8) / 1e9  # Mpc-3

z_func = interp1d(
        z_cen, z_count / dV_arr, kind="linear", bounds_error=False, fill_value=0
    )

kperpbins = np.linspace(0,0.06,16)
kparabins = np.linspace(0,0.4,21)
k1dbins = np.linspace(0.01, 0.2, 21)

ps_grid_downres_factor_radial = 2
ps_grid_downres_factor_transverse = 2
# downres map frequency resolution to reduce compute
downres_factor_freq = 5

grid_scheme = "cic"

sim_upres_radial = 1/3
sim_upres_transverse = 1/3

# increase for less memory usage
batch_number = 3

# need radec_range for healpix initialisation
ra_range = (123, 176)
dec_range = (-10.5, 7)

######################################
# Foreground Substraction Estimation #
######################################

# number of foreground components
n_fg = 5

pickle_file = "DESI_1.pkl"
def generate_healpix_mask():
    # extract the basic metadata
    mock = MockSimulation(
        nu_min=nu_min,
        nu_max=nu_max,
        pickle_file=pickle_file,
        downres_factor_radial=sim_upres_radial,
        downres_factor_transverse=sim_upres_transverse,
        batch_number=batch_number,
        discrete_source_dndz=z_func,
        tracer_bias_2=1.0,
        tracer_bias_1=1.0,
        sigma_v_1=100,
        sigma_v_2=100,
        mean_amp_1="average_hi_temp",
    )
    mock.read_from_pickle()
    mock.nu = mock.nu[::downres_factor_freq] # no need for extremely fine frequency resolution
    mock.W_HI = mock.W_HI[:,:,::downres_factor_freq]
    mock.w_HI = mock.w_HI[:,:,::downres_factor_freq]
    mock.data = np.zeros((mock.num_pix_x,mock.num_pix_y,mock.nu.size))
    mock.trim_map_to_range()
    mock_hp = MockSimulation(
        hp_nside=128, # roughly the same as wcs map
        nu = mock.nu,
        ra_range = ra_range,
        dec_range = dec_range,
    )
    pixel_id_wcs = hp.ang2pix(mock_hp.hp_nside,mock.ra_map,mock.dec_map,lonlat=True)
    hit_counts_hp = np.zeros((hp.nside2npix(mock_hp.hp_nside),len(mock.nu)))
    for i in range(len(mock.nu)):
        np.add.at(hit_counts_hp[:,i],pixel_id_wcs.ravel(),mock.w_HI[:,:,i].ravel())
        # smooth it
        map_temp = hp.ud_grade(hit_counts_hp[:,i],mock_hp.hp_nside//2)
        hit_counts_hp[:,i] = hp.ud_grade(map_temp,mock_hp.hp_nside)
    hit_counts_hp = hit_counts_hp[mock_hp.pixel_id]
    return mock.nu, hit_counts_hp

def generate_mock_healpix(nu=None,hit_counts_hp=None):
    if hit_counts_hp is None:
        nu, hit_counts_hp = generate_healpix_mask()
    mock = MockSimulation(
        hp_nside=128,
        nu = nu,
        ra_range = ra_range,
        dec_range = dec_range,
    )
    mock.W_HI = hit_counts_hp>0
    mock.w_HI = hit_counts_hp
    return mock

#nu_arr, hit_counts_hp = generate_healpix_mask()
#np.save('nu_arr',nu_arr)
#np.save('hit_counts_hp',hit_counts_hp)

nu_arr = np.load('../specs/nu_arr.npy')
nu_resol = np.diff(nu_arr).mean()

fg_map = np.load('../specs/fg_map_wcs.npy')
hit_counts_hp = np.load('../specs/hit_counts_hp.npy')
fg_map = fg_map[generate_mock_healpix(nu_arr, hit_counts_hp).pixel_id]

#######################
# Detector Resolution #
#######################

z_ch = freq_to_redshift(nu_arr)
_sigma_beam_ch = dish_beam_sigma(13.5, nu_arr)
_comov_dist = Planck18.comoving_distance(z_ch).value
sigma_beam_new = 1 / _comov_dist * _sigma_beam_ch
sigma_beam_new *= _sigma_beam_ch.mean() / sigma_beam_new.mean()

##################
# Detector Noise #
##################

NU_MHZ = np.array([565.2928416485901, 578.3080260303688, 585.6832971800434, 591.7570498915401, 606.5075921908893, 616.9197396963124, 626.4642082429501, 631.236442516269, 643.3839479392625, 646.4208242950108, 654.2299349240781, 665.5097613882863, 677.6572668112798, 690.2386117136659, 704.5553145336225, 720.1735357917571, 738.82863340564, 751.8438177874186, 755.3145336225597, 768.763557483731, 791.7570498915402, 802.1691973969631, 820.824295010846, 837.7440347071583, 847.2885032537961, 859.002169197397, 868.5466377440348, 873.7527114967462, 884.1648590021691, 892.407809110629, 911.062906724512, 923.644251626898, 933.1887201735358, 960.9544468546637, 983.5140997830803, 996.9631236442517, 1011.7136659436009, 1030.3687635574838, 1047.288503253796, 1055.0976138828632, 1060.303687635575])

TSYS_OVER_ETA_K = np.array([36.75302245250432, 35.673575129533674, 34.98272884283247, 33.773747841105354, 33.1692573402418, 32.65112262521589, 32.089810017271155, 31.528497409326427, 31.355785837651123, 30.362694300518136, 29.32642487046632, 30.40587219343696, 29.585492227979273, 29.02417962003454, 27.858376511226254, 27.5993091537133, 27.08117443868739, 25.094991364421418, 26.260794473229705, 25.9153713298791, 25.310880829015545, 23.97236614853195, 23.97236614853195, 22.979274611398964, 23.238341968911918, 22.461139896373055, 21.8566493955095, 21.33851468048359, 19.8272884283247, 22.202072538860104, 21.8566493955095, 21.986183074265977, 21.07944732297064, 20.77720207253886, 20.129533678756477, 19.654576856649395, 19.870466321243523, 20.08635578583765, 21.511226252158895, 23.324697754749568, 28.808290155440414])

tsys_inter = CubicSpline(NU_MHZ, TSYS_OVER_ETA_K, bc_type="natural")
add_boundary_knots(tsys_inter)

def sigma_N(num_pix):
    nu = nu_arr * u.Hz
    dnu = nu_resol * u.Hz

    tsys_over_eta = tsys_inter(nu.to(u.MHz).value) * u.K

    t_tot = 20 * u.hr
    n_dish = 64
    n_feeds = 2
    t_pixel = n_dish * t_tot / num_pix

    return tsys_over_eta / np.sqrt(n_feeds * (dnu * t_pixel).to(1).value)


