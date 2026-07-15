import multiprocessing as mp
import os
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from meer21cm.util import dft_matrix
import niceplots
import numpy as np
import seaborn as sns
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import binned_statistic, binned_statistic_2d

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)
from classy_wraper_for_m21cm import *
from specs_v2 import *

Cs = sns.color_palette("colorblind")
niceplots.initPlot()

sims_file = "/home/sefa/Desktop/projects/meerklass/get_fg_reduction/power_spectra_hp_with_pca_batch1.npz"

###################
# Get mock object #
###################
z_func = interp1d(
    z_cen, z_count / dV_arr, kind="linear", bounds_error=False, fill_value=0
)

mock = MockSimulation(
    hp_nside=128,
    nu=nu_arr,
    ra_range=ra_range,
    dec_range=dec_range,
    seed=0,
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
mock.taper_func = getattr(windows, window_name)
mock.sigma_beam_ch = dish_beam_sigma(13.5, mock.nu)
num_gal = int(mock.survey_volume * n_gal)
mock.num_discrete_source = num_gal

mock.W_HI = hit_counts_hp > 0
mock.w_HI = hit_counts_hp
mock.get_enclosing_box()

mock.grid_scheme = "cic"
mock.downres_factor_transverse = 1.5
mock.downres_factor_radial = 1
mock.get_enclosing_box()

dndz_box = mock.discrete_source_dndz(mock.box_voxel_redshift)
mock.weights_1 = mock.counts_in_box.astype(np.float32)
mock.apply_taper_to_field(1, axis=[0, 1, 2])

mock.include_sky_sampling = [True, False]
mock.compensate = [True, True]
mock.include_beam = [True, False]

mock.weights_field_2 = dndz_box
mock.weights_grid_2 = ((dndz_box > 0) * mock.counts_in_box).astype("float")
mock.apply_taper_to_field(2, axis=[0, 1, 2])

##########################
# Get scalecuts and grid #
##########################
k = mock.k_mode
mu = mock.mu_mode

kpar = k * mu
kperp = k * np.sqrt(np.clip(1 - mu**2, 0, 1))

kvec = mock.k_vec
kmask = (
    (np.abs(kvec[0]) < 0.06)[:, None, None]
    * (np.abs(kvec[0]) > 0.005)[:, None, None]
    * (np.abs(kvec[1]) < 0.06)[None, :, None]
    * (np.abs(kvec[1]) > 0.005)[None, :, None]
    * (np.abs(kvec[2]) < 0.3)[None, None, :]
)

kbinedges = np.linspace(k[kmask].min(), k[kmask].max(), 20)
kcenters = 0.5 * (kbinedges[1:] + kbinedges[:-1])

N, _, bins = binned_statistic(k[kmask], [], "count", kbinedges)
B1d = np.zeros((*bins.shape, *N.shape))
for i, b in enumerate(bins):
    B1d[i, b - 1] += 1
B1d *= 1 / N


def bin_Pk_1d(Pk_3d):
    return np.einsum("ji, j -> i", B1d, Pk_3d[kmask])


def get_1d_cov(Pk_3d_arr):
    Pks = np.einsum("ji, lj -> li", B1d, Pk_3d_arr[:, kmask])
    DPks = Pks - np.mean(Pks, axis=0)

    Ni, Nk = Pks.shape
    cov = np.einsum("ij, ik->jk", DPks, DPks) / Ni
    return (Ni - 1) / (Ni - Nk - 2) * cov


#########################
# Observational effects #
#########################
sampling_resol = mock.map_sampling()
gridding_compensation = mock.gridding_compensation()
beam_attenutation = mock.beam_attenuation()

##################################
# Compute PCA reduction estimate #
##################################
data = np.load(sims_file)
Rmat_arr = data["R_mat"]

xarr = mock.cosmo.comoving_distance(mock.z_ch).value
comov_dist = xarr.max() - xarr.min()
F_mat = dft_matrix(mock.nu.shape[0])
R_mat_fourier = np.zeros(Rmat_arr.shape, dtype="complex")
for i, R_i in enumerate(Rmat_arr):
    R_mat_fourier[i] = F_mat @ R_i @ np.conj(F_mat).T / len(R_i)
# naive approximation along los
k_para_pseudo = np.fft.fftfreq(mock.nu.size, d=comov_dist / mock.nu.size) * 2 * np.pi

# xx,_ = np.meshgrid(np.fft.fftshift(k_para_pseudo),np.fft.fftshift(k_para_pseudo))
xx = np.fft.fftshift(k_para_pseudo)
xarr, yarr = np.meshgrid(mock.k_para, mock.k_para)

nnmask = ~(np.diag(Rmat_arr.mean(0)) == 0.0)
renorm = nnmask.mean() ** -1

signal_loss = np.diagonal(R_mat_fourier.mean(0) @ np.conj(R_mat_fourier.mean(0)).T).real
signal_loss = renorm * signal_loss

tf_interp = interp1d(
    np.fft.fftshift(k_para_pseudo),
    np.fft.fftshift(signal_loss),
    bounds_error=None,
    fill_value="extrapolate",
)
tf_test_1D = tf_interp(mock.k_para)
