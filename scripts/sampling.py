import logging
import os
import sys
from time import time

import corner
import niceplots
import numpy as np
import seaborn as sns
from nautilus import Prior, Sampler
from scipy import linalg
from scipy.stats import binned_statistic

from meer21cm.power import get_modelpk_conv
from meer21cm.util import dft_matrix

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)
from classy_wraper_for_m21cm import *
from specs_v2 import *

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

Cs = sns.color_palette("colorblind")
niceplots.initPlot()

sims_file = "/home/sefa/Desktop/projects/meerklass/get_fg_reduction/power_spectra_hp_with_pca_batch1.npz"

###################
# Get mock object #
###################
logger.info("Starting Get mock object...")
_t0 = time()

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

logger.debug(f"Get mock object finished in {time() - _t0:.2f}s")

##########################
# Get scalecuts and grid #
##########################
logger.info("Starting Get scalecuts and grid...")
_t0 = time()

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


logger.debug(f"Get scalecuts and grid finished in {time() - _t0:.2f}s")

#########################
# Observational effects #
#########################
logger.info("Starting Observational effects...")
_t0 = time()

sampling_resol = mock.map_sampling()
gridding_compensation = mock.gridding_compensation()
beam_attenutation = mock.beam_attenuation()

logger.debug(f"Observational effects finished in {time() - _t0:.2f}s")

##################################
# Compute PCA reduction estimate #
##################################
logger.info("Starting Compute PCA reduction estimate...")
_t0 = time()

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

logger.debug(f"Compute PCA reduction estimate finished in {time() - _t0:.2f}s")

##############
# covariance #
##############
logger.info("Starting Covariance...")
_t0 = time()

Delta = 0.052  # Mpc**-1


def W(x):
    x = np.atleast_1d(x)
    W = np.zeros_like(x)
    W[np.abs(x) <= 1] = (4 - 6 * np.abs(x) ** 2 + 3 * np.abs(x) ** 3)[np.abs(x) <= 1]
    W[np.logical_and(1 < np.abs(x), np.abs(x) <= 2)] = (
        8 - 12 * np.abs(x) + 6 * np.abs(x) ** 2 - np.abs(x) ** 3
    )[np.logical_and(1 < np.abs(x), np.abs(x) <= 2)]
    return W.squeeze()


def build_design_matrix_desi(k, nblocks=1):
    X_block = np.vstack([W(k / Delta - n) for n in range(-1, 8)]).T
    X = linalg.block_diag(*[X_block] * nblocks)

    return X


X = build_design_matrix_desi(kcenters)

cov = get_1d_cov(data["phixgal_cleaned"])
invcov = np.linalg.inv(cov)
Fisher = linalg.inv(X.T @ invcov @ X)
invCmarg = invcov - invcov @ X @ Fisher @ X.T @ invcov

logger.debug(f"Covariance finished in {time() - _t0:.2f}s")

#################
# Fiducial data #
#################
logger.info("Starting Fiducial data...")
_t0 = time()

classcosmo = Class_cosmo_model({})

fid_BAO = {
    "alpha_Iso": 1,
    "alpha_AP": 1,
    "sigma_p": 1.6,  # Mpc
    "sigma_v": 1.6,  # Mpc
    "bias": 1.5,
    "bias_2": 1.5,
}

baopars = fid_BAO.copy()
baopars["alpha_Iso"] = 1.0
fid_ps_obj = power_spectrum_from_baopars(classcosmo, baopars)

phixgal_fid = fid_ps_obj.powerspectrum(k, mu, mock.z, which="both")[1, ...]
phixgal_fid[k == 0] = 0.0

phixgal_fid_obs = (
    mock.average_hi_temp
    * gridding_compensation**2
    * beam_attenutation
    * sampling_resol
    * tf_test_1D
    * phixgal_fid
)
phixgal_fid_obs_conv = get_modelpk_conv(
    phixgal_fid_obs, mock.weights_1, mock.weights_field_2 * mock.weights_grid_2
)
phixgal_1d_fid = bin_Pk_1d(phixgal_fid_obs_conv)

logger.debug(f"Fiducial data finished in {time() - _t0:.2f}s")

##############
# Likelihood #
##############
logger.info("Starting Likelihood...")
_t0 = time()

fid_beam = mock.sigma_beam_ch
def get_model(params):
    ps = params.copy()

    if "beamfac" in ps:
        mock.sigma_beam_ch = fid_beam * ps.pop("beamfac")

    baopars = fid_BAO.copy()
    baopars.update(ps)
    ps_obj = power_spectrum_from_baopars(classcosmo, baopars)

    phixgal = np.zeros_like(k)
    phixgal[~(k==0)] = ps_obj.powerspectrum(k[~(k==0)], mu[~(k==0)], mock.z, which="both")[1, ...]

    phixgal_obs = (
        mock.average_hi_temp
        * gridding_compensation**2
        * beam_attenutation
        * sampling_resol
        * tf_test_1D
        * phixgal
    )
    phixgal_obs_conv = get_modelpk_conv(
        phixgal_obs,
        mock.weights_1, mock.weights_field_2 * mock.weights_grid_2
    )
    return bin_Pk_1d(phixgal_obs_conv)

def logprior(params):
    ps = params.copy()
    rvalue = 0
    if "beamfac" in ps:
        sigma = 0.2 / (2 * np.sqrt(2 * np.log(2)))
        mu = 1
        rvalue += -(ps.pop("beamfac") - mu)**2 / (2 * sigma**2)

    if "bias_2" in ps:
        sigma = 0.3
        mu = 1.5
        rvalue += -(ps.pop("bias_2") - mu)**2 / (2 * sigma**2)

    return rvalue

fudge_fac = 12
def loglike(params):
    phixgal_1d = get_model(params)
    Dphixgal_1d = phixgal_1d - phixgal_1d_fid
    chi2 = np.einsum("i, ij, j", Dphixgal_1d, invCmarg * fudge_fac, Dphixgal_1d)
    return -0.5 * chi2

def logpost(params):
    return loglike(params) + logprior(params)
    

###########
# Sampler #
###########
logger.info("Starting Sampler...")
_t0 = time()

prior = Prior()
prior.add_parameter("bias", (0.1, 10))
prior.add_parameter("bias_2", (0.1, 5))
prior.add_parameter("alpha_Iso", (0.01, 10))
prior.add_parameter("beamfac", (0.5, 1.5))
prior.add_parameter("sigma_p", (0.01, 15))

sampler = Sampler(prior, logpost, n_live=1000, filepath="../data/chain_v2_desi_like_fg.hdf5")

logger.debug(f"Sampler finished in {time() - _t0:.2f}s")

############
# Plotting #
############
logger.info("Starting Plotting...")
_t0 = time()

points, log_w, log_l = sampler.posterior()
fig = corner.corner(
    points,
    weights=np.exp(log_w),
    bins=20,
    labels=prior.keys,
    color="purple",
    plot_datapoints=False,
    range=np.repeat(0.999, len(prior.keys)),
)
fig.savefig("../data/chain_v2_desi_like_fg.png", dpi=300, bbox_inches="tight")

logger.debug(f"Plotting finished in {time() - _t0:.2f}s")
