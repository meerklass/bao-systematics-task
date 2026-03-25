# specifications for the simulation
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18
from meer21cm.telescope import dish_beam_sigma
from meer21cm.util import create_wcs, freq_to_redshift, redshift_to_freq
from scipy.interpolate import CubicSpline

from utils import add_boundary_knots

num_pix_x = 120
num_pix_y = 40

# z_min = 0.8
z_min = 0.6
z_max = 0.8
# z_max = 1.1
nu_min = redshift_to_freq(z_max)
nu_max = redshift_to_freq(z_min)
nu_resol = 132812.5
num_ch = int((nu_max - nu_min) / nu_resol)
nu_arr = np.linspace(nu_min, nu_min + (num_ch - 1) * nu_resol, num_ch)
z_ch = freq_to_redshift(nu_arr)

wcs = create_wcs(
    ra_cr=150,
    dec_cr=-2.5,
    ngrid=[num_pix_x, num_pix_y],
    resol=0.5,
)

ra_range = [125, 175]
dec_range = [-10.1, 5]

window_name = "blackmanharris"

# dndz_data = np.load("LRG_dndz.npz")
dndz_data = np.load("LRGELG_dndz.npz")
z_bin = dndz_data["z_bin"]
z_count = dndz_data["z_count"]
z_cen = (z_bin[:-1] + z_bin[1:]) / 2
dV_arr = Planck18.differential_comoving_volume(z_cen)

# LRG3, DESI DR1
# n_gal = 859824 / 5 / 1e9 #Mpc-3
# LRG2, DESI DR1
n_gal = 771875 / 4 / 1e9  # Mpc-3

k1dbins = np.linspace(0.003, 0.2, 25)[1:]
kperpbins = np.linspace(0, 0.048, 17)[2:]
kparabins = np.linspace(0, 0.5, 51)

#######################
# Detector Resolution #
#######################

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

