import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18
from classy import Class
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.signal import find_peaks
from scipy.special import roots_legendre, legendre

from utils import add_boundary_knots

"""
Design conventions
------------------

All callable functions in this module follow a consistent shape convention:

- Functions of z alone return arrays with shape `z.shape`
- Functions of (k, z) return arrays with shape `(*k.shape, *z.shape)`
- Functions of (k, μ) require broadcastable inputs and return the broadcasted shape
- Functions of (k, μ, z) return shape `(*broadcast(k, μ), *z.shape)`
- Scalar inputs always return scalars
"""


class Class_cosmo_model:
    K_LINEAR = 1e-3 #The scale at which lienar quantities like the growth rate are computed at
    Z_MAX = 9 #This should the number passed to classy - 1

    def __init__(self, cosmodict):
        # Class initialisation
        cosmo = Class()
        cosmo_input, class_input = self.transform_input_dict_to_class(cosmodict)
        self.cosmodict = cosmo_input
        cosmo.set(class_input)
        cosmo.compute()

        # Linear power spectrum
        Pk_inter, k_inter, z_inter = cosmo.get_pk_and_k_and_z(nonlinear=False)
        z_inter, Pk_inter = z_inter[::-1], Pk_inter[:,::-1] # Reorder z to be growing
        def Pk_class(k, z, grid=True):
            k = np.atleast_1d(k)

            z = np.atleast_1d(z)
            zs = z.shape
            z = z.flatten()
            indicies = np.searchsorted(z_inter, z)

            logPk = np.empty((*k.shape, *z.shape))
            for i, index in enumerate(indicies):
                zi = z[i]
                index = np.clip(index, 1, len(z_inter)-1)

                Pk_zinter = (
                    (
                        Pk_inter[:, index-1] * (z_inter[index] - zi)
                        + Pk_inter[:, index] * (zi - z_inter[index-1])
                    ) /
                    (z_inter[index] - z_inter[index-1])
                )
                natural = CubicSpline(np.log(k_inter), np.log(Pk_zinter), bc_type='natural')
                add_boundary_knots(natural)
                logPk[..., i] = natural(np.log(k))
            return np.exp(logPk).reshape((*k.shape, *zs))
        self.Pk_lin = Pk_class

        # Some simple functions and constants
        self.Hubble = cosmo.Hubble
        self.comoving_Distance = cosmo.comoving_distance
        self.rsdrag = cosmo.rs_drag

        self.D_lin_raw = cosmo.scale_independent_growth_factor

    def fill_cosmo_defaults(self, input_dict: dict)->dict:
        """Fill missing cosmological parameters with default values.

        Parameters
        ----------
        input_dict : dict
            Dictionary containing cosmological parameters. Missing entries will be
            replaced with default values.

        Returns
        -------
        dict
            Complete cosmological parameter dictionary with defaults applied.
        """
        cosmo_dict = {
        "tau" : 0.0561,
        "h" : 0.6766,
        "Neff" : 3.046,
        "neutrino_mass" : 0.06,
        "omega_cold" : 0.30966,
        "omega_baryon" : 0.04897,
        "w0" : -1,
        "wa" : 0.0,
        "As" : 2.105209331337507e-09,
        "ns" : 0.9665,
        } # Defaults for CAMB

        cosmo_dict.update(input_dict)
        return cosmo_dict

    def transform_input_dict_to_class(self, input_dict: dict)->tuple:
        """Translate input cosmology dictionary into CLASS input format.

        This function converts a cosmology dictionary used in the pipeline into the
        parameter format required by the CLASS Boltzmann solver. It follows the
        neutrino prescription of [2303.09451].

        Parameters
        ----------
        input_dict : dict
            Dictionary of cosmological parameters.

        Returns
        -------
        tuple
            (cosmo_dict, class_input) where:
            - cosmo_dict : dict
                Cleaned cosmology dictionary used internally
            - class_input : dict
                Parameter dictionary passed to CLASS
        """
        #hardcoded
        T_cmb = Planck18.Tcmb(0).to(u.K).value
        Omegak = 0.0

        # Transformation constants
        Neff_fid = 3.044
        g = Neff_fid/3
        T_ncdm = (4/11)**(1/3) * g**(1/4)

        # input for CAMB
        cosmo_dict = self.fill_cosmo_defaults(input_dict)
        cosmo_dict_start = cosmo_dict.copy()

        # Class input
        Nur = cosmo_dict.pop("Neff") - Neff_fid/3
        Omeganu = cosmo_dict.pop("neutrino_mass") / 94.07 * g**(3/4) / cosmo_dict["h"]**2
        Omegacdm = cosmo_dict.pop("omega_cold") - cosmo_dict["omega_baryon"] - Omeganu

        class_input = {
            #Primordial Pk
            "A_s": cosmo_dict.pop("As"),
            "n_s": cosmo_dict.pop("ns"),
            #CMB
            "T_cmb": T_cmb,
            "tau_reio": cosmo_dict.pop("tau"),
            "reio_parametrization": "reio_camb",
            "Omega_b": cosmo_dict.pop("omega_baryon"),
            #Neutrinos
            "N_ur": Nur,
            "N_ncdm": 1,
            "T_ncdm": T_ncdm,
            "Omega_ncdm": Omeganu,
            #Background
            "h":cosmo_dict.pop("h"),
            "Omega_cdm": Omegacdm,
            "Omega_Lambda":0.0,
            "w0_fld":cosmo_dict.pop("w0"),
            "wa_fld":cosmo_dict.pop("wa"),
            "Omega_k":Omegak,
            #Class output
            "output":"mTk, mPk",
            "P_k_max_1/Mpc": 5,
            "z_max_pk": self.Z_MAX + 1,
        }
        if not cosmo_dict:
            print(cosmo_dict)

        return cosmo_dict_start, class_input

    def Dk_lin(self, k:np.ndarray, z:np.ndarray) -> np.ndarray:
        """Compute the dimensionless linear matter power spectrum.

        This returns the dimensionless power spectrum:
            Δ²(k) = k³ P(k) / (2π²)

        Parameters
        ----------
        k : array_like
            Wavenumbers in units of 1/Mpc.
        z : array_like
            Redshifts.

        Returns
        -------
        np.ndarray
            Dimensionless power spectrum with shape `(*k.shape, *z.shape)`.
        """
        z = np.atleast_1d(z)
        Pk_lin = self.Pk_lin(k, z, grid=True)
        Dk_lin = 4 * np.pi * Pk_lin * ((k / 2 / np.pi)**3)[..., *(z.ndim * (None, ))]
        return Dk_lin

    def P_nw_shape(self, k: np.ndarray) -> np.ndarray:
        """Eisenstein--Hu no-wiggle transfer function (unnormalised).

        This provides a smooth approximation to the power spectrum shape without BAO
        oscillations.

        Parameters
        ----------
        k : array_like
            Wavenumbers in units of 1/Mpc.

        Returns
        -------
        np.ndarray
            Smooth power spectrum shape (unnormalised).
        """
        cosmo_inputs = self.cosmodict

        # Get cosmologyical quantities for the fit
        h = cosmo_inputs["h"]
        wm = cosmo_inputs["omega_cold"] * h**2
        wb = cosmo_inputs["omega_baryon"] * h**2

        # This should be changed to the actuall CMB background temp
        theta = Planck18.Tcmb(0).to(u.K).value / 2.7

        rb = wb / wm
        ns = cosmo_inputs["ns"]

        k = k# .to(u.Mpc**-1).value #input has to be in units of 1/Mpc
        s = 44.5 * np.log(9.83 / wm) / np.sqrt(1 + 10 * wb ** (3 / 4))
        alpha = 1 - 0.328 * np.log(431 * wm) * rb + 0.38 * np.log(22.3 * wm) * rb**2

        Gamma = (wm / h) * (alpha + (1 - alpha) / (1 + (0.43 * k * s) ** 4))
        q = k / h * theta**2 / Gamma

        L0 = np.log(2 * np.e + 1.8 * q)
        C0 = 14.2 + 731 / (1 + 62.5 * q)
        T_nw = L0 / (L0 + C0 * q**2)
        return T_nw**2 * k**ns

    def D_lin(self, z):
        """Compute the linear growth factor D(z).

        Parameters
        ----------
        z : float or array_like
            Redshift(s).

        Returns
        -------
        float or np.ndarray
            Linear growth factor evaluated at z.
        """
        D = np.sqrt(self.Pk_lin(self.K_LINEAR, z, grid=False) / self.Pk_lin(self.K_LINEAR, 0, grid=False))
        return np.squeeze(D)

    def f_lin(self, z):
        """Compute the linear growth rate f(z) = d ln D / d ln a.

        Parameters
        ----------
        z : float or array_like
            Redshift(s).

        Returns
        -------
        float or np.ndarray
            Growth rate evaluated at z.
        """
        zinternal = np.linspace(0, self.Z_MAX)
        ainternal = 1 / (1 + zinternal)
        Dinternal = self.D_lin(zinternal)
        f = UnivariateSpline(np.log(ainternal)[::-1], np.log(Dinternal)[::-1]).derivative(1)(
            np.log(1 / (1 + z))
        )
        return f

    def Pk_nw(self, k: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute the smooth (no-wiggle) matter power spectrum.

        The BAO oscillations are removed using a spline-based smoothing procedure.

        Parameters
        ----------
        k : array_like
            Wavenumbers in units of 1/Mpc.
        z : array_like
            Redshifts.

        Returns
        -------
        np.ndarray
            Smooth power spectrum with BAO features removed.
        """
        z = np.atleast_1d(z)
        zs = z.shape
        z = z.flatten()

        # wave number grids
        kmin_loc = 1e-3
        kmax_loc = 10
        loc_samples = 800
        width = 1.5

        kgrid_savgol = np.geomspace(kmin_loc, kmax_loc, loc_samples)
        logkgrid_savgol = np.log(kgrid_savgol)

        P = self.Pk_lin(kgrid_savgol, z)
        P_shape = self.P_nw_shape(kgrid_savgol)
        P_shapeless = (P / P_shape[:, None])

        P_reshape = self.P_nw_shape(k)

        Psmoothed = np.empty((*k.shape, len(z)))
        for iz, zi in enumerate(z):
            natural = CubicSpline(logkgrid_savgol, P_shapeless[:, iz], bc_type="natural")
            Pprime_inter = natural(logkgrid_savgol, nu=1)

            ilogkmin = np.argmin(np.abs(kgrid_savgol - kmin_loc * width))
            ilogkmax = np.argmin(np.abs(kgrid_savgol - kmax_loc / width))

            logkpeaks, _ = find_peaks(Pprime_inter)
            logkvalleys, _ = find_peaks(-Pprime_inter)

            ipeaks = [
                *range(ilogkmin),
                *logkpeaks[(logkpeaks > ilogkmin) & (logkpeaks < ilogkmax)],
                *range(ilogkmax, loc_samples),
            ]
            natural = CubicSpline(logkgrid_savgol[ipeaks], P_shapeless[ipeaks, iz], bc_type="natural")
            add_boundary_knots(natural) # Add power law extrapolation
            Psl_peaks = natural(np.log(k))

            ivalleys = [
                *range(ilogkmin),
                *logkvalleys[(logkvalleys > ilogkmin) & (logkvalleys < ilogkmax)],
                *range(ilogkmax, loc_samples),
            ]
            natural = CubicSpline(logkgrid_savgol[ivalleys], P_shapeless[ivalleys, iz], bc_type="natural")
            add_boundary_knots(natural) # Add power law extrapolation
            Psl_valleys = natural(np.log(k))

            Psmoothed[..., iz] = 0.5 * (Psl_peaks + Psl_valleys) * P_reshape

        return Psmoothed.reshape((*k.shape, *zs,))

    def Pk_wiggle(self, k:np.ndarray, z:np.ndarray) -> np.ndarray:
        """Compute the BAO wiggle component of the power spectrum.

        Defined as:
            P_wiggle = P_lin - P_nw

        Parameters
        ----------
        k : array_like
            Wavenumbers in units of 1/Mpc.
        z : array_like
            Redshifts.

        Returns
        -------
        np.ndarray
            BAO oscillatory component of the power spectrum.
        """
        k = np.atleast_1d(k)
        z = np.atleast_1d(z)

        Pk_mm = self.Pk_lin(k, z, grid=True).reshape((*k.shape, *z.shape))
        Pk_nw = self.Pk_nw(k, z)
        return Pk_mm - Pk_nw

    def sigmav(self, z: np.ndarray):
        """Compute the velocity dispersion σ_v.

        This corresponds to the variance of the linear velocity divergence field.

        Parameters
        ----------
        z : array_like
            Redshifts.

        Returns
        -------
        np.ndarray
            Velocity dispersion at each redshift.
        """
        z = np.atleast_1d(z)
        alpha = 3
        t = np.linspace(0, 1, 200)
        t = t[1:-1]
        k = (1 / t - 1)**alpha
        ret = (3 * k**2 * t * (1 - t))[..., *(z.ndim * (None, ))]
        Dk = self.Dk_lin(k[::-1], z)[::-1]
        integrand = alpha * Dk / ret

        return np.sqrt(np.trapz(integrand, t, axis=0))

    def Pk_QNL(self, k:np.ndarray, mu:np.ndarray, z:np.ndarray, sigmaV:np.ndarray) -> np.ndarray:
        """Compute the quasi-nonlinear matter power spectrum.

        This uses the dewiggling formalism, where BAO oscillations are damped by
        large-scale bulk flows.

        Parameters
        ----------
        k : array_like
            Wavenumbers in 1/Mpc.
        mu : array_like
            Cosine of the angle to the line of sight.
        z : array_like
            Redshifts.
        sigmaV : array_like
            Velocity dispersion, must match shape of z.

        Returns
        -------
        np.ndarray
            Quasi-nonlinear matter power spectrum.

        Notes
        -----
        k and mu must be broadcastable. z and sigmaV must have identical shapes.
        """
        z = np.atleast_1d(z)
        sigmaV = np.atleast_1d(sigmaV)
        assert z.shape == sigmaV.shape
        f = self.f_lin(z)

        Pk_nw = self.Pk_nw(k, z)
        Pk_wiggle = self.Pk_wiggle(k, z)

        # extend dims in z (maybe we want to do this on a grid for AP effects?)
        f = f[*(np.ndim(k * mu) * (None, )), ...]
        sigmaV = sigmaV[*(np.ndim(k * mu) * (None, )), ...]
        k = k[..., *(np.ndim(z) * (None, ))]
        mu = mu[..., *(np.ndim(z) * (None, ))]

        gmu = sigmaV**2 * (1 - mu**2 + mu**2 * (1 + f)**2)
        Pk_QNL = (Pk_nw + np.exp(-gmu * k**2) * Pk_wiggle)
        return Pk_QNL


class power_spectrum_from_baopars:
    def __init__(self, cosmo_fid:Class_cosmo_model, bao_pars:dict, desi_like_bao=False):
        self.cosmo_fid = cosmo_fid

        if "alpha_Iso" in bao_pars.keys():
            self.alpha_AP = bao_pars.get("alpha_AP", 1)
            self.alpha_Iso = bao_pars["alpha_Iso"]

            self.alpha_parr = self.alpha_Iso * np.power(self.alpha_AP, 2/3)
            self.alpha_perp = self.alpha_Iso * np.power(self.alpha_AP, -1/3)
        elif "alpha_parr" in bao_pars.keys() and "alpha_perp" in bao_pars.keys():
            self.alpha_parr = bao_pars["alpha_parr"]
            self.alpha_perp = bao_pars["alpha_perp"]

            self.alpha_Iso = self.alpha_perp**(2 / 3) * self.alpha_parr**(1 / 3)
            self.alpha_AP =  self.alpha_parr / self.alpha_perp

        self.sigma_v = bao_pars["sigma_v"]
        self.sigma_p = bao_pars["sigma_p"]
        self.bias = bao_pars["bias"]
        self.bias_2 = bao_pars.get("bias_2", self.bias)

        self.desi_like_bao = desi_like_bao

    def convert_modes(self, k, mu):
        """Apply Alcock--Paczyński (AP) scaling to Fourier modes.

        Parameters
        ----------
        k : array_like
            Observed wavenumbers.
        mu : array_like
            Cosine of the angle to the line of sight.

        Returns
        -------
        tuple
            (k', mu') transformed to the trial cosmology.
        """
        kparr_prime = k * mu / self.alpha_parr
        kperp_prime = k * np.sqrt(np.clip(1-mu**2, 0, 1)) / self.alpha_perp

        k_prime = np.sqrt(kperp_prime**2 + kparr_prime**2)
        mu_prime = kparr_prime / k_prime
        return k_prime, mu_prime


    def rsd(self, b1, mu, z, b2=None):
        if b2==None:
            b2 = b1

        z = np.atleast_1d(z)
        nz = np.ndim(z)
        rsd = (
            (b1 + self.cosmo_fid.f_lin(z) * mu[..., *(nz*(None,))]**2)
            * (b2 + self.cosmo_fid.f_lin(z) * mu[..., *(nz*(None,))]**2)
        )
        return rsd


    def powerspectrum_nw(self,k, mu, z, which="1"):
        """Compute the smooth no-wiggle part of tracer power spectra in redshift space.

        Includes:
        - Alcock--Paczyński scaling (if asked for)
        - Linear RSD (Kaiser)
        - Finger-of-God damping

        Parameters
        ----------
        k : array_like
            Wavenumbers.
        mu : array_like
            Cosine of angle to the line of sight.
        z : array_like
            Redshifts.
        which : str
            Which tracer the power spectrum should be computed of.
            Either "1", "2", or "both"

        Returns
        -------
        np.ndarray
            Model power spectrum P(k, μ, z).
        z = np.atleast_1d(z)
        """
        k = np.atleast_1d(k)
        z = np.atleast_1d(z)
        nz = np.ndim(z)

        if self.desi_like_bao:
            fAP = 1
            k_prime, mu_prime = k, mu
        else:
            fAP = 1 / (self.alpha_Iso**3) # Very degenerate with b
            k_prime, mu_prime = self.convert_modes(k, mu)

        Pnw = self.cosmo_fid.Pk_nw(k_prime, z).reshape((*k.shape, *z.shape))
        fFOG = (
            (1 + 1 / 2 * (k_prime * mu_prime)[..., *(nz*(None,))]**2 * self.sigma_p**2)**-2
        ).reshape((*k.shape, *z.shape))
        P_mm_real = fAP * fFOG * Pnw

        output = []
        if which in ["1", "both"]:
            fRSD = self.rsd(self.bias, mu_prime, z)
            output.append((fRSD * P_mm_real).squeeze())
        if which == "both":
            fRSD = self.rsd(self.bias, mu_prime, z, self.bias_2)
            output.append((fRSD * P_mm_real).squeeze())
        if which in ["2", "both"]:
            fRSD = self.rsd(self.bias_2, mu_prime, z)
            output.append((fRSD * P_mm_real).squeeze())

        return  np.array(output).squeeze()

    def powerspectrum_w(self,k, mu, z, which="1"):
        """Compute the wiggle part of tracer power spectra in redshift space.

        Includes:
        - Alcock--Paczyński scaling
        - Linear RSD (Kaiser)
        - Finger-of-God damping
        - BAO damping

        Parameters
        ----------
        k : array_like
            Wavenumbers.
        mu : array_like
            Cosine of angle to the line of sight.
        z : array_like
            Redshifts.
        which : str
            Which tracer the power spectrum should be computed of.
            Either "1", "2", or "both"

        Returns
        -------
        np.ndarray
            Model power spectrum P(k, μ, z).
        z = np.atleast_1d(z)
        """
        k = np.atleast_1d(k)
        z = np.atleast_1d(z)
        nz = np.ndim(z)

        k_prime, mu_prime = self.convert_modes(k, mu)

        if self.desi_like_bao:
            fAP = 1
        else:
            fAP = 1 / (self.alpha_Iso**3)

        Pwiggle = self.cosmo_fid.Pk_wiggle(k_prime, z).reshape((*k.shape, *z.shape))
        fFOG = (
            (1 + 1 / 2 * (k_prime * mu_prime)[..., *(nz*(None,))]**2 * self.sigma_p**2)**-2
        ).reshape((*k.shape, *z.shape))
        gmu = self.sigma_v**2 * (
            (1 - mu_prime[..., *(nz*(None,))]**2)
            + (1 + self.cosmo_fid.f_lin(z))**2 * mu_prime[..., *(nz*(None,))]**2
        )
        fDamp = np.exp(-k_prime[..., *(nz*(None,))]**2 * gmu)

        P_mm_wiggle = fAP * fDamp * fFOG * Pwiggle
        output = []
        if which in ["1", "both"]:
            fRSD = self.rsd(self.bias, mu_prime, z)
            output.append((fRSD * P_mm_wiggle).squeeze())
        if which == "both":
            fRSD = self.rsd(self.bias, mu_prime, z, self.bias_2)
            output.append((fRSD * P_mm_wiggle).squeeze())
        if which in ["2", "both"]:
            fRSD = self.rsd(self.bias_2, mu_prime, z)
            output.append((fRSD * P_mm_wiggle).squeeze())

        return  np.array(output).squeeze()

    def powerspectrum(self,k, mu, z, which="1"):
        """Compute the galaxy and termperature power spectra.

        Includes:
        - Alcock--Paczyński scaling
        - Linear RSD (Kaiser)
        - Finger-of-God damping
        - BAO damping

        Parameters
        ----------
        k : array_like
            Wavenumbers.
        mu : array_like
            Cosine of angle to the line of sight.
        z : array_like
            Redshifts.
        which : str
            Which tracer the power spectrum should be computed of.
            Either "1", "2", or "both"

        Returns
        -------
        np.ndarray
            Model power spectrum P(k, μ, z).
        z = np.atleast_1d(z)
        """
        pwiggle = self.powerspectrum_w(k, mu, z, which=which)
        pnowiggle = self.powerspectrum_nw(k, mu, z, which=which)
        return pwiggle + pnowiggle

    def powerspectrum_multipoles(self, k, ell, z):
        """Compute multipoles of the power spectrum.

        Multipoles are defined as:
            P_ell(k) = (2ell+1)/2 ∫_{-1}^{1} dμ P(k,μ) L_ell(μ)

        The integral is evaluated using Gauss--Legendre quadrature.

        Parameters
        ----------
        k : array_like
            Wavenumbers.
        ell : array_like
            Multipole orders (e.g. [0, 2, 4]).
        z : array_like
            Redshifts.

        Returns
        -------
        np.ndarray
            Multipole power spectra with shape `(*k.shape, *ell.shape, *z.shape)`.
        """
        k = np.atleast_1d(k)
        ks = k.shape
        k = k.flatten()

        ell = np.atleast_1d(ell)
        z = np.atleast_1d(z)

        mu, wi = roots_legendre(9)
        raise NotImplementedError # need to change this to new inoput type
        power = self.powerspectrum(k[:, None, None], mu[None, :, None], z)

        output = []
        for elli in ell:
            if elli % 2 != 0:
                output.append(np.zeros((*k.shape, *z.shape)))
                continue

            output.append(
                (2 * elli + 1) / 2 * np.sum(
                    power
                    * wi[None, :, None]
                    * legendre(elli)(mu)[None, :, None],
                    axis=1
                )
            )
        output = np.array(output).transpose((1, 0, 2)).reshape((*ks, *ell.shape, *z.shape)).squeeze()
        return output


    def broadband(k, ell, broadbandpars):
        """Compute broadband polynomial contribution to multipoles.

        The broadband is modeled as:
            P_ell^bb(k) = Σ_i a_{ell,i} k^{p_i}

        with powers p_i = [-2, -1, 0, 1, 2].

        Parameters
        ----------
        k : array_like
            Wavenumbers.
        ell : array_like
            Multipole orders.
        broadbandpars : dict
            Dictionary containing coefficients a_{ell,i}.

        Returns
        -------
        np.ndarray
            Broadband contribution for each multipole.
        """
        k = np.atleast_1d(k)
        ell = np.atleast_1d(ell)

        powers = np.array([-2, -1, 0, 1, 2])
        labels = np.arange(len(powers))

        kp = k[..., None]**powers

        output = []
        for elli in ell:
            ai = np.array([broadbandpars[f"a_{elli}_{i}"] for i in labels])
            output.append(
                np.einsum("...i, i", kp, ai)
            )
        return np.array(output).T.squeeze()