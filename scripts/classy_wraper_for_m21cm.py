from astropy.cosmology import Planck18
from astropy import units as u
from classy import Class
from scipy.interpolate import UnivariateSpline, CubicSpline
from scipy.signal import find_peaks
import numpy as np

"""
The design process here assumes that all callable functions should follow the following paradigm:

- All functions that are callables of z alone should return a value with the shape of `z.shape`
- All functions that are callables of k and z should return a value with the shape of `(*k.shape, *z.shape)`
- All functions of k and $\mu$ alone should require that `k.shape` and `mu.shape` are broadcastable. The return value should have the combined shape.
- All Functions of k, $\mu$, and z should require that `k.shape` and `mu.shape` are broadcastable. The return value should have the shape `(*combined, *z.shape)`
- Finally, all functions should return a scalar if given a scalar input. 
"""

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
        """Fills an incomplete cosmology input with meer21cm defaults

        This function will take a dictionary for input for the meer21cm cosmology module and fill missing input with defaults

        Parameters
        ----------
        input_dict: dict
            dictionarry containing the kwargs for meer21cm `get_camb_pars` member function.

        Returns
        -------
        dict
            dictionarry with missing parameters default values
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
        """Translates a imput dict for the meer21cm cosmology module into class input
        
        This function follows the neutrino prescription of [2303.09451].

        Parameters
        ----------
        input_dict: dict
            dictionarry containing the kwargs for meer21cm `get_camb_pars` member function.

        Returns
        -------
        tuple[iterable, dict]
            dictionary of cosmological input, and dictionary to be passed to class `Class.set()` functions
        
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
        """The dimensionless linear matter power spectrum
        """
        z = np.atleast_1d(z)
        Pk_lin = self.Pk_lin(k, z, grid=True)
        Dk_lin = 4 * np.pi * Pk_lin * ((k / 2 / np.pi)**3)[..., *(z.ndim * (None, ))]
        return Dk_lin

    def P_nw_shape(self, k: np.ndarray) -> np.ndarray:
        """Unnormalised Eisenstein--Hu function for the no-wiggle power spectrum

        This function returns a fitting function for the no-wiggle power spectrum.

        Parameters
        ----------
        input_dict: dict
            dictionarry containing the kwargs for meer21cm `get_camb_pars` member function.

        k: float | np.ndarray
            wavenumbers in 1/Mpc units.
        
        Returns
        -------
        float | np.ndarray
            Powerspectrum approximation. If k is not scalar, the output will have the same shape.
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
        """Calculate the linear growth factor D(z)

        Parameters
        ----------
        z: float | np.ndarray
            The redshift of interest.

        Returns
        -------
        float | np.ndarray
            Growth factor at redshift z. If z is not scalar, D will have the same shape as z 
        """
        D = np.sqrt(self.Pk_lin(self.K_LINEAR, z, grid=False) / self.Pk_lin(self.K_LINEAR, 0, grid=False))
        return np.squeeze(D)

    def f_lin(self, z):
        """Calculate the linear growth rate f(z)
        
        Parameters
        ----------
        z: float | np.ndarray
            The redshift of interest.

        Returns
        -------
        float | np.ndarray
            Growth rate at redshift z. If z is not scalar, f will have the same shape as z """
        zinternal = np.linspace(0, self.Z_MAX)
        ainternal = 1 / (1 + zinternal)
        Dinternal = self.D_lin(zinternal)
        f = UnivariateSpline(np.log(ainternal)[::-1], np.log(Dinternal)[::-1]).derivative(1)(
            np.log(1 / (1 + z))
        )
        return f

    def Pk_nw(self, k: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Calculate the power spectrum at a specific redshift and wavenumber,
        after smoothing to remove baryonic acoustic oscillations (BAO).

        Parameters
        ----------
        k: float | np.ndarray
            An array of wavenumbers at which to compute the power spectrum.
        z: float | np.ndarray
            The redshift of interest.
            
        Returns
        -------
        float | np.array
            An array of smooth component power spectrum values corresponding to the
            input wavenumbers.
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
        """Compute the wiggle component of the BAO

        Substract the smooth broad-band component from the model matter power spectrum

        Parameters
        ----------
        k: float | np.ndarray
            An array of wavenumbers at which to compute the power spectrum.
        z: float | np.ndarray
            The redshift of interest.
            
        Returns
        -------
        float | np.array
            An array of power spectrum BAO wiggle values corresponding to the
            input wavenumbers.
        """
        k = np.atleast_1d(k)
        z = np.atleast_1d(z)
        
        Pk_mm = self.Pk_lin(k, z, grid=True).reshape((*k.shape, *z.shape))
        Pk_nw = self.Pk_nw(k, z)
        return Pk_mm - Pk_nw

    def sigmav(self, z: np.ndarray):
        """Computes the variance of the linear! velocity divergence field.

        Parameters
        ----------
        z: float | np.ndarray
            The redshift of interest.

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
        """Compute the QNL power spectrum of matter
        
        This function computes the quasi-nonlinear matter power spectrum using
        the dewiggling formalism. The smooth component is inferred from a cubic spline
        of inflection points.
        
        Parameters
        ----------
        k: float | np.ndarray
            An array of wavenumbers at which to compute the power spectrum.
        mu: float | np.ndarray
            The cosine of the angle between the LOS and the 3D k mode.
        z: float | np.ndarray
            The redshift of interest.
        sigmaV: float | np.ndarray
            The variance of the velocity dispersion field. Should have the same shape as z
        
        Returns
        -------
        float | np.ndarray
            The quasi-nonlinear power spectrum for a given modulus of k, mu and redshift.
            If the input is not scalar the resulting shape will be the fully broadcasted shape

        Notes
        -----
        The shapes of k and mu should be broadcastable. z and sigmaV need to have the same shape.
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