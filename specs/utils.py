import matplotlib.pyplot as plt
import numpy as np
from meer21cm.plot import plot_map

####################
# Ploting routines #
####################


def plot_cy_power(xbins, ybins, pdatacy, pmodcy, vmin_ratio, vmax_ratio):
    arr = np.array(
        [
            np.log10(pdatacy.mean(axis=0).T),
            np.log10(pmodcy.T),
        ]
    )
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    fig, axes = plt.subplots(1, 3)
    axes[0].pcolormesh(
        xbins,
        ybins,
        np.log10(pdatacy.mean(axis=0).T),
        vmin=vmin,
        vmax=vmax,
    )
    im = axes[1].pcolormesh(
        xbins,
        ybins,
        np.log10(pmodcy.T),
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, ax=axes[:-1], location="top", fraction=0.046, pad=0.04)
    im = axes[2].pcolormesh(
        xbins,
        ybins,
        (pdatacy.mean(axis=0).T) / (pmodcy.T),
        vmin=vmin_ratio,
        vmax=vmax_ratio,
        cmap="bwr",
    )
    plt.colorbar(im, ax=axes[2], location="top", fraction=0.046, pad=0.04)
    return fig


def plot_1d_power(
    keff,
    pdatad,
    pmodd,
    ratio_min,
    ratio_max,
):
    keff = np.array(keff)
    pdatad = np.array(pdatad)
    pmodd = np.array(pmodd)
    sel = keff == keff
    keff = keff[sel]
    pdatad = pdatad[:, sel]
    pmodd = pmodd[sel]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 5),
        sharex=True,
        height_ratios=[2, 1],
    )
    axes[0].errorbar(
        keff,
        pdatad.mean(axis=0) * keff,
        yerr=pdatad.std(axis=0) * keff,
        label="mock",
    )
    axes[0].plot(keff, pmodd * keff, label="model", ls="--")
    axes[0].set_ylim((pmodd * keff).min() * 0.7, (pmodd * keff).max() * 1.2)
    axes[0].legend()
    axes[1].errorbar(
        keff,
        (pdatad.mean(axis=0)) / (pmodd) - 1,
        yerr=(pdatad.std(axis=0)) / (pmodd),
    )
    axes[1].axhline(0, color="black", ls="--")
    axes[1].fill_between(
        np.linspace(keff.min() - 0.005, keff.max() + 0.005, 100),
        -0.05,
        0.05,
        color="black",
        alpha=0.2,
    )
    axes[1].set_xlim(keff.min() - 0.005, keff.max() + 0.005)
    axes[1].set_ylim(ratio_min, ratio_max)
    axes[1].legend()
    return fig


#########################
# Extrapolation routine #
#########################


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


