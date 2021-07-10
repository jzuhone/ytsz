"""
Projection class for the Sunyaev-Zeldovich effect. Requires SZpack (at least
version 1.1.1) to be downloaded and installed:

http://www.chluba.de/SZpack/

For details on the computations involved please refer to the following references:

Chluba, Nagai, Sazonov, Nelson, MNRAS, 2012, arXiv:1205.5778
Chluba, Switzer, Nagai, Nelson, MNRAS, 2012, arXiv:1211.3206
"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from yt.utilities.physical_constants import clight, hcgs, kboltz, Tcmb
from yt.funcs import fix_axis
from yt.visualization.volume_rendering.off_axis_projection import \
    off_axis_projection
from yt.units import steradian
from yt.utilities.on_demand_imports import _astropy

import numpy as np

I0 = (2*(kboltz*Tcmb)**3/((hcgs*clight)**2)/steradian).in_units("MJy/steradian")


def setup_sunyaev_zeldovich_fields(ds, ftype):
    def _t_squared(field, data):
        return data[ftype, "optical_depth"]*data[ftype, "kT"]*data[ftype, "kT"]
    ds.add_field((ftype, "t_squared"), function=_t_squared, sampling_type="local",
                 units="keV**2/cm", force_override=True)

    def _beta_par_squared(field, data):
        return data[ftype, "beta_par"]**2/data[ftype, "optical_depth"]
    ds.add_field((ftype, "beta_par_squared"), function=_beta_par_squared,
                 sampling_type="local", units="1/cm", force_override=True)

    def _beta_perp_squared(field, data):
        ret = data[ftype, "optical_depth"]*data[ftype, "velocity_magnitude"]**2
        ret /= clight**2
        ret -= data[ftype, "beta_par_squared"]
        return ret
    ds.add_field((ftype, "beta_perp_squared"), function=_beta_perp_squared,
                 sampling_type="local", units="1/cm", force_override=True)

    def _t_beta_par(field, data):
        return data[ftype, "kT"]*data[ftype, "beta_par"]
    ds.add_field((ftype, "t_beta_par"), function=_t_beta_par,
                 sampling_type="local", units="keV/cm", force_override=True)

    def _t_sz(field, data):
        return data[ftype, "optical_depth"]*data[ftype, "kT"]
    ds.add_field((ftype, "t_sz"), function=_t_sz,
                 sampling_type="local", units="keV/cm", force_override=True)


def generate_beta_par(L, ftype):
    def _beta_par(field, data):
        vpar = data[ftype, "optical_depth"]*(data[ftype, "velocity_x"]*L[0] +
                                             data[ftype, "velocity_y"]*L[1] +
                                             data[ftype, "velocity_z"]*L[2])
        return vpar/clight
    return _beta_par


class SZImage(object):

    def __init__(self, data, nx, bounds):
        self.nx = nx
        self.bounds = bounds
        self.data = data
        self.dx = (bounds[1]-bounds[0])/nx

    def write_fits(self, filename, sky_scale=None, sky_center=None, overwrite=True,
                   **kwargs):
        r""" Export images to a FITS file. Writes the SZ distortion in all
        specified frequencies as well as the mass-weighted temperature and the
        optical depth. Distance units are in kpc, unless *sky_center*
        and *scale* are specified.

        Parameters
        ----------
        filename : string
            The name of the FITS file to be written.
        sky_scale : tuple
            Conversion between an angle unit and a length unit, if sky
            coordinates are desired, e.g. (1.0, "arcsec/kpc")
        sky_center : tuple, optional
            The (RA, Dec) coordinate in degrees of the central pixel. Must
            be specified with *sky_scale*.
        overwrite : boolean, optional
            If the file already exists, do we overwrite?

        Additional keyword arguments are passed to
        :meth:`~astropy.io.fits.HDUList.writeto`.

        Examples
        --------
        >>> # This example just writes out a FITS file with kpc coords
        >>> szimg.write_fits("SZbullet.fits", overwrite=False)
        >>> # This example uses sky coords
        >>> sky_scale = (1., "arcsec/kpc") # One arcsec per kpc
        >>> sky_center = (30., 45., "deg")
        >>> szimg.write_fits("SZbullet.fits", sky_center=sky_center, sky_scale=sky_scale)
        """
        from yt.visualization.fits_image import FITSImageData

        dx = self.dx.in_units("kpc")

        w = _astropy.pywcs.WCS(naxis=2)
        w.wcs.crpix = [0.5*(self.nx+1)]*2
        w.wcs.cdelt = [dx.v]*2
        w.wcs.crval = [0.0]*2
        w.wcs.cunit = ["kpc"]*2
        w.wcs.ctype = ["LINEAR"]*2

        fib = FITSImageData(self.data, fields=list(self.data.keys()), wcs=w)
        if sky_scale is not None and sky_center is not None:
            fib.create_sky_wcs(sky_center, sky_scale)
        fib.writeto(filename, overwrite=overwrite, **kwargs)

    def write_hdf5(self, filename):
        r"""Export the set of S-Z fields to a set of HDF5 datasets.

        Parameters
        ----------
        filename : string
            This file will be opened in "write" mode.

        Examples
        --------
        >>> szimg.write_hdf5("SZsloshing.h5")
        """
        import h5py
        for field, data in self.items():
            data.write_hdf5(filename, dataset_name=field)
        f = h5py.File(filename, "r+")
        f.flush()
        f.close()

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def __getitem__(self, key):
        return self.data[key]

    @property
    def shape(self):
        return self.nx, self.nx


class SZProjection(object):
    r""" Initialize a SZProjection object.

    Parameters
    ----------
    ds : ~yt.data_objects.static_output.Dataset 
        The dataset
    freqs : array_like
        The frequencies (in GHz) at which to compute the SZ spectral distortion.
    high_order : boolean, optional
        Should we calculate high-order moments of velocity and temperature?

    Examples
    --------
    >>> freqs = [90., 180., 240.]
    >>> szprj = SZProjection(ds, freqs, high_order=True)
    """
    def __init__(self, ds, freqs, high_order=False, no_rel=False,
                 no_kinetic=False, ftype="gas"):

        self.ds = ds
        self.num_freqs = len(freqs)
        self.ftype = ftype
        if no_kinetic:
            self.high_order = False
        else:
            self.high_order = high_order
        self.no_rel = no_rel
        self.no_kinetic = no_kinetic
        self.freqs = ds.arr(freqs, "GHz")
        self.xinit = hcgs*self.freqs.in_units("Hz")/(kboltz*Tcmb)
        self.freq_fields = ["%d_GHz" % (int(freq)) for freq in freqs]

    def on_axis(self, axis, center="c", width=(1, "unitary"), nx=800, source=None):
        r""" Make an on-axis projection of the SZ signal.

        Parameters
        ----------
        axis : integer or string
            The axis of the simulation domain along which to make the SZprojection.
        center : A sequence of floats, a string, or a tuple.
            The coordinate of the center of the image. If set to 'c', 'center' or
            left blank, the plot is centered on the middle of the domain. If set to
            'max' or 'm', the center will be located at the maximum of the
            ('gas', 'density') field. Centering on the max or min of a specific
            field is supported by providing a tuple such as ("min","temperature") or
            ("max","dark_matter_density"). Units can be specified by passing in *center*
            as a tuple containing a coordinate and string unit name or by passing
            in a YTArray. If a list or unitless array is supplied, code units are
            assumed.
        width : tuple or a float.
            Width can have four different formats to support windows with variable
            x and y widths.  They are:

            ==================================     =======================
            format                                 example
            ==================================     =======================
            (float, string)                        (10,'kpc')
            ((float, string), (float, string))     ((10,'kpc'),(15,'kpc'))
            float                                  0.2
            (float, float)                         (0.2, 0.3)
            ==================================     =======================

            For example, (10, 'kpc') requests a plot window that is 10 kiloparsecs
            wide in the x and y directions, ((10,'kpc'),(15,'kpc')) requests a
            window that is 10 kiloparsecs wide along the x axis and 15
            kiloparsecs wide along the y axis.  In the other two examples, code
            units are assumed, for example (0.2, 0.3) requests a plot that has an
            x width of 0.2 and a y width of 0.3 in code units.  If units are
            provided the resulting plot axis labels will use the supplied units.
        nx : integer, optional
            The dimensions on a side of the projection image.
        source : yt.data_objects.data_containers.YTSelectionContainer, optional
            If specified, this will be the data source used for selecting regions to project.

        Examples
        --------
        >>> szprj.on_axis("y", center="max", width=(1.0, "Mpc"), source=my_sphere)
        """
        axis = fix_axis(axis, self.ds)
        ctr, dctr = self.ds.coordinates.sanitize_center(center, axis)
        width = self.ds.coordinates.sanitize_width(axis, width, None)
        res = (nx, nx)

        L = np.zeros(3)
        L[axis] = 1.0

        beta_par = generate_beta_par(L, self.ftype)
        self.ds.add_field((self.ftype, "beta_par"), function=beta_par, units="1/cm",
                          sampling_type='local', force_override=True)
        setup_sunyaev_zeldovich_fields(self.ds, self.ftype)
        proj = self.ds.proj((self.ftype, "optical_depth"), axis, center=ctr,
                            data_source=source)
        frb = proj.to_frb(width[0], nx, height=width[1])
        tau = frb[self.ftype, "optical_depth"]
        Te = frb[self.ftype, "t_sz"]/tau
        if self.no_kinetic:
            bpar = np.zeros(res)
        else:
            bpar = frb[self.ftype, "beta_par"]/tau
        omega1 = frb[self.ftype, "t_squared"]/tau/(Te*Te) - 1.
        bperp2 = np.zeros(res)
        sigma1 = np.zeros(res)
        kappa1 = np.zeros(res)
        if self.high_order:
            bperp2 = frb[self.ftype, "beta_perp_squared"]/tau
            sigma1 = frb[self.ftype, "t_beta_par"]/tau/Te - bpar
            kappa1 = frb[self.ftype, "beta_par_squared"]/tau - bpar*bpar

        nx, ny = frb.buff_size
        bounds = frb.bounds

        data = self._compute_intensity(tau, Te, bpar, omega1, sigma1,
                                       kappa1, bperp2)

        self.ds.field_info.pop(("gas", "beta_par"))

        return SZImage(data, nx, bounds)

    def off_axis(self, L, center="c", width=(1.0, "unitary"), depth=(1.0,"unitary"),
                 nx=800, north_vector=None, source=None):
        r""" Make an off-axis projection of the SZ signal.

        Parameters
        ----------
        L : array_like
            The normal vector of the projection.
        center : A sequence of floats, a string, or a tuple.
            The coordinate of the center of the image. If set to 'c', 'center' or
            left blank, the plot is centered on the middle of the domain. If set to
            'max' or 'm', the center will be located at the maximum of the
            ('gas', 'density') field. Centering on the max or min of a specific
            field is supported by providing a tuple such as ("min","temperature") or
            ("max","dark_matter_density"). Units can be specified by passing in *center*
            as a tuple containing a coordinate and string unit name or by passing
            in a YTArray. If a list or unitless array is supplied, code units are
            assumed.
        width : tuple or a float.
            Width can have four different formats to support windows with variable
            x and y widths.  They are:

            ==================================     =======================
            format                                 example
            ==================================     =======================
            (float, string)                        (10,'kpc')
            ((float, string), (float, string))     ((10,'kpc'),(15,'kpc'))
            float                                  0.2
            (float, float)                         (0.2, 0.3)
            ==================================     =======================

            For example, (10, 'kpc') requests a plot window that is 10 kiloparsecs
            wide in the x and y directions, ((10,'kpc'),(15,'kpc')) requests a
            window that is 10 kiloparsecs wide along the x axis and 15
            kiloparsecs wide along the y axis.  In the other two examples, code
            units are assumed, for example (0.2, 0.3) requests a plot that has an
            x width of 0.2 and a y width of 0.3 in code units.  If units are
            provided the resulting plot axis labels will use the supplied units.
        depth : A tuple or a float
            A tuple containing the depth to project through and the string
            key of the unit: (width, 'unit').  If set to a float, code units
            are assumed
        nx : integer, optional
            The dimensions on a side of the projection image.
        north_vector : a sequence of floats
            A vector defining the 'up' direction in the plot.  This
            option sets the orientation of the slicing plane.  If not
            set, an arbitrary grid-aligned north-vector is chosen.
        source : yt.data_objects.data_containers.YTSelectionContainer, optional
            If specified, this will be the data source used for selecting regions 
            to project.

        Examples
        --------
        >>> L = np.array([0.5, 1.0, 0.75])
        >>> szprj.off_axis(L, center="c", width=(2.0, "Mpc"))
        """
        wd = self.ds.coordinates.sanitize_width(L, width, depth)
        w = tuple(el.in_units('code_length').v for el in wd)
        ctr, dctr = self.ds.coordinates.sanitize_center(center, L)
        res = (nx, nx)

        if source is None:
            source = self.ds

        beta_par = generate_beta_par(L, self.ftype)
        self.ds.add_field((self.ftype, "beta_par"), function=beta_par, units="1/cm",
                          sampling_type="local", force_override=True)
        setup_sunyaev_zeldovich_fields(self.ds, self.ftype)

        tau = off_axis_projection(source, ctr, L, w, res, (self.ftype, "optical_depth"),
                                  north_vector=north_vector)
        Te = off_axis_projection(source, ctr, L, w, res, "t_sz",
                                 north_vector=north_vector)/tau
        if self.no_kinetic:
            bpar = np.zeros(res)
        else:
            bpar = off_axis_projection(source, ctr, L, w, res, (self.ftype, "beta_par"),
                                       north_vector=north_vector)/tau
        omega1 = off_axis_projection(source, ctr, L, w, res, (self.ftype, "t_squared"),
                                     north_vector=north_vector)/tau
        omega1 = omega1/(Te*Te) - 1.
        if self.high_order:
            bperp2 = off_axis_projection(source, ctr, L, w, res, (self.ftype, "beta_perp_squared"),
                                         north_vector=north_vector)/tau
            sigma1 = off_axis_projection(source, ctr, L, w, res, (self.ftype, "t_beta_par"),
                                         north_vector=north_vector)/tau
            sigma1 = sigma1/Te - bpar
            kappa1 = off_axis_projection(source, ctr, L, w, res, (self.ftype, "beta_par_squared"),
                                         north_vector=north_vector)/tau
            kappa1 -= bpar
        else:
            bperp2 = np.zeros(res)
            sigma1 = np.zeros(res)
            kappa1 = np.zeros(res)

        bounds = (-0.5*wd[0], 0.5*wd[0], -0.5*wd[1], 0.5*wd[1])

        data = self._compute_intensity(tau, Te, bpar, omega1, sigma1,
                                       kappa1, bperp2)

        self.ds.field_info.pop((self.ftype, "beta_par"))

        return SZImage(data, nx, bounds)

    def _compute_intensity(self, tau, Te, bpar, omega1, sigma1, kappa1, bperp2):
        try:
            import SZpack
        except ImportError:
            raise ImportError("Could not import SZpack's Python bindings! Check your install!")

        # Bad hack, but we get NaNs if we don't do something like this
        small_beta = np.abs(bpar) < 1.0e-20
        bpar[small_beta] = 1.0e-20

        signal = SZpack.compute_combo_means_map(self.xinit, np.asarray(tau, order="C"),
                                                np.asarray(Te, order="C"),
                                                np.asarray(bpar, order="C"),
                                                np.asarray(omega1, order="C"),
                                                np.asarray(sigma1, order="C"),
                                                np.asarray(kappa1, order="C"),
                                                np.asarray(bperp2, order="C"))

        data = {}

        for i, field in enumerate(self.freq_fields):
            data[field] = I0*self.xinit[i]**3*signal[i,:,:]
        data["Tau"] = self.ds.arr(tau, "dimensionless")
        data["TeSZ"] = self.ds.arr(Te, "keV")

        return data
