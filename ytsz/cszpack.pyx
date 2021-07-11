import numpy as np
cimport numpy as np
cimport cython

# Grab function definitions directly from SZpack.h

cdef extern from "cszpack.h":
    double compute_SZ_signal_combo_means(double xo, double tau,
                                         double TeSZ, double betac_para,
                                         double omega1, double sigma,
                                         double kappa, double betac2_perp) nogil


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def compute_combo_means_map(np.ndarray[np.float64_t, ndim=1, mode="c"] xo,
                            np.ndarray[np.float64_t, ndim=2, mode="c"] tau,
                            np.ndarray[np.float64_t, ndim=2, mode="c"] TeSZ,
                            np.ndarray[np.float64_t, ndim=2, mode="c"] betac_para,
                            np.ndarray[np.float64_t, ndim=2, mode="c"] omega1,
                            np.ndarray[np.float64_t, ndim=2, mode="c"] sigma,
                            np.ndarray[np.float64_t, ndim=2, mode="c"] kappa,
                            np.ndarray[np.float64_t, ndim=2, mode="c"] betac2_perp):
    # This function generates a map at each frequency based on input maps of
    # various weighted quantities
    cdef int i, j, k
    cdef int nx = tau.shape[0]
    cdef int ny = tau.shape[1]
    cdef int nf = xo.size
    cdef np.ndarray[np.float64_t, ndim=3, mode="c"] I
    I = np.zeros((nf, ny, nx))
    for k in range(nf):
        for j in range(ny):
            for i in range(nx):
                I[k,j,i] = compute_SZ_signal_combo_means(xo[k], tau[j,i], TeSZ[j,i],
                                                         betac_para[j,i], omega1[j,i],
                                                         sigma[j,i], kappa[j,i],
                                                         betac2_perp[j,i])
    return I