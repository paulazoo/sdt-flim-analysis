import numba as nb
import numpy as np

sdt_image = np.load("sdt_image.npy")

@nb.njit(parallel=True)
def nb_bin2d(a, K):
    m_bins = a.shape[0]//K
    n_bins = a.shape[1]//K
    res = np.zeros((m_bins, n_bins), dtype=np.float64)

    for k in nb.prange(m_bins*n_bins):
        i = k//m_bins
        j = k%m_bins
        for y in range(i*K+1, (i+1)*K):
            for x in range(j*K, (j+1)*K):
                a[i*K, x] += a[y,x]
        s=0.0
        for x in range(j*K, (j+1)*K):
            s+=a[i*K, x]
        res[i,j] = s
    return res

# local binning size
K = 5

# for all along time bin axis
sdt_im_br = np.array([nb_bin2d(slice, K) for slice in np.rollaxis(sdt_image, 2)])
sdt_im_b = np.rollaxis(sdt_im_br, 0, start=3)

# np.count_nonzero(int_image > 5)
np.save('sdt_im_b', sdt_im_b)
