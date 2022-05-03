import numba as nb
import numpy as np

dir = "L:/880_FLIM/paula_zhu/hela/dish2_CCCP/"

print('ok startin program...')

tpc_image = np.load(dir+"tpc_image1.npy")

print('tpc_image loaded...')

# @nb.njit(parallel=True)
def as_strided2d(a, K):
    b = a
    view = np.lib.stride_tricks.as_strided(a,
        shape=(K, K, a.shape[0] - (K-1), a.shape[1] - (K-1)),
        strides=a.strides * 2
    )
    b[2:-2, 2:-2] = view.sum(axis=(0, 1))
    return b

print('function defined')
# local binning size
K = 5

print('starting sr...')
# for all along time bin axis
tpc_image_sr = np.array([as_strided2d(slice, K) for slice in np.rollaxis(tpc_image, 2)])
print('finished, rolling axis...')
tpc_image_s = np.rollaxis(tpc_image_sr, 0, start=3)

print('np saving sdt result...')
np.save(dir+"tpc_image1_s", tpc_image_s)

print('calculating int image...')
int_image_s = tpc_image_s.sum(axis=2)
print('np saving int image of sdt result...')
np.save(dir+"int_image1_s", int_image_s)


