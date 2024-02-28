import numpy as np

from caliber.utils.joint_ecdf import JointECDF


def test_joint_ecdf():
    x = np.array([[1, 2], [3, 3], [0, 2]])
    z = np.array([[2, 2.5], [0.5, 0.5], [0.5, 2.5], [4, 4]])
    true_vals = np.array([2 / 3, 0, 1 / 3, 1])
    jecdf = JointECDF(x)
    assert np.all(jecdf.evaluate(z) == true_vals)
