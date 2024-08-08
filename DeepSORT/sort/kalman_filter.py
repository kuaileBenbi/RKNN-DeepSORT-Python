# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.上一时刻的状态均值向量，通常包含位置和速度等信息
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.上一时刻的状态协方差矩阵，表示状态的不确定性

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        """
        在 predict 方法中，会使用位置和速度的标准差权重来计算过程噪声协方差矩阵。
        这里可以通过调整 _std_weight_position 和 _std_weight_velocity 参数
        来控制过程噪声（Q）的大小。可以根据实际情况适当调整这些权重。
        """
        # Calculate the speed and adjust noise factors dynamically
        speed = np.linalg.norm(
            mean[4:6]
        )  # Speed is calculated from velocity components
        position_noise_factor = 1.0 + speed / 10.0  # Dynamic adjustment based on speed
        velocity_noise_factor = 1.0 + speed / 20.0

        # std_pos = [
        #     self._std_weight_position * mean[3],
        #     self._std_weight_position * mean[3],
        #     1e-2,
        #     self._std_weight_position * mean[3]]
        # std_vel = [
        #     self._std_weight_velocity * mean[3],
        #     self._std_weight_velocity * mean[3],
        #     1e-5,
        #     self._std_weight_velocity * mean[3]]

        # 可以根据目标的速度或加速度来动态调整过程噪声协方差矩阵。
        # 例如，速度越快或加速度越大，过程噪声可能需要增大，以反映更高的不确定性
        std_pos = [
            self._std_weight_position * mean[3] * position_noise_factor,
            self._std_weight_position * mean[3] * position_noise_factor,
            1e-2,
            self._std_weight_position * mean[3] * position_noise_factor,
        ]
        std_vel = [
            self._std_weight_velocity * mean[3] * velocity_noise_factor,
            self._std_weight_velocity * mean[3] * velocity_noise_factor,
            1e-5,
            self._std_weight_velocity * mean[3] * velocity_noise_factor,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))  # 过程噪声协方差矩阵

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        """
        在 project 方法中，使用位置的标准差权重来计算测量噪声协方差矩阵。
        类似地，可以通过调整 _std_weight_position 参数来控制测量噪声的大小。
        """
        # Adjust measurement noise dynamically based on some condition
        measurement_noise_factor = self.get_measurement_noise_factor(mean)

        # std = [
        #     self._std_weight_position * mean[3],
        #     self._std_weight_position * mean[3],
        #     1e-1,
        #     self._std_weight_position * mean[3]]

        # 测量噪声协方差矩阵可以根据测量的不确定性或环境变化来动态调整。
        # 例如，如果检测器的置信度较低，可以增加测量噪声，以反映较大的测量不确定性。
        std = [
            self._std_weight_position * mean[3] * measurement_noise_factor,
            self._std_weight_position * mean[3] * measurement_noise_factor,
            1e-1,
            self._std_weight_position * mean[3] * measurement_noise_factor,
        ]
        innovation_cov = np.diag(np.square(std))  # 测量噪声协方差矩阵

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def get_measurement_noise_factor(self, mean):
        # Example: Adjust based on some condition
        # For instance, if the target height (mean[3]) is large, increase the noise
        if mean[3] > 100:
            return 1.5
        else:
            return 1.0

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        # new_covariance = covariance - np.linalg.multi_dot((
        #     kalman_gain, projected_cov, kalman_gain.T))
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, self._update_mat, covariance)
        )
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        """马氏距离考虑了数据的协方差结构，是一个标准化的距离度量，可以用于判断一个样本是否属于某个分布。相比于欧氏距离，马氏距离在处理具有相关性的变量时更为有效。"""
        # print("cholesky_distance:")
        # print(f"measurements: {measurements}")
        # print(f"track: {mean, covariance}")
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        squared_maha = np.sum(z * z, axis=0)
        # print(f"squared_maha: {squared_maha}")
        return squared_maha  # 数值越小 数据点与均值的距离越近 数据点更可能是正常数据，因为它们靠近数据分布的中心
