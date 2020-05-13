#include <iostream>
#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.0;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  // no measurement received yet
	is_initialized_ = false;
	// dimension of initial and augmented state 
	n_x_ = 5;
	n_aug_ = 7;
	// lambda factor
	lambda_ = 3-n_x_;
	// sigma points
	n_sig_ = 2*n_aug_+1;
	Xsig_pred_ = MatrixXd(n_x_, n_sig_);
	// weights
	weights_ = VectorXd(n_sig_);
	// state uncertainty covariance matrix
	P_ << 1, 0, 0, 0, 0,
			0, 1, 0, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 0, std_laspx_*std_laspy_, 0, 
			0, 0, 0, 0, std_laspx_*std_laspy_;
	// last measurement
	previous_timestamp_ = 0;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	long long timestamp = meas_package.timestamp_;
	if (!is_initialized_) {
		// first measurement is directly assumed to be the initial state
		previous_timestamp_ = timestamp;
		// make sure you switch between lidar and radar measurements.
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			// values of measurement space
			float rho = meas_package.raw_measurements_(0);
			float phi = meas_package.raw_measurements_(1);
			float rhod = meas_package.raw_measurements_(2);
			// convert to state space
			float px = rho*cos(phi);
			float py = rho*sin(phi);
			float v = rhod;
			x_ << px, py, v, 0, 0;
		} else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			// state space == measurement space
			x_ << meas_package.raw_measurements_(0),
			      meas_package.raw_measurements_(1),
			      0,
			      0,
			      0;
		}
		is_initialized_ = true;
	} else if ((meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) || 
				(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)) {
		// subsequent measurements are more complicated
		double delta_t = (timestamp - previous_timestamp_) / 1000000.;
		previous_timestamp_ = timestamp;
		// predict where we should be right now
		Prediction(delta_t);
		// update with the new measurement package
		// make sure you switch between lidar and radar measurements.
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			UpdateRadar(meas_package);
		} else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			UpdateLidar(meas_package);
		}
	}
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /*
	  Estimates the object's location. Modifies the state
	  vector (x_), predicts sigma points, the state, and the state covariance matrix.
	*/
	// 1. generate sigma points
	// (see lesson: "Augmentation Assignment 1")
	// augmented state including noise 
	VectorXd x_aug = VectorXd(n_aug_);
	// augmented covariance matrix
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
	// matrix of sigma points 
	MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
	// augmented state centers around current state (best estimate)
	x_aug.fill(0);
	x_aug.head(5) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;
	// process noise covariance matrix
	MatrixXd Q = MatrixXd(2, 2);
	Q << pow(std_a_*std_a_,2), 0,
		 0, pow(std_yawdd_*std_yawdd_,2); 
	// augmented covariance matrix
	P_aug.fill(0);
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug.bottomRightCorner(2, 2) = Q;
	// square root matrix
	MatrixXd A = P_aug.llt().matrixL();
	// create augmented sigma points
	Xsig_aug.col(0) = x_aug;
	for (int i = 0; i < n_aug_; i++) {
		Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_+n_aug_) * A.col(i);
		Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_+n_aug_) * A.col(i);
	}
	// 2. predict sigma points & weights
	// (see lesson: "Sigma Point Prediction Assignment 1")
	for (int i = 0; i < n_sig_; i++) {
		// current sigma point values
		double p_x = Xsig_aug(0, i);
		double p_y = Xsig_aug(1, i);
		double v = Xsig_aug(2, i);
		double yaw = Xsig_aug(3, i);
		double yawd = Xsig_aug(4, i);
		double nu_a = Xsig_aug(5, i);
		double nu_yawdd = Xsig_aug(6, i);
		// predicted state values
		double p_x_pred, p_y_pred;
		// motion process model:
		// ... perform movement, avoid division by zero if no change of yaw angle
		if (fabs(yawd > 0.001)) {
			p_x_pred = p_x + v / yawd * (sin(yaw+yawd*delta_t) - sin(yaw));
			p_y_pred = p_y + v / yawd * (cos(yaw) - cos(yaw+yawd*delta_t));
		} else {
			p_x_pred = p_x + v*delta_t*cos(yaw);
			p_y_pred = p_y + v*delta_t*sin(yaw);
		}
		// ... linear velocity stays constant, yaw angle changes by angular 
		//     velocity, angular velocity stays constant
		double v_pred = v;
		double yaw_pred = yaw + yawd*delta_t;
		double yawd_pred = yawd;
		// add process noise to posterior values
		p_x_pred += 0.5 * nu_a * delta_t*delta_t * cos(yaw);
		p_y_pred += 0.5 * nu_a * delta_t*delta_t * sin(yaw);
		v_pred += nu_a * delta_t;
		yaw_pred += 0.5 * nu_yawdd * delta_t*delta_t;
		yawd_pred += nu_yawdd * delta_t;
		// store predicted sigma point
		Xsig_pred_(0, i) = p_x_pred;
		Xsig_pred_(1, i) = p_y_pred;
		Xsig_pred_(2, i) = v_pred;
		Xsig_pred_(3, i) = yaw_pred;
		Xsig_pred_(4, i) = yawd_pred;
		// set weight
		if (i == 0) {
			weights_(i) = lambda_ / (lambda_+n_aug_);
		} else {
			weights_(i) = 0.5 / (lambda_+n_aug_);
		}
	}
	// 3. predict new state (mean)
	x_.fill(0.0);
	for (int i = 0; i < n_sig_; i++) {
		x_ += weights_(i) * Xsig_pred_.col(i);
	}
	// 4. predict new state covariance
	P_.fill(0.0);
	for (int i = 0; i < n_sig_; i++) {
		// state difference
		VectorXd delta_x = Xsig_pred_.col(i) - x_;
		// normalization of yaw angle after prediction
		while (delta_x(3)> M_PI) delta_x(3) -= 2.*M_PI;
		while (delta_x(3)<-M_PI) delta_x(3) += 2.*M_PI;
		P_ += weights_(i) * delta_x * delta_x.transpose();
	}
}


/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /*
	  Uses lidar data to update the belief about the object's
	  position. Modify the state vector, x_, and covariance, P_.
	  Also calculates the lidar NIS.
	*/
	// 1. measurement dimension & values
	int n_z = 2;
	MatrixXd z = meas_package.raw_measurements_;
	// 2. measurement function
	MatrixXd H = MatrixXd(n_z, n_x_);
	H << 1, 0, 0, 0, 0,
		 0, 1, 0, 0, 0;
	MatrixXd Ht = H.transpose();
	// 3. measurement prediction
	VectorXd z_pred = H * x_;
	// 4. measurement noise covariance for lidar
	MatrixXd R = MatrixXd(n_z, n_z);
	R << std_laspx_*std_laspx_, 0,
		0, std_laspy_*std_laspy_;
	// 5. measurement covariance matrix (with noise R)
	MatrixXd S = H * P_ * Ht + R;
	// 6. Kalman gain
	MatrixXd K = P_ * Ht * S.inverse();
	// 7. error calculation
	VectorXd y = z - z_pred;
	while (y(1)> M_PI) y(1) -= 2.*M_PI;
	while (y(1)<-M_PI) y(1) += 2.*M_PI;
	// 8. update mean and covariance
	x_ = x_ + K * y;
	MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
	P_ = (I - K * H) * P_;
	// 9. calculate Radar NIS
	float eta = y.transpose() * S.inverse() * y;
	// 10. print the output
	//std::cout << "NIS for Lidar:\t" << eta << std::endl;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
	/**
	  Uses radar data to update the belief about the object's
	  position. Modifies the state vector, x_, and covariance, P_.
	  Also needs to calculate the radar NIS.
	*/
	// (see lesson "Predict Radar Measurement Assignment 1")
	// 1. measurement dimension & values
	int n_z = 3;
	MatrixXd z = meas_package.raw_measurements_;
	// 2. generate sigma points & 3. measurement prediction
	MatrixXd Zsig = MatrixXd(n_z, n_sig_);
	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0.0);
	for (int i = 0; i < n_sig_; i++) {
		// re-use old sigma points and transform into measurement space
		VectorXd xsig_pred = Xsig_pred_.col(i);
		double p_x = xsig_pred(0);
		double p_y = xsig_pred(1);
		double v = xsig_pred(2);
		double yaw = xsig_pred(3);
		double v1 = cos(yaw)*v;
		double v2 = sin(yaw)*v;
		// avoid division by zero
		double epsilon = 0.00001;
		if (fabs(p_x) < epsilon) {
			p_x = epsilon;
		}
		if (fabs(p_y) < epsilon) {
			p_y = epsilon;
		}
		// transform from state space to measurement space
		double rho = sqrt((p_x*p_x) + (p_y*p_y));
		double phi = atan2(p_y, p_x);
		double rhod = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);
		Zsig.col(i) << rho, phi, rhod;
		z_pred += weights_(i) * Zsig.col(i);
	}
	// 4. measurement noise covariance for radar 
	MatrixXd R = MatrixXd(3, 3);
	R << std_radr_*std_radr_, 0, 0,
		0, std_radphi_*std_radphi_, 0,
		0, 0, std_radrd_*std_radrd_;
	// 5. measurement covariance matrix (with noise R)
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd delta_z = Zsig.col(i) - z_pred;
		while (delta_z(1)> M_PI) delta_z(1) -= 2.*M_PI;
		while (delta_z(1)<-M_PI) delta_z(1) += 2.*M_PI;
		S += weights_(i) * delta_z * delta_z.transpose();
	}
	S += R;
	// (see lesson "UKF Update Assignment 1")
	// 6. Kalman gain
	// cross correlation matrix
	MatrixXd Tc = MatrixXd(n_x_, n_z);
	Tc.fill(0.0);
	for (int i = 0; i < n_sig_; i++) {
		MatrixXd delta_x = Xsig_pred_.col(i) - x_;
		while (delta_x(3)> M_PI) delta_x(3) -= 2.*M_PI;
		while (delta_x(3)<-M_PI) delta_x(3) += 2.*M_PI;
		MatrixXd delta_z = Zsig.col(i) - z_pred;
		while (delta_z(1)> M_PI) delta_z(1) -= 2.*M_PI;
		while (delta_z(1)<-M_PI) delta_z(1) += 2.*M_PI;
		Tc += weights_(i) * delta_x * delta_z.transpose();
	}
	// Kalman gain
	MatrixXd K = Tc * S.inverse();
	// 7. error calculation
	VectorXd y = z - z_pred;
	while (y(1)> M_PI) y(1) -= 2.*M_PI;
	while (y(1)<-M_PI) y(1) += 2.*M_PI;
	// 8. update mean and covariance
	x_ = x_ + K * y;
	P_ = P_ - K * S * K.transpose();
	// 9. calculate Radar NIS
	float eta = y.transpose() * S.inverse() * y;
	// 10. print the output
	//std::cout << "NIS for Radar:\t" << eta << std::endl;
}