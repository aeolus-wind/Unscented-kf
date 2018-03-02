#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
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

  Xsig_pred_ = MatrixXd(5, 15);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/13.; //this is huge...
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
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
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off... problably std_yawdd
  */
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3.0 - n_aug_;

  is_initialized_ = false;

  w_0 = (lambda_)/ (lambda_ + n_aug_);
  w_other = 0.5/(lambda_ + n_aug_);
  

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
	cout << "Process measurement called" << endl;
	if (meas_package.timestamp_ <= 0.01)
		return;
	
	if (!is_initialized_) {
		
		
		if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			x_(0) = meas_package.raw_measurements_(0);
			x_(1) = meas_package.raw_measurements_(1);
			x_(2) = 0; //lidar has no acceleration and velocity
			x_(3) = 0;
			x_(4) = 0;
			
			previous_timestamp = meas_package.timestamp_;
		}
		else if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			//x_ = PolarToCTRV(meas_package.raw_measurements_);
			double rho = meas_package.raw_measurements_(0);
			double phi = meas_package.raw_measurements_(1);
			double rho_dot = meas_package.raw_measurements_(2);
			x_ << rho*cos(phi), rho * sin(phi), 0, 0, 0;

			previous_timestamp = meas_package.timestamp_;
		}
		
		/*
		
		TODO: FIGURE OUT THE PROPER INITIALIZATION FOR P...

		*/
		/*
		P_ << 0.00548207, -0.00249815, 0.00340508, -0.00357408, -0.0030908,
		-0.00249815, 0.0110547, 0.00151778, 0.00990746, 0.00806631,
		0.00340508, 0.00151778, 0.0058, 0.00078, 0.0008,
		-0.00357408, 0.00990746, 0.00078, 0.011924, 0.01125,
		-0.0030908, 0.00806631, 0.0008, 0.01125, 0.0127;*/
		
		P_ << 1, 0, 0.05, 0, 0,
			0, 1, 0.05, 0, 0,
			0, 0, 1, 0.01, 0,
			0, 0, 0, 1, 0.01,
			0, 0, 0, 0, 1;
		
		/*
		P_ << 1, 0, 0, 0, 0,
			0, 1, 0, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 0, 1, 0,
			0, 0, 0, 0, 1;
*/
		
		is_initialized_ = true;
		
	} else {	
		double delta_t = (meas_package.timestamp_ - previous_timestamp) / 1000000.0;
			Prediction(delta_t);
			previous_timestamp = meas_package.timestamp_;
			cout << "x_ is " << x_<<" after prediction "<< endl;
		if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_ ) {
			cout << "laser update" << endl;
			UpdateLidar(meas_package);
		}
		else if(meas_package.sensor_type_== MeasurementPackage::RADAR && use_radar_) {
			cout << "radar update" << endl;
			UpdateRadar(meas_package);

			cout << "P_ is " << P_ << " after radar update" << endl;

			cout << "x_ is " << x_ << " after radara update" << endl;
		}
	}
	
}

VectorXd UKF::CTRVToPolar(const VectorXd& CTRV_Vector){
    
    
    VectorXd PolarVector(3);
    double px = CTRV_Vector(0);
    double py = CTRV_Vector(1);
    double nu = CTRV_Vector(2);
    double rho = CTRV_Vector(3);
    double rho_dot = CTRV_Vector(4);
    
    double r = pow(px*px + py*py, 0.5);

	if (isEqual(r, 0.0) || isEqual(px,0)) {
		cout << "division by zero in CTRV to Polar" << endl;
		return PolarVector;
	}
    PolarVector(0) = r;
    PolarVector(1) = atan2(py, px);
    PolarVector(2) = (px* cos(rho) + py*sin(rho))*nu/r;
    
    return PolarVector;
}

VectorXd UKF::PolarToCTRV(const VectorXd &PolarVector) {

	VectorXd CTRVVector(5);
	
	double rho = PolarVector(0);

	double theta = PolarVector(1);

	double rho_dot = PolarVector(2);

	CTRVVector << rho*cos(theta) , rho*sin(theta) , rho_dot , theta, 0;
	
	return CTRVVector;

}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
	
	MatrixXd Q(2, 2);
	Q << std_a_ * std_a_, 0,
	  0, std_yawdd_ * std_yawdd_;

	MatrixXd P_aug = GenerateAugmentedMatrix(P_, Q);
	MatrixXd Xsig = GenerateSigmaPoints(P_aug);
	
	
	SigmaPointPrediction(Xsig, delta_t);
	cout << "XSigPred_ is " << Xsig_pred_ << endl;


	x_ = Xsig_pred_.col(0) * w_0;

	
	
	for(int i = 0; i<n_aug_*2; i++){
		x_ += Xsig_pred_.col(i+1)*w_other;
	}
	VectorXd res = Xsig_pred_.col(0) - x_;
	P_ = res* res.transpose() * w_other;

	
  
	for(int i = 0; i<n_aug_*2;i++){
		VectorXd res = Xsig_pred_.col(i + 1) - x_;
		P_ += res* res.transpose() * w_other;
	}
	
	
}

VectorXd UKF::CTRVToLaserMeasurement(const VectorXd& CTRV_Vector) {

	return CTRV_Vector.head(2);
}

void UKF::PredictLidar(MatrixXd& Zsig, VectorXd& z_pred, const MatrixXd& R, MatrixXd& S) {

	for(int i = 0; i<2*n_aug_ + 1; i++) {
      Zsig.col(i) = CTRVToLaserMeasurement(Xsig_pred_.col(i));
  }

	z_pred = w_0 * Zsig.col(0);
	for (int i = 0; i < 2 * n_aug_; i++) {
		z_pred += w_other* Zsig.col(i + 1);
	}
	VectorXd res = Zsig.col(0) - z_pred;
	S = w_0*res*res.transpose();

	for (int i = 0; i < 2 * n_aug_; i++) {
		res = Zsig.col(i + 1) - z_pred;
		S += w_other * res* res.transpose();
	}
	S += R;
}

void UKF::UpdateLidarCross(const MeasurementPackage meas_package, const MatrixXd&  Zsig, 
						   const VectorXd& z_pred, const MatrixXd& S, MatrixXd& P,VectorXd& x_pred) {
	MatrixXd T(5, 2);
	T = w_0*(Xsig_pred_.col(0) - x_pred) * (Zsig.col(0) - z_pred).transpose();
	for (int i = 0; i < 2 * n_aug_; i++) {
		T += w_other*(Xsig_pred_.col(i+1) - x_pred) * (Zsig.col(i+1) - z_pred).transpose();
	}
	MatrixXd Si = S.inverse();
	MatrixXd K = T*Si;
	VectorXd res = meas_package.raw_measurements_ - z_pred;
	x_pred = x_pred + K*res;

	P = P - K*S*K.transpose();

	
	NIS_Lidar = res.transpose() * Si * res;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  VectorXd z_pred;
  MatrixXd Zsig;
  MatrixXd S;
  MatrixXd R;
  int n_z = 2;

  z_pred.resize(n_z);

  Zsig.resize(n_z, n_aug_ * 2 + 1);

  S = MatrixXd(n_z,n_z);
  R = MatrixXd::Zero(n_z, n_z);
  R(0, 0) = std_laspx_*std_laspx_ ;
  R(1, 1) = std_laspy_*std_laspy_;

  PredictLidar(Zsig, z_pred, R, S);

  UpdateLidarCross(meas_package, Zsig, z_pred, S, P_,  x_);
	
  	
}


double NormalizeBetweePiMinusPi(double rad) {
	while (rad > M_PI) rad -= 2.*M_PI;
	while (rad < -M_PI) rad += 2.*M_PI;
	return rad;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */

void UKF::PredictRadar(MatrixXd& Zsig, VectorXd& z_pred, const MatrixXd& R, MatrixXd& S) {

	//when do I include checks for initial conditions?
	for(int i = 0; i<2*n_aug_ + 1; i++) {
      Zsig.col(i) = CTRVToPolar(Xsig_pred_.col(i));
  }

	z_pred = w_0 * Zsig.col(0);
	
	for (int i = 0; i < 2 * n_aug_; i++) {
		z_pred += w_other* Zsig.col(i + 1);
	}

	cout << "z_pred is " << z_pred << " after update" << endl;
	VectorXd res = Zsig.col(0) - z_pred;
	res(1) = NormalizeBetweePiMinusPi(res(1));
	S = w_0*res*res.transpose();

	for (int i = 0; i < 2 * n_aug_; i++) {
		res = Zsig.col(i + 1) - z_pred;
		res(1) = NormalizeBetweePiMinusPi(res(1));
		S += w_other * res* res.transpose();
	}
	S += R;


}

void UKF::UpdateRadarCross(const MeasurementPackage meas_package, const MatrixXd&  Zsig, 
						   const VectorXd& z_pred, const MatrixXd& S, MatrixXd& P,VectorXd& x_pred) {
	MatrixXd T(5, 3);
	T = w_0*(Xsig_pred_.col(0) - x_pred) * (Zsig.col(0) - z_pred).transpose();
	for (int i = 0; i < 2 * n_aug_; i++) {
		T += w_other*(Xsig_pred_.col(i+1) - x_pred) * (Zsig.col(i+1) - z_pred).transpose();
	}


	MatrixXd Si = S.inverse();
	MatrixXd K = T*Si;
	VectorXd res = meas_package.raw_measurements_ - z_pred;
	x_pred = x_pred + K*res;

	cout << "x_pred is " << x_pred << endl;
	P = P - K*S*K.transpose();

	
	NIS_Radar = res.transpose() * Si * res;

	
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  VectorXd z_pred;
  MatrixXd Zsig;
  MatrixXd S;
  MatrixXd R;
  int n_z = 3;

  z_pred.resize(n_z);

  Zsig.resize(n_z, n_aug_ * 2 + 1);

  S = MatrixXd(n_z,n_z);
  R = MatrixXd::Zero(n_z, n_z);
  R(0, 0) = std_radr_*std_radr_;
  R(1, 1) = std_radphi_*std_radphi_;
  R(2, 2) = std_radrd_*std_radrd_;


	PredictRadar(Zsig, z_pred, R, S);

	UpdateRadarCross(meas_package, Zsig, z_pred, S, P_, x_);
	
	/*
	
	TODO: Calculate Radar NIS

	*/



}


MatrixXd UKF::GenerateSigmaPoints(MatrixXd P) {
	
	VectorXd x_aug(n_aug_); 
	x_aug << x_, 0, 0;
	
	MatrixXd ch = P.llt().matrixL();

	cout << "ch is " << ch << endl;
	MatrixXd Xsigma(7, 15);
	
	Xsigma.col(0) = x_aug;
	
	double lambda_scalar = pow(lambda_ + n_aug_, 0.5);


	for (int i = 0; i < n_aug_; i++) {
		Xsigma.col(i + 1) = x_aug + lambda_scalar * ch.col(i);
		Xsigma.col(i + n_aug_ + 1) = x_aug - lambda_scalar * ch.col(i);

		
	}
	cout << "Xsigma is " << Xsigma << endl;
	return Xsigma;
	
}

MatrixXd UKF::GenerateAugmentedMatrix(MatrixXd P, MatrixXd Q) {
	
	MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
	if (P.rows() + Q.rows() != n_aug_ || P.cols() + Q.cols() != n_aug_) {
		cout << "augmented matrix inputs incorrect dimensions" << endl;
		return P_aug;
	}
	
	
	P_aug.topLeftCorner(P.rows(), P.cols()) = P;
	P_aug.bottomRightCorner(Q.rows(), Q.cols()) = Q;
	return P_aug;
}


void UKF::SigmaPointPrediction(const MatrixXd& Xsig, double delta_t) {


	//predict sigma points
	//avoid division by zero
	//write predicted sigma points into right column
	
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {

		VectorXd x = Xsig.col(i);
		
		double px(x(0));
		double py(x(1));
		double v(x(2));
		double yaw(x(3));
		double yawdot(x(4));

		double nu_a(x(5));
		double nu_yaw(x(6));

		VectorXd nu(5);
		nu << 0.5*delta_t*delta_t*cos(yaw)*nu_a,
			0.5*delta_t*delta_t*sin(yaw)*nu_a,
			delta_t*nu_a,
			0.5*delta_t*delta_t*nu_yaw,
			delta_t*nu_yaw;
		

		
		if (isEqual(yawdot, 0.0) ) {
			cout << "yawdot was zero" << endl;
			x(0) = px + v*cos(yaw)*delta_t;
			x(1) = py + v*sin(yaw)*delta_t;
			
		}
		else {
				
			x(0) = px + v / yawdot*(sin(yaw + yawdot*delta_t) - sin(yaw));
			x(1) = py + v / yawdot*(-cos(yaw + yawdot*delta_t) + cos(yaw));
			x(3) = yaw + yawdot*delta_t;
			
		}
		
		Xsig_pred_.col(i) = x.head(5) + nu;
	}
	
}




bool isEqual(double x, double y) {
    double epsilon = 1e-6;
    return std::abs(x-y) <= std::abs(x)*epsilon;

}
