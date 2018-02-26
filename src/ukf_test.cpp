#include "gmock/gmock.h"
#include "ukf.h"
#include <stdio.h>

using namespace testing;
using namespace std;



TEST(UnscentedKalmanFilterInitialization, OnInitializationUKFHasis_initialized_set_to_false) {
	UKF unitialized;
	ASSERT_THAT(unitialized.is_initialized_, Eq(false));
}

class UninitializedUKFWithLaserRadarSamples : public Test {
public:
	UKF ukf;
	
	MeasurementPackage ALaserMeasurement;
	MeasurementPackage ARadarMeasurement;
	VectorXd LaserVector;
	VectorXd RadarVector;

	void SetUp() override {

		ALaserMeasurement.sensor_type_ = MeasurementPackage::LASER;
		ARadarMeasurement.sensor_type_ = MeasurementPackage::RADAR;
		
		LaserVector.resize(5);
		LaserVector << 1, 2, 0, 0, 1;
		ALaserMeasurement.raw_measurements_ = LaserVector;

		RadarVector.resize(3);
		RadarVector << 1, 3, 0.5;
		ARadarMeasurement.raw_measurements_ = RadarVector;

	}
};

TEST_F(UninitializedUKFWithLaserRadarSamples, UninitializedUKFAcceptsLaserKeepsFirstTwoCoordinates) {
	ukf.ProcessMeasurement(ALaserMeasurement);


	ASSERT_THAT(ukf.x_.head(2), Eq(LaserVector.head(2)));
	ASSERT_THAT(ukf.x_.tail(3), Not(Eq(LaserVector.tail(3))));
}



TEST_F(UninitializedUKFWithLaserRadarSamples, OnRadarReadingDataTransformedToCTRV) {
	ukf.ProcessMeasurement(ARadarMeasurement);

	VectorXd Expected(5);
	Expected << -0.9899924966004454, 0.1411200080598672, 0.5, 3, 0;

	ASSERT_THAT(ukf.x_, Eq(Expected));
}

TEST_F(UninitializedUKFWithLaserRadarSamples, UKFInitializeProcessMatrixWithIdentity) {
	ukf.ProcessMeasurement(ARadarMeasurement);

	ASSERT_TRUE(ukf.P_.isApprox(MatrixXd::Identity(5, 5)));

}


class InitializedUKF : public UninitializedUKFWithLaserRadarSamples {
public:
	MatrixXd P;
	MatrixXd Q;
	

	void SetUp() override {
		ARadarMeasurement.sensor_type_ = MeasurementPackage::RADAR;

		ukf.x_ << 5.7441,
					   1.3800,
					   2.2049,
					   0.5015,
					   0.3528;

		P.resize(5, 5);

		P << 0.0043, -0.0013, 0.0030, -0.0022, -0.0020,
			-0.0013, 0.0077, 0.0011, 0.0071, 0.0060,
			0.0030, 0.0011, 0.0054, 0.0007, 0.0008,
			-0.0022, 0.0071, 0.0007, 0.0098, 0.0100,
			-0.0020, 0.0060, 0.0008, 0.0100, 0.0123;

		Q.resize(2, 2);
		Q << pow(0.2,2), 0,
			0, pow(0.2,2);

	}
};



TEST_F(UninitializedUKFWithLaserRadarSamples, GenerateSigmaPointsTrivialExamplePasses) {
	ukf.ProcessMeasurement(ALaserMeasurement);
	
	MatrixXd Expected(7, 15);
	VectorXd x_aug(7);
	x_aug << ukf.x_, 0, 0;
	int n_aug = 7;
	double lambda = 3 - n_aug;
	MatrixXd ch = MatrixXd::Identity(7, 7);

	Expected.col(0) = x_aug;
	for (int i = 0; i < n_aug; i++) {
		Expected.col(i + 1) = x_aug + pow(n_aug + lambda, 0.5)*ch.col(i);
		Expected.col(n_aug + i + 1) = x_aug - pow(n_aug + lambda, 0.5)*ch.col(i);

	}

	ASSERT_TRUE(ukf.GenerateSigmaPoints(MatrixXd::Identity(7, 7)).isApprox(Expected, 1e-5));
}

TEST_F(InitializedUKF, GenerateSigmaPointsNonTrivialExamplePasses) {

	ukf.P_ = ukf.GenerateAugmentedMatrix(P, Q);
		
	
	MatrixXd Expected(7, 15);
	Expected<<
  5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,  5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
    1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,  1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
  2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
  0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,  0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
  0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
       0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641,        0,
       0,        0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641;
	;
	
	ASSERT_TRUE(ukf.GenerateSigmaPoints(ukf.P_).isApprox(Expected, 1e-5));
	
}

TEST_F(InitializedUKF, GenerateAugmentedMatrixConcatenatesProcessAndQMatrix) {

	MatrixXd Expected(ukf.n_aug_, ukf.n_aug_);
	Expected = MatrixXd::Zero(ukf.n_aug_, ukf.n_aug_);
	Expected.topLeftCorner(P.rows(), P.cols()) = P;

	Expected.bottomRightCorner(Q.rows(), Q.cols()) = Q;
	
	ASSERT_THAT(ukf.GenerateAugmentedMatrix(P, Q), Eq(Expected));
}


TEST_F(InitializedUKF, PredictOnSigmaPoints) {
	 //create example sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(ukf.n_aug_, 2 * ukf.n_aug_ + 1);
     Xsig_aug <<
    5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
         0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
         0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;

	 MatrixXd Expected(5, 15);
	 Expected << 5.93553, 6.06251, 5.92217, 5.9415, 5.92361, 5.93516, 5.93705, 5.93553, 5.80832, 5.94481, 5.92935, 5.94553, 5.93589, 5.93401, 5.93553,
		 1.48939, 1.44673, 1.66484, 1.49719, 1.508, 1.49001, 1.49022, 1.48939, 1.5308, 1.31287, 1.48182, 1.46967, 1.48876, 1.48855, 1.48939,
		 2.2049, 2.28414, 2.24557, 2.29582, 2.2049, 2.2049, 2.23954, 2.2049, 2.12566, 2.16423, 2.11398, 2.2049, 2.2049, 2.17026, 2.2049,
		 0.53678, 0.473387, 0.678098, 0.554557, 0.643644, 0.543372, 0.53678, 0.538512, 0.600173, 0.395462, 0.519003, 0.429916, 0.530188, 0.53678, 0.535048,
		 0.3528, 0.299973, 0.462123, 0.376339, 0.48417, 0.418721, 0.3528, 0.387441, 0.405627, 0.243477, 0.329261, 0.22143, 0.286879, 0.3528, 0.318159;
	 
	  ukf.SigmaPointPrediction(Xsig_aug, 0.1);
	  
	 ASSERT_TRUE(ukf.Xsig_pred_.isApprox(Expected, 1e-06));
	
}

TEST_F(InitializedUKF, AddVectorsWorks) {

	VectorXd val1(2);
	VectorXd val2(2);
	val1 << 1, 2;
	val2 << 2, 4;


}

TEST_F(InitializedUKF, PredictionStepPassesNonTrivialExample) {
	
	ukf.P_ <<  0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
           0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;


	ukf.std_a_ = 0.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
	ukf.std_yawdd_ = 0.2;

	VectorXd x_expected(5);
	x_expected << 5.93446, 1.48886, 2.2049, 0.53678, 0.3528;
	
	
	MatrixXd P_expected(5,5);
	P_expected << 0.00548207, -0.00249815, 0.00340508, -0.00357408, -0.0030908,
		-0.00249815, 0.0110547, 0.00151778, 0.00990746, 0.00806631,
		0.00340508, 0.00151778, 0.0058, 0.00078, 0.0008,
		-0.00357408, 0.00990746, 0.00078, 0.011924, 0.01125,
		-0.0030908, 0.00806631, 0.0008, 0.01125, 0.0127;
	
	//ukf.Xsig_pred_ = Xsig_pred;
	double delta_t = 0.1;
	ukf.Prediction(delta_t);

	ASSERT_TRUE(ukf.x_.isApprox(x_expected, 1e-5));
	ASSERT_TRUE(ukf.P_.isApprox(P_expected, 1e-5));

}


TEST(PolarToCTRV, PolarToCTRVPassesTrivialExample) {
	UKF ukf;
	VectorXd APolarVector(3);

	double rho = 2;
	double theta = M_PI / 3;
	double rho_dot = 1;

	APolarVector << rho, theta, rho_dot;

	VectorXd ExpectedCTRVVector(5);

	ExpectedCTRVVector << cos(theta)*rho, sin(theta)*rho, rho_dot, theta, 0;

	ASSERT_EQ(ukf.PolarToCTRV(APolarVector), ExpectedCTRVVector);
}


TEST_F(InitializedUKF, UpdateUsesXsig_predFromLastCallOfPredictForSigmaPoints) {
	
	MatrixXd Expected(5, 15);
	VectorXd x_aug(7);
	x_aug << ukf.x_, 0, 0;
	int n_aug = 7;
	double lambda = 3 - n_aug;
	MatrixXd ch = MatrixXd::Identity(7, 7);

	Expected << 5.93553, 7.66758, 5.93553, 6.08591, 5.6054, 5.92503, 5.94312, 5.93553, 4.20348, 5.93553, 5.78515, 5.82133, 5.94413, 5.92794, 5.93553,
		1.48939, 1.48939, 3.22144, 1.57532, 1.55138, 1.50531, 1.49355, 1.48939, 1.48939, -0.242664, 1.40346, 1.17349, 1.47235, 1.48522, 1.48939,
		2.2049, 2.2049, 2.2049, 3.93695, 2.2049, 2.2049, 2.37811, 2.2049, 2.2049, 2.2049, 0.472849, 2.2049, 2.2049, 2.03169, 2.2049,
		0.53678, 0.53678, 0.53678, 0.53678, 2.26883, 0.709985, 0.53678, 0.54544, 0.53678, 0.53678, 0.53678, -1.19527, 0.363575, 0.53678, 0.52812,
		0.3528, 0.3528, 0.3528, 0.3528, 0.3528, 2.08485, 0.3528, 0.526005, 0.3528, 0.3528, 0.3528, 0.3528, -1.37925, 0.3528, 0.179595;
	double delta_t = 0.1;

	MatrixXd Xsig = ukf.GenerateSigmaPoints(MatrixXd::Identity(7, 7));

	ukf.SigmaPointPrediction(Xsig, delta_t);


	ASSERT_TRUE(ukf.Xsig_pred_.isApprox(Expected, 1e-6));
}

TEST_F(InitializedUKF, CalculateZsig_predPassesNonTrivialExample) {
	
	//So that the measurement noise matches 
	ukf.std_radr_ = 0.3;
	ukf.std_radphi_ = 0.0175;
	ukf.std_radrd_ = 0.1;
	ukf.Xsig_pred_ << 5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
           1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
         0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;
	MatrixXd R = MatrixXd::Zero(3,3);
	R(0,0) = ukf.std_radr_*ukf.std_radr_;
	R(1,1) = ukf.std_radphi_*ukf.std_radphi_;
	R(2,2) = ukf.std_radrd_*ukf.std_radrd_;


	int n_z = 3;

	MatrixXd Zsig;
	Zsig.resize(n_z, ukf.n_aug_ * 2 + 1);
	VectorXd z_pred(3);
	MatrixXd S(n_z, n_z);

	VectorXd z_expected(3);
	z_expected << 6.12155, 0.245993, 2.10313;
	MatrixXd S_expected(3, 3);
	S_expected << 0.0946171, -0.000139448, 0.00407016,
		-0.000139448, 0.000617548, -0.000770652,
		0.00407016, -0.000770652, 0.0180917;

	ukf.PredictRadar(Zsig, z_pred, R, S);


	ASSERT_TRUE(z_pred.isApprox(z_expected,1e-6));
	ASSERT_TRUE(S.isApprox(S_expected, 1e-6));

}


TEST_F(InitializedUKF, RadarUpdatePassesNonTrivialExample) {
	

	ukf.P_ << 0.0054342,  -0.002405,  0.0034157, -0.0034819, -0.00299378,
			 -0.002405,    0.01084,   0.001492,  0.0098018,  0.00791091,
			  0.0034157,   0.001492,  0.0058012, 0.00077863, 0.000792973,
			 -0.0034819,  0.0098018, 0.00077863,   0.011923,   0.0112491,
			 -0.0029937, 0.0079109, 0.00079297, 0.011249, 0.0126972;
	
	ukf.x_<<
     5.93637,
     1.49035,
     2.20528,
    0.536853,
    0.353577;
	
	ukf.Xsig_pred_ <<5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
           1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
         0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;
	
	int n_z = 3;
	
	MatrixXd Zsig = MatrixXd(n_z, 2 * ukf.n_aug_ + 1);
	Zsig <<
      6.1190,  6.2334,  6.1531,  6.1283,  6.1143,  6.1190,  6.1221,  6.1190,  6.0079,  6.0883,  6.1125,  6.1248,  6.1190,  6.1188,  6.12057,
     0.24428,  0.2337, 0.27316, 0.24616, 0.24846, 0.24428, 0.24530, 0.24428, 0.25700, 0.21692, 0.24433, 0.24193, 0.24428, 0.24515, 0.245239,
      2.1104,  2.2188,  2.0639,   2.187,  2.0341,  2.1061,  2.1450,  2.1092,  2.0016,   2.129,  2.0346,  2.1651,  2.1145,  2.0786,  2.11295;


	VectorXd z_pred = VectorXd(n_z);
	z_pred <<
      6.12155,
     0.245993,
      2.10313;

 
  
	MatrixXd S = MatrixXd(n_z, n_z);
	S <<
    0.0946171, -0.000139448,   0.00407016,
   -0.000139448,  0.000617548, -0.000770652,
    0.00407016, -0.000770652,    0.0180917;

	VectorXd z_measure = VectorXd(n_z);
	z_measure <<
      5.9214,   //rho in m
      0.2187,   //phi in rad
      2.0062;
	MeasurementPackage ARadarMeasurement;
	
	VectorXd RadarVector;

		ARadarMeasurement.sensor_type_ = MeasurementPackage::RADAR;
		

		RadarVector.resize(3);
		RadarVector << z_measure;
		ARadarMeasurement.raw_measurements_ = RadarVector;
		cout << "z_pred is " << z_pred << endl;
		cout << "z_measure is " << z_measure << endl;

		ukf.UpdateRadarCross(ARadarMeasurement, Zsig, z_pred, S, ukf.P_, ukf.x_);

		VectorXd x_expected(5);
		MatrixXd P_expected(5, 5);
		x_expected << 5.92276, 1.41823, 2.15593, 0.489274, 0.321338;
		P_expected << 0.00361579, - 0.000357881, 0.00208316, - 0.000937196, - 0.00071727,
			- 0.000357881, 0.00539867, 0.00156846, 0.00455342, 0.00358885,
			0.00208316, 0.00156846, 0.00410651, 0.00160333, 0.00171811,
			- 0.000937196, 0.00455342, 0.00160333, 0.00652634, 0.00669436,
			- 0.00071719, 0.00358884, 0.00171811, 0.00669426, 0.00881797;


		ASSERT_TRUE(ukf.x_.isApprox(x_expected, 1e-6));
		ASSERT_TRUE(ukf.P_.isApprox(P_expected, 1e-6));
}


TEST_F(InitializedUKF, PredictLidarPassesNonTrivialExample) {
	
	//So that the measurement noise matches 
	

	int n_z = 2;

	ukf.Xsig_pred_ << 5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
           1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
         0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;
	MatrixXd R = MatrixXd::Zero(n_z,n_z);
	R(0,0) = ukf.std_laspx_*ukf.std_laspx_;
	R(1,1) = ukf.std_laspy_*ukf.std_laspy_;



	MatrixXd Zsig;
	Zsig.resize(n_z, ukf.n_aug_ * 2 + 1);
	VectorXd z_pred(n_z);
	MatrixXd S(n_z, n_z);

	VectorXd z_expected(n_z);
	z_expected << 5.93637333, 1.49035;
	MatrixXd S_expected(n_z, n_z);
	S_expected << 0.02793425, -0.0024053,
		-0.0024053 ,  0.033345;

	ukf.PredictLidar(Zsig, z_pred, R, S);

	cout << "z is " << z_pred << endl;
	cout << "S is " << S << endl;
	cout << "R is " << R << endl;
	ASSERT_TRUE(z_pred.isApprox(z_expected,1e-6));
	ASSERT_TRUE(S.isApprox(S_expected, 1e-6));

}


TEST_F(InitializedUKF, UpdateLidarCrossPassesNonTrivialExample) {
	//So that the measurement noise matches 

	ukf.P_ << 0.0054342,  -0.002405,  0.0034157, -0.0034819, -0.00299378,
			 -0.002405,    0.01084,   0.001492,  0.0098018,  0.00791091,
			  0.0034157,   0.001492,  0.0058012, 0.00077863, 0.000792973,
			 -0.0034819,  0.0098018, 0.00077863,   0.011923,   0.0112491,
			 -0.0029937, 0.0079109, 0.00079297, 0.011249, 0.0126972;
	
	ukf.x_<<
     5.93637,
     1.49035,
     2.20528,
    0.536853,
    0.353577;
	int n_z = 2;



	ukf.Xsig_pred_ << 5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
           1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
         0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;
	MatrixXd R = MatrixXd::Zero(n_z,n_z);
	R(0,0) = ukf.std_laspx_*ukf.std_laspx_;
	R(1,1) = ukf.std_laspy_*ukf.std_laspy_;

	VectorXd x_pred(5); 
	x_pred<< 5.93637333, 1.49035, 2.20528333, 0.53685267, 0.3535765;

	MatrixXd Zsig;
	Zsig.resize(n_z, ukf.n_aug_ * 2 + 1);
	VectorXd z_pred(n_z);
	MatrixXd S(n_z, n_z);



	VectorXd z_measure = VectorXd(n_z);
	z_measure <<
		5.78035,   
		1.28471;
	

	ukf.PredictLidar(Zsig, z_pred, R, S);

	cout << "z is " << z_pred << endl;
	cout << "S is " << S << endl;
	cout << "R is " << R << endl;

	VectorXd LidarVector;
	MeasurementPackage ALaserMeasurement;
	ALaserMeasurement.sensor_type_ = MeasurementPackage::LASER;
		

	LidarVector.resize(2);
	LidarVector << z_measure;
	ALaserMeasurement.raw_measurements_ = LidarVector;

	ukf.UpdateLidarCross(ALaserMeasurement, Zsig, z_pred, S, ukf.P_, ukf.x_);
	
	VectorXd x_expected(5);
	x_expected << 5.91882972, 1.43345767, 2.17439569, 0.49348302, 0.31970312;

	MatrixXd P_expected(5, 5);
	P_expected << 0.00426377, -0.00131514, 0.00285565, -0.002249, -0.00196394,
		-0.00131514, 0.00721792, 0.00121264, 0.00645166, 0.00519633,
		0.00285565, 0.00121264, 0.00528722, 0.00069216, 0.00074648,
		-0.002249, 0.00645166, 0.00069216, 0.00876436, 0.00868146,
		-0.00196386, 0.00519632, 0.00074648, 0.00868136, 0.01060888;


	ASSERT_TRUE(ukf.x_.isApprox(x_expected, 1e-6));
	ASSERT_TRUE(ukf.P_.isApprox(P_expected, 1e-6));
	
}



int main(int argc, char** argv) {
	InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
}


