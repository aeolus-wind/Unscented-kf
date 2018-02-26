#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
	VectorXd RMSE;

	if (estimations.size() != ground_truth.size()) {
		cout << "estimations and ground truth vectors not same size" << endl;
		return RMSE;
	}
	RMSE = VectorXd::Zero(estimations.at(0).size());
	VectorXd res;
	for (int i = 0; i < estimations.size(); i++) {
		res = estimations.at(i) - ground_truth.at(i);
		res = res.array()*res.array();
		RMSE += res;
	}

	RMSE = RMSE / estimations.size();
	RMSE = RMSE.array().sqrt();
	return RMSE;
}