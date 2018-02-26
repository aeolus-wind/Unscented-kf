#include "gmock/gmock.h"  

#include "tools.h"



using Eigen::VectorXd;

using Eigen::MatrixXd;

using std::vector;

using std::is_scalar;



using namespace::testing;



class InputVectorsForRMSETests : public Test {

public:

	vector<VectorXd> estimations;

	vector<VectorXd> ground_truth;

	Tools tool;



	void SetUp() {

		//the input list of estimations

		VectorXd e(4);

		e << 1, 1, 0.2, 0.1;

		estimations.push_back(e);

		e << 2, 2, 0.3, 0.2;

		estimations.push_back(e);

		e << 3, 3, 0.4, 0.3;

		estimations.push_back(e);



		//the corresponding list of ground truth values

		VectorXd g(4);

		g << 1.1, 1.1, 0.3, 0.2;

		ground_truth.push_back(g);

		g << 2.1, 2.1, 0.4, 0.3;

		ground_truth.push_back(g);

		g << 3.1, 3.1, 0.5, 0.4;

		ground_truth.push_back(g);

	}

};



TEST_F(InputVectorsForRMSETests, RMSEFunctionPassesBasicExample) {

	VectorXd expected(4);

	expected << 0.1, 0.1, 0.1, 0.1;

	Tools t;

	cout << t.CalculateRMSE(estimations, ground_truth).isApprox(expected, 1e-6);
	ASSERT_TRUE(t.CalculateRMSE(estimations, ground_truth).isApprox(expected, 1e-6));


}