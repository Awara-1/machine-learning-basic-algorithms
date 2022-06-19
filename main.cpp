#include <iostream>
#include <stats.h>
#include <linearregression.h>
#include <logisticregression.h>

#include <Eigen/Dense>
#include <math.h>


using namespace std;

static float sigmoid_test(float cell){
    return 1/(1+ std::exp(-cell));
}

static float exponential(float cell){
    return std::exp(-cell);
}

static float natural_log(float cell){
    return std::log(-cell);
}


int main()
{
    cout << "Hello basic statistics playground!" << endl;

    // mean
    float test_mean;
    vector<float> v = {1, 5.0, 5.3, 9.5, 0.9, 13};

    test_mean = mean(5, 400);
    test_mean = mean_vector(v);
    cout << "mean is: " << test_mean << endl;

    // std
    float test_std = std_vector(v);
    cout << "std is: " << test_std << endl;

    // linear interpolation
    vector<Coordinates> coodinates_example {
        {1, 4},
        {2, 7},
        {3, 20},
        {6, 35},
        {10, 53}
    };

    vector<Coordinates> c = linear_interpolation(coodinates_example, {5, 7, 1.5, 200});
    cout << "linear interpolated coordinates are:" << endl;
    print_vector_struct(c);

    // covariance
    vector<float> x = {1, 5.0, 5.3, 9.5, 0.9, 13};
    vector <float> y = {2, 4, 5, 6, 7, 10};
    cout << "covariance is: " << covariance(x, y) << endl;

    // correlation
    cout << "correlation is: " << correlation(x, y) << endl;

    // linear regression
    x = {1, 2, 3, 4}; // independant variable
    y = {2, 4, 6, 8}; // output label
    float m = 0;
    float c_bias = 0;
    float learning_rate = 0.0001;
    int num_iterations = 10000;
    LinearRegression linear_regression_obj(x, y, m, c_bias, learning_rate, num_iterations);

    vector<float> y_predicted = linear_regression_obj.calculatePredicted();
    print_vector(y_predicted);

    float loss_function = linear_regression_obj.calculateLossFunction(y_predicted);
    cout << "Loss function is: " << loss_function << endl;

    float d_c = linear_regression_obj.calculatePartialDerivativeC();
    cout << "Parital Derivative w.r.t c: " << d_c << endl;

    float d_m = linear_regression_obj.calculatePartialDerivativeM();
    cout << "Parital Derivative w.r.t m: " << d_m << endl;

    cout << "Implementing Linear Regression Alogorithm:" << endl;
    linear_regression_obj.MinimizeLossFunction();

    // Experiment with EIGEN vectors
    Eigen::MatrixXf a(10,15);
    cout << "The matrix m is of size "
              << a.rows() << "x" << a.cols() << std::endl;

    int RowSize = 3; int ColSize = 5;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> test;

    test.resize(RowSize, ColSize);

    cout << "The matrix test is of size "
              << test.rows() << "x" << test.cols() << std::endl;


    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> features {     // construct a 2x2 matrix
          {1, 2, 7},    // first row
          {3, 4, 9},    // second row
          {4, 6, 19},   // third row
          {6, 10, 29},  // fourth row
          {8, 14, 45}   // fifth row
    };

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> theta { // some random weight initialization
          {0.4, 0.03, -0.4},
    };

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> labels {
          {1, 0, 1, 1, 0},
    };

    cout << "FEATURE matrix is: " << endl << features << endl;
    cout << "THETA matrix is: " << endl << theta << endl;
    cout << "LABEL matrix is: " << endl << labels << endl;

    cout << "Transpose result is: " << endl << theta.transpose() << endl;

    Eigen::VectorXf result;
    result = features*theta.transpose();
    cout << "features*thetaT result is: " << endl << result << endl;

    result = result.unaryExpr(&sigmoid_test);
    cout << "sigmoid applied features*thetaT result is: " << endl << result << endl;

    // element wise product testing
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> y_hat {
          {0.7, 0, 1, 1.8, 0}
    };

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> product_result = labels.cwiseProduct(y_hat);

    cout << "Test element-wise multiplicaion" << endl << product_result << endl;

    /* implement logistic regression */
    // test predicted
    LogisticRegression logistic_regression_obj(features, labels, theta, learning_rate, num_iterations);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> labels_predicted = logistic_regression_obj.calculatePredicted();

    cout << "Predicted labels are: " << endl << labels_predicted << endl;

    // test loss function
    float loss = logistic_regression_obj.calculateLoss();
    cout << "total loss is: " << endl << loss << endl;

    // test gradient descent algo
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> test_sub;
    test_sub = labels_predicted - labels.transpose();
    cout << "Subtraction of predicted vs. actual is: " << endl << test_sub << endl;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> test_sub_mult_x = test_sub.transpose()*features;
    cout << "Multiplication with x is " << endl << test_sub_mult_x << endl;

    // test gradietn calculation
    float gradient = logistic_regression_obj.calculateGradient();
    cout << "Calculated gradient is: " << endl << gradient << endl;

    // test gradient descent algo from class
    logistic_regression_obj.updateWeights();
    cout << "Updated wieghts through gradient descent is: " << endl << logistic_regression_obj.getTheta() << endl;

    // minimize cost function for logistic regression
    logistic_regression_obj.minimizeCostFunction();

    return 0;
}
