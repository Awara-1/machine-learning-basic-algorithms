#include "logisticregression.h"

// constructor
LogisticRegression::LogisticRegression(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> x,
                                       Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> y,
                                       Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> theta,
                                       float learning_rate,
                                       int num_iterations)
{
    SetLogisticRegression(x, y, theta, learning_rate, num_iterations);
}

void LogisticRegression::SetLogisticRegression(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> x,
                                               Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> y,
                                               Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> theta,
                                               float learning_rate,
                                               int num_iterations)
{
    x_ = x,
    y_ = y,
    theta_ = theta,
    learning_rate_ = learning_rate;
    num_iterations_ = num_iterations;

}

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> LogisticRegression::calculatePredicted() {

    // Testing function pointers and lambda - NOTE: NOT USED BUT KEEPING IN BECAUSE IT'S COOL!
//    func = &LogisticRegression::sigmoid_;
//    float output = (this->*func)(5);
//    cout << "OUTPUT IS:" << output << endl;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> z = x_*theta_.transpose();
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> y_hat = z.unaryExpr(std::ptr_fun(sigmoid));
    return y_hat;
}

float LogisticRegression::calculateLoss() {
    /* TODO: The cost function matrix calculation below is what will be calculated */
    // J = -1/(n)*sum(y.*log(sigmoid(x*w))+(1-y).*log(1-sigmoid(x*w);

    int nr_rows = x_.rows();
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> labels = y_.transpose(); // enusre correct oreintation
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> y_hat;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> y_hat_log;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> lhs;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> rhs;
    float loss;

    // LHS
    y_hat = calculatePredicted();
    y_hat_log = y_hat.unaryExpr(std::ptr_fun(natural_log));
    lhs = labels.cwiseProduct(y_hat_log);
//    cout << "LHS is: " << endl << lhs << endl;

    // RHS
    rhs = (labels.unaryExpr(std::ptr_fun(subtract_from_one))).cwiseProduct((y_hat.unaryExpr(std::ptr_fun(subtract_from_one))).unaryExpr(std::ptr_fun(natural_log)));
//    cout << "RHS is: " << endl << rhs << endl;

    loss = -1*(lhs + rhs).sum()/nr_rows;
//    cout << "loss is: " << loss << endl;

    return loss;
}

float LogisticRegression::calculateGradient(){

    float gradient = ((calculatePredicted() - y_.transpose()).transpose()*x_).sum();

    return gradient;
}

//Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> LogisticRegression::convertPredictedProbability(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> y_predicted){
//    return
//}


void LogisticRegression::updateWeights(){
// TODO: Figure out how to subtract the gradient for gradient descent to update the weights

    float gradient = calculateGradient();
    theta_ = theta_.array() - learning_rate_*gradient; // update weights thru gradient descent algorithm
}

void LogisticRegression::minimizeCostFunction(){

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> y_predicted_final;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> y_predicted_binary;

    for (int i = 0; i < num_iterations_; i++) {
        updateWeights();
        float cost = calculateLoss();
        cout << "EPOCH NUMBER: " << i << " COST LOSS IS: " << cost <<  endl;
    }

    y_predicted_final = calculatePredicted(); // final probability output from sigmoid function
    y_predicted_binary = y_predicted_final.unaryExpr(std::ptr_fun(return_binary)); // convert to binary

    cout << "FINAL THETA (WEIGHTS) IS: " << theta_ << endl;
    cout << "FINAL PREDICTED PROBABILITIES IS: " << endl;
    cout << y_predicted_final << endl;
    cout << "FINAL PREDICTED BINARY OUTPUT IS: " << endl;
    cout << y_predicted_binary << endl;

}

