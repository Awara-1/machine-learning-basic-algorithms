#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <iostream>
#include <Eigen/Dense>
#include <functional>

using namespace std;


class LogisticRegression
{

private:
    // NOTE:
    // n -> number of samples
    // m -> number of features
    // c -> number of classes

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> x_; // n x m
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> y_; // n x 1 for binary, n x c for multi-class (softmax than argmax to bring down to n x 1)
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> theta_; // m x 1 for binary, m x c for multiclass
    float learning_rate_;
    int num_iterations_;

    float (LogisticRegression::*func)(float); // class function pointer

    float sigmoid_ (float cell){
        return 1/(1+ std::exp(-cell));
    }


public:

    static float sigmoid (float cell){
        return 1/(1+ std::exp(-cell));
    }

    static float exponential(float cell){
        return std::exp(-cell);
    }

    static float natural_log(float cell){
        return std::log(cell);
    }

    static float subtract_from_one(float cell){
        return 1 - cell;
    }

    // convert predicted probability from calculatePredicted() to binary values based on <0.5, >0.5 of sigmoid function
    static float return_binary(float cell){
        if (cell > 0.5){
            return 1;
        } else if (cell < 0.5){
            return 0;
        }
    }

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> getTheta(){return theta_;}

    LogisticRegression(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> x,
                       Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> y,
                       Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> theta,
                       float learning_rate,
                       int num_iterations);

    void SetLogisticRegression(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> x,
                               Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> y,
                               Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> theta,
                               float learning_rate,
                               int num_iterations);

    // calculate predicted y probability i.e. h(theta), should be of size n x 1
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> calculatePredicted();

    float calculateLoss();

    float calculateGradient();

    void updateWeights();

    void minimizeCostFunction();

    // convert predicted probability from calculatePredicted() to binary values based on <0.5, >0.5 of sigmoid function
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> convertPredictedProbability (Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>
                                                                                      y_predicted);











};

#endif // LOGISTICREGRESSION_H
