#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <vector>
#include <iostream>

using namespace std;


class LinearRegression
{

private:
    vector<float> x_; // independant variable
    vector<float> y_; // output label
    float m_;
    float c_;
    float learning_rate_;
    int num_iterations_;

public:
    LinearRegression(vector<float> x, vector<float> y, float m, float c, float learning_rate, int num_iterations);

    void SetLinearRegression(vector<float> x, vector<float> y, float m, float c, float learning_rate, int num_iterations);

    float getGradient(){return m_;}
    float getBias(){return c_;}

    float calculateLossFunction(vector<float> y_predicted);

    vector<float> calculatePredicted();

    float calculatePartialDerivativeM();

    float calculatePartialDerivativeC();

    void updateM();

    void updateC();

    void MinimizeLossFunction();







};

#endif // LINEARREGRESSION_H
