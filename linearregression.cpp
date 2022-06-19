#include "linearregression.h"
#include <cmath>


void print_vector_(vector<float> v){
    for (auto it : v){
        cout << it << endl;
    }
}

//  LinearRegression constructor
LinearRegression::LinearRegression(vector<float> x, vector<float> y, float m, float c, float learning_rate, int num_iterations)
{
    SetLinearRegression(x, y, m , c, learning_rate, num_iterations);
}

void LinearRegression::SetLinearRegression(vector<float> x, vector<float> y, float m, float c, float learning_rate, int num_iterations)
{
    x_ = x;
    y_ = y;
    m_ = m;
    c_ = c;
    learning_rate_ = learning_rate;
    num_iterations_ = num_iterations;
}

float LinearRegression::calculateLossFunction(vector<float> y_predicted){

//    vector<float> y_predicted = calculatePredicted();
    float sum_mean_squares = 0;

    for (size_t i = 0; i < y_predicted.size(); ++i){
        float mean_square = pow((y_[i] - y_predicted[i]), 2);
        sum_mean_squares += mean_square;
    }

    return sum_mean_squares / y_predicted.size();
}

vector<float> LinearRegression::calculatePredicted(){

    vector<float> y_predicted;
    for (auto it: x_){
        y_predicted.push_back(m_*it + c_);
    }
    return y_predicted;
}


float LinearRegression::calculatePartialDerivativeM(){
    vector<float> y_predicted = calculatePredicted();

    float x_y_difference_sum = 0;

    for (size_t i = 0; i < y_predicted.size(); ++i){
        float x_y_difference = x_[i]*(y_[i] - y_predicted[i]);
        x_y_difference_sum += x_y_difference;
    }
    return (-2 * x_y_difference_sum) / y_predicted.size();
}


float LinearRegression::calculatePartialDerivativeC(){

    vector<float> y_predicted = calculatePredicted();

    float y_difference_sum = 0;

    for (size_t i = 0; i < y_predicted.size(); ++i){
        float y_difference = y_[i] - y_predicted[i];
        y_difference_sum += y_difference;
    }
    return (-2 * y_difference_sum) / y_predicted.size();
}

void LinearRegression::updateM(){
    m_ = m_ - learning_rate_ * calculatePartialDerivativeM();
}

void LinearRegression::updateC(){
    c_ = c_ - learning_rate_ * calculatePartialDerivativeC();
}

void LinearRegression::MinimizeLossFunction(){

    for (int i = 0; i < num_iterations_; i++) {
      vector<float> y_predicted = calculatePredicted();
      float mse_loss = calculateLossFunction(y_predicted);
      updateM();
      updateC();
      cout << "EPOCH NUMBER: " << i << " MSE LOSS IS: " << mse_loss <<  endl;
    }
    cout << "FINAL GRADIENT IS: " << m_ << endl;
    cout << "FINAL BIAS IS: " << c_ << endl;
    cout << "FINAL PREDICTED OUTCOME IS: " << endl;
    vector<float> y_predicted_final = calculatePredicted();
    print_vector_(y_predicted_final);
}


