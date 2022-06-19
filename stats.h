#ifndef STATS_H
#define STATS_H

#endif // STATS_H

#include <vector>
#include <iostream>
#include <functional>
#include <numeric>
#include <algorithm>
#include <cmath>

using namespace std;

float mean(float a, float b){

    return a + b / 2;
}

float mean_vector(vector<float> v){

    int vector_size = v.size();
    float vector_sum = std::accumulate(v.begin(), v.end(), 0);

    return vector_sum / vector_size;
}

float std_vector (vector<float> v){

    float mean = mean_vector(v);

    float sum_differences = 0;

    for (auto it : v){
        float sample_difference_from_mean_squared = pow(it - mean, 2.0);
        sum_differences += sample_difference_from_mean_squared;
    }

    float standard_deviation = sqrt(sum_differences / v.size());
    return standard_deviation;

}

struct Coordinates {
    float x, y;
};

bool is_x_less_than(const Coordinates& first, const Coordinates& second) { return first.x < second.x; }

void print_vector_struct(vector<Coordinates> c){
    for (auto vectorit = c.begin(); vectorit != c.end(); ++vectorit){
        cout << vectorit->x << ", " << vectorit->y << endl;
    }
}

void print_vector(vector<float> v){
    for (auto it : v){
        cout << it << endl;
    }
}


vector<Coordinates> linear_interpolation (vector<Coordinates> c, vector<float> x_list){

    // for each x check which coordinates of x it lies between
        // insert x_target into coordinates struct (set y_target to 0?)
        // sort order new coordinates struct
        // look for -1 index x,y value and +1 index x, y value from where x_target was inserted to get values inbetween
        // compute y_target based on linear interpolation formula


    for (auto x : x_list){

        // insert new point x into struct
        Coordinates c_insert = {x, 0};
        c.insert(c.begin(), c_insert);
        // print_vector_struct(c);

        // sort struct by order of x coordinate
        std::sort(c.begin(), c.end(), is_x_less_than);
        // print_vector_struct(c);

        // (a) get index of inserted x value in sorted position
        // (b) from before and after index value, calculate interploated y value for x

        float y_interp;
        int x_found_pos;
        Coordinates c_insert_full;

        for (size_t i = 0; i < c.size(); ++i){
            if (c[i].x == x){
                // cout << "Index position is: " << i << endl;

                y_interp = (c[i-1].y*(c[i+1].x - x) + c[i+1].y*(x - c[i-1].x)) / (c[i+1].x - c[i-1].x);
                // cout<< "y_interp is: " << y_interp << endl;
                x_found_pos = i;
                c_insert_full = {x, y_interp};

            }

        }
        c.at(x_found_pos) = c_insert_full;
        // print_vector_struct(c);

    }
    return c;
}


float covariance(vector<float> x, vector<float> y){

    // calculate mean of x and y
    float x_mean = mean_vector(x);
    float y_mean = mean_vector(y);
    float sum_product = 0;

    for(int i=0; i < x.size(); i++){
        float index_product = (x[i] - x_mean) * (y[i] - y_mean);
        sum_product += index_product;
    }

    return sum_product / (x.size() - 1);
}

float correlation(vector<float> x, vector<float> y){

    // calculate mean of x and y
    float x_mean = mean_vector(x);
    float y_mean = mean_vector(y);
    float sum_product = 0;
    float x_from_mean_squared_sum = 0;
    float y_from_mean_squared_sum = 0;

    for(int i=0; i < x.size(); i++){
        float index_product = (x[i] - x_mean) * (y[i] - y_mean);
        float x_from_mean_squared_index = pow((x[i] - x_mean), 2);
        float y_from_mean_squared_index = pow((y[i] - y_mean), 2);

        sum_product += index_product;
        x_from_mean_squared_sum += x_from_mean_squared_index;
        y_from_mean_squared_sum += y_from_mean_squared_index;
    }
    return sum_product /  sqrt((x_from_mean_squared_sum * y_from_mean_squared_sum));

}
