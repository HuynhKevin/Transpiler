#include <stdio.h> 
#include <stdlib.h> 

float linear_regression_prediction(float* features, int n_feature) 
{
    float res = 1.038166; 
    int n_thetas = 2; 
    float thetas[] = {3.1758576989688354,-222.96141367816955}; 
    for (int i = 0; i < n_thetas; i++) 
    { 
        res += features[i] * thetas[i]; 
    } 
    return res; 
}

int main(int argc, char *argv[])
{
    int n_features = argc - 1;
    char *first_features = argv[1];
    char *second_features = argv[2];
    float array[] = {atof(first_features), atof(second_features)};
    printf("%f", linear_regression_prediction(array, n_features));
}