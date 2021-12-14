import textwrap
import os
import joblib

def coeff_model(model):
    loaded_model = joblib.load(model)
    return loaded_model.coef_[0], loaded_model.intercept_

def predict_function():
    coeff = coeff_model("./model.joblib")
    thetas = coeff[0]
    bias = coeff[1]
    thetas_str = "{"
    n_thetas = len(thetas)
    for i in range(n_thetas - 1):
        thetas_str += str(thetas[i])
        thetas_str += ","
    thetas_str += str(thetas[n_thetas - 1])
    thetas_str += "}"

    c_function = textwrap.dedent(f"""
    #include <stdio.h>
    #include <stdlib.h>
    float linear_regression_prediction(float* features, int n_feature)
    {{
        float res = %f;
        int n_thetas = %d;
        float thetas[] = %s;
        for (int i = 0; i < n_thetas; i++)
        {{
            res += features[i] * thetas[i];
        }}
        return res;
    }}
    """) % (bias, n_thetas, thetas_str)
    return c_function

def main_functions():
    c_function = textwrap.dedent(f"""
    int main(int argc, char *argv[])
    {{
        int n_features = argc - 1;
        char *first_features = argv[1];
        char *second_features = argv[2];
        float array[] = {{atof(first_features), atof(second_features)}};
        printf(\"%f\", linear_regression_prediction(array, n_features));
    }}""")
    return c_function

if __name__ == '__main__':
    f = open("linear_regression.c", "w")
    f.write(predict_function())
    f.write(main_functions())
    f.close()
    os.system("gcc linear_regression.c")