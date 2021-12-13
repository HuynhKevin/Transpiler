import joblib
import textwrap

def coeff_model(model):
    loaded_model = joblib.load(model)
    return loaded_model.coef_[0], loaded_model.intercept_

def predict_function():
    coeff = coeff_model("./model")
    thetas = coeff[0]
    bias = coeff[1]
    thetas_str = "{"
    n_thetas = len(thetas)
    for i in range(n_thetas - 1):
        thetas_str += str(thetas[i])
        thetas_str += ","
    thetas_str += str(thetas[n_thetas - 1])
    thetas_str += "}"

    c_function = textwrap.dedent("""\
    #include <stdio.h> \n\
    #include <stdlib.h> \n\n\
    float linear_regression_prediction(float* features, int n_feature) \n\
    {\n\
        float res = %f; \n\
        int n_thetas = %d; \n\
        float thetas[] = %s; \n\
        for (int i = 0; i < n_thetas; i++) \n\
        { \n\
            res += features[i] * thetas[i]; \n\
        } \n\
        return res; \n\
    }\n\n""") % (bias, n_thetas, thetas_str)
    return c_function

def main_functions():
    c_function = textwrap.dedent("""\
    int main(int argc, char *argv[])\n\
    {\n\
        int n_features = argc - 1;\n\
        char *first_features = argv[1];\n\
        char *second_features = argv[2];\n\
        float array[] = {atof(first_features), atof(second_features)};\n\
        printf(\"%f\", linear_regression_prediction(array, n_features));\n\
    }""")
    return c_function

if __name__ == '__main__':
    f = open("linear_regression.c", "w")
    f.write(predict_function())
    f.write(main_functions())
    f.close()