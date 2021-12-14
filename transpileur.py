import textwrap
import os
import joblib
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def save_model():
    data = pd.read_csv('./tumors.csv')
    X = data.iloc[:, :2].values
    y = data.iloc[:, 2:].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    regr = LinearRegression().fit(X_train, y_train)
    filename = 'model.joblib'
    joblib.dump(regr, open(filename, 'wb'))
    return regr, X_test

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
    model, X_test = save_model()
    f = open("linear_regression.c", "w")
    f.write(predict_function())
    f.write(main_functions())
    f.close()
    os.system("gcc linear_regression.c")
    print("Sklearn predict: \n", model.predict(X_test[:3]) , "\n")
    print ("Transpiler predict:")
    for i in range(len(X_test[:3])):
        command = "./a.out " + str(X_test[i][0]) + " " +  str(X_test[i][1])
        os.system(command)
        print()