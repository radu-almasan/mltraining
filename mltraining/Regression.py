import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_fuel_consumption_against_emissions():
    plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
    plt.xlabel("FUELCONSUMPTION_COMB")
    plt.ylabel("Emission")
    plt.show()


def plot_engine_size_against_emissions():
    plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

def plot_numer_of_cylinders_against_emissions():
    plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='red')
    plt.xlabel("Cylinders")
    plt.ylabel("Emission")
    plt.show()


def split_data_set():
    msk = np.random.rand(len(df)) < 0.8
    return (cdf[msk]), (cdf[~msk])


def train_and_get_model():
    from sklearn import linear_model
    model = linear_model.LinearRegression()
    train_x = np.asanyarray(train_data[['ENGINESIZE']])
    train_y = np.asanyarray(train_data[['CO2EMISSIONS']])
    model.fit(train_x, train_y)

    # The coefficients
    print ('Coefficients: ', model.coef_)
    print ('Intercept: ', model.intercept_)

    return model


def plot_regression_line():
    train_x = np.asanyarray(train_data[['ENGINESIZE']])
    plt.scatter(train_data.ENGINESIZE, train_data.CO2EMISSIONS, color='blue')
    plt.plot(train_x, model.coef_[0][0]*train_x + model.intercept_[0], '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()


def predict():
    from sklearn.metrics import r2_score
    test_x = np.asanyarray(test_data[['ENGINESIZE']])
    test_y = np.asanyarray(test_data[['CO2EMISSIONS']])
    test_y_ = model.predict(test_x)
    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y_, test_y))


if __name__ == '__main__':
    df = pd.read_csv("../FuelConsumption.csv")
    cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
    viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]

    viz.hist()
    plt.show()

    plot_fuel_consumption_against_emissions()
    plot_engine_size_against_emissions()
    plot_numer_of_cylinders_against_emissions()

    train_data, test_data = split_data_set()
    model = train_and_get_model()
    plot_regression_line()
    predict()
