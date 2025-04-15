import os

from house_price_prediction import *
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import xlabel, ylabel
from polynomial_fitting import *


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data_frame = pd.read_csv(filename, parse_dates=['Date'])
    data_frame = data_frame.dropna()
    data_frame['DayOfYear'] = data_frame['Date'].dt.dayofyear
    # Filter out rows with extreme temperature values
    data_frame = data_frame[(data_frame['Temp'] >= -50) & (data_frame['Temp'] <= 50)]
    return data_frame


def plot_temp_vs_day_of_year(data_frame: pd.DataFrame, path: str = ".") -> None:
    temps = data_frame['Temp']
    days_of_year = data_frame['DayOfYear']
    years = data_frame['Year']
    scatter = plt.scatter(days_of_year, temps, c=years, s=0.5)
    xlabel("day of the year")
    ylabel("temperature")
    cbar = plt.colorbar(scatter)
    cbar.set_label('Year')
    plt.title("temperatures vs day of the year in Israel")
    plt.savefig(f"{path}{os.sep}temperatures_vs_day_of_year.png")


def plot_std_vs_month(data_frame, path: str = ".") -> None:
    month_stds = data_frame.groupby('Month')['Temp'].agg("std")
    months = month_stds.index
    stds = month_stds.values
    plt.bar(months, stds)
    plt.xlabel("month")
    plt.ylabel("standard deviation")
    plt.title("std of temperatures vs month")
    plt.savefig(f"{path}{os.sep}month_stds.png")


def plot_mean_vs_month(data_frame, path: str = ".") -> None:
    temps = data_frame.groupby(['Month', "Country"])['Temp'].agg(["mean", "std"])
    colors = ['red', 'blue', 'green', 'orange']
    i = 0
    for country in data_frame['Country'].unique():
        country_data = temps.loc[temps.index.get_level_values('Country') == country]
        months = country_data.index.get_level_values('Month')
        means = country_data['mean'].values
        stds = country_data['std'].values
        plt.errorbar(months, means, yerr=stds, fmt='o', label=country,color=colors[i])
        plt.plot(months, means, color=colors[i])
        i = i + 1
    plt.xlabel("Month")
    plt.ylabel("Mean Temperature")
    plt.title("Mean Temperatures with Standard Deviation by Month and Country")
    plt.legend(title="Country")
    plt.savefig(f"{path}{os.sep}means_vs_month.png")


def plot_loss_vs_k(samples, responses, output_path: str = ".") -> PolynomialFitting:
    training_samples, training_response, test_samples, test_response = generate_sets_and_responses(samples, responses)
    training_samples, training_response, test_samples, test_response = training_samples.to_numpy().flatten(), training_response.to_numpy(), test_samples.to_numpy().flatten(), test_response.to_numpy()
    ks = range(1, 11)
    losses = []
    k4fit = None
    for k in ks:
        polyfit = PolynomialFitting(k)
        polyfit.fit(training_samples, training_response)
        loss = round(polyfit.loss(test_samples, test_response), 2)
        print(f"Loss for k={k}: {loss}")
        losses.append(loss)
        if k == 4:
            k4fit = polyfit
    plt.bar(ks, losses)
    plt.xlabel("polynomial degree")
    plt.ylabel("loss")
    plt.title("Loss vs polynomial degree")
    plt.savefig(f"{output_path}{os.sep}loss vs polynomial degree.png")
    return k4fit


def loss_per_country(data, feature):
    countries = data['Country'].unique().tolist()
    countries.remove('Israel')
    losses = []
    for country in countries:
        county_data = data[data['Country'] == country]
        loss = fit.loss(county_data[feature], county_data['Temp'])
        losses.append(loss)
    plt.bar(countries, losses)
    plt.xlabel("country")
    plt.ylabel("loss")
    plt.title("loss for different countries")
    plt.savefig(f"{output_path}{os.sep}loss for different countries.png")


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")
    output_path = "plots"

    # Question 3 - Exploring data for specific country
    il_temp = df[df['Country'] == 'Israel']
    plot_temp_vs_day_of_year(il_temp, output_path)
    plt.clf()
    plot_std_vs_month(il_temp, output_path)
    plt.clf()

    # Question 4 - Exploring differences between countries
    plot_mean_vs_month(df, output_path)
    plt.clf()

    # Question 5 - Fitting model for different values of `k`
    feature = "DayOfYear"
    y = il_temp['Temp']
    data5 = il_temp[feature].to_frame()
    fit = plot_loss_vs_k(data5, y, output_path)
    plt.show()
    plt.clf()
    # Question 6 - Evaluating fitted model on different countries
    loss_per_country(df, feature)

