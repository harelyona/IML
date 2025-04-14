from house_price_prediction import *
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import xlabel, ylabel


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
    return data_frame


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")
    output_path = "plots"
    # Question 3 - Exploring data for specific country
    il_temp = df[df['Country'] == 'Israel']
    temps = il_temp['Temp']
    days_of_year = il_temp['DayOfYear']
    years = il_temp['Year']
    scatter = plt.scatter(days_of_year, temps, c=years,s=0.5)
    xlabel("day of the year")
    ylabel("temperature")
    cbar = plt.colorbar(scatter)
    cbar.set_label('Year')
    plt.title("temperatures vs day of the year in Israel")
    plt.savefig(f"{output_path}{os.sep}temperatures_vs_day_of_year.png")
    plt.show()



    # Question 4 - Exploring differences between countries

    # Question 5 - Fitting model for different values of `k`

    # Question 6 - Evaluating fitted model on different countries

    pass
