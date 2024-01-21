# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 09:10:42 2024

@author: Dilhara22034062
"""

import errors as err
import scipy.optimize as opt
import matplotlib.cm as cm
from scipy.stats import mstats
import sklearn.preprocessing as pp
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
import sklearn.cluster as cluster
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


def read_clean_cvs(file_path):
    co2emi = pd.read_csv("co2kgpcapta.csv", encoding='windows-1254')
    co2emi = co2emi[(co2emi["1990 [YR1990]"].notna()) &
                    (co2emi["2020 [YR2020]"].notna())]
    co2emi = co2emi.reset_index(drop=True)
    co2emi["1990 [YR1990]"] = pd.to_numeric(
        co2emi["1990 [YR1990]"], errors='coerce')
    co2emi["2020 [YR2020]"] = pd.to_numeric(
        co2emi["2020 [YR2020]"], errors='coerce')
    return co2emi


def cal_growth(data):
    growth = co2emi[["Country Name", "1990 [YR1990]"]].copy()
    # and calculate the growth over 30 years
    growth["Growth"] = 100.0/30.0 * \
        (co2emi["2020 [YR2020]"]-co2emi["1990 [YR1990]"]) / \
        co2emi["1990 [YR1990]"]
    return growth

# print(growth.describe())
# print(growth.dtypes)


def drop_desc(data):
    data = data.dropna()
    data = data[~(data.isin([np.inf, -np.inf]).any(axis=1))]
    print(data.describe())
    return data


def scatter_plot(data, x_label, y_label):
    plt.figure(figsize=(8, 8))
    plt.scatter(growth["1990 [YR1990]"], growth["Growth"])
    plt.xlabel("CO2 emissions (metric tons per capita) in 1990")
    plt.ylabel("Growth per year [%]")
    plt.show()


# create a scaler object
scaler = pp.RobustScaler()


def scale_and_plot(data, scaler, x_label, y_label):
    dt_ex = co2emi[["1990 [YR1990]", "2020 [YR2020]"]]
    print(growth.describe())
    scaler.fit(dt_ex)
    # apply the scaling
    norm = scaler.transform(dt_ex)
    plt.figure(figsize=(8, 8))
    plt.scatter(norm[:, 0], norm[:, 1])
    plt.xlabel(" CO2 emision per head 1990")
    plt.ylabel("Growth per year [%]")
    plt.show()
    return norm


def one_silhoutte(xy, n):
    """ Calculates silhoutte score for n clusters """
    xy_no_nan = np.nan_to_num(xy, nan=0)
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy_no_nan)     # fit done on x,y pairs

    labels = kmeans.labels_

    # calculate the silhoutte score
    score = (skmet.silhouette_score(xy_no_nan, labels))

    return score


def print_silhouette_scores(norm):
    # calculate silhouette score for 2 to 10 clusters
    for ic in range(2, 11):
        score = one_silhoutte(norm, ic)
        # allow for minus signs
        print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")


def kmeans_and_plot(norm, n_clusters, scaler, co2emi):
    norm_no_nan = np.nan_to_num(norm, nan=0)
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=3, n_init=20)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(norm_no_nan)     # fit done on x,y pairs
    # extract cluster labels
    labels = kmeans.labels_
    # extract the estimated cluster centres and convert to original scales
    cen = kmeans.cluster_centers_
    cen = scaler.inverse_transform(cen)
    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]

    plt.figure(figsize=(8.0, 8.0))
    cmap = cm.get_cmap('viridis')
    # plot data with kmeans cluster number
    plt.scatter(co2emi["1990 [YR1990]"], co2emi["2020 [YR2020]"],
                10, labels, marker="o", cmap=cmap)
    # show cluster centres
    plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")

    plt.xlabel("Co2 emission")
    plt.ylabel("Co2 emission/year [%]")

    plt.show()


if __name__ == "__main__":
    file_path = "co2kgpcapta.csv"
    co2emi = read_clean_cvs(file_path)

    growth = cal_growth(co2emi)
    print(growth.describe())

    growth = drop_desc(growth)
    scatter_plot(growth, "1990 [YR1990]", "Growth")

    scaler = pp.RobustScaler()
    norm = scale_and_plot(
        growth, scaler, "CO2 emission per head 1990", "Growth per year [%]")

    print_silhouette_scores(norm)

    n_clusters = 3
    kmeans_and_plot(norm, n_clusters, scaler, co2emi)


def read_population_data(file_path):  # read csv file
    """Reads population data from a CSV file into a DataFrame."""
    df_pop = pd.read_csv(file_path)
    return df_pop


# calculate exponential growth
def exponential_growth_model(t, scale, growth, t_offset=1960):
    """Computes exponential growth model."""
    t = t - t_offset
    f = scale * np.exp(growth * t)
    return f


def fit_exponential_growth(df, initial_guess=None):
    """Fits exponential growth model to population data."""
    if initial_guess is None:
        initial_guess = [4e8, 0.03]

    popt, _ = opt.curve_fit(exponential_growth_model,
                            df["Year"], df["Population"], p0=initial_guess)
    return popt


def plot_exponential_growth(df, popt):  # plot exponentialgrowth
    """Plots exponential growth model."""
    df["pop_exp"] = exponential_growth_model(df["Year"], *popt)
    plt.figure()
    plt.plot(df["Year"], df["Population"], label="data")
    plt.plot(df["Year"], df["pop_exp"], label="fit")
    plt.legend()
    plt.title("Exponential Growth Model")
    plt.show()


def logistics_model(t, a, k, t0):  # calculate logistic model
    """Computes logistics model."""
    f = a / (1.0 + np.exp(-k * (t - t0)))
    return f


def fit_logistics(df, initial_guess=None):
    """Fits logistics model to population data."""
    if initial_guess is None:
        initial_guess = [16e8, 0.04, 1985]

    popt, _ = opt.curve_fit(
        logistics_model, df["Year"], df["Population"], p0=initial_guess, method='trf', maxfev=10000)
    return popt


def plot_logistics(df, popt):
    """Plots logistics model."""
    df["pop_logistics"] = logistics_model(df["Year"], *popt)
    plt.figure()
    plt.plot(df["Year"], df["Population"], label="data")
    plt.plot(df["Year"], df["pop_logistics"], label="fit")
    plt.legend()
    plt.title("Logistics Model")
    plt.show()


def plot_with_error(df, popt, pcovar):
    """Plots data with error bars using extrapolation."""
    years = np.linspace(1960, 2030)
    pop_logistics = logistics_model(years, *popt)

    sigma = err.error_prop(years, logistics_model, popt, pcovar)
    low = pop_logistics - sigma
    up = pop_logistics + sigma

    plt.figure()
    plt.title("Logistics Model with Error Bars")
    plt.plot(df["Year"], df["Population"], label="data")
    plt.plot(years, pop_logistics, label="fit")
    plt.fill_between(years, low, up, alpha=0.5, color="y")
    plt.legend(loc="upper left")
    plt.show()


def main():
    file_path = "AuPopulation.csv"
    df_pop = read_population_data(file_path)

    # Exponential Growth
    popt_exp = fit_exponential_growth(df_pop)
    plot_exponential_growth(df_pop, popt_exp)

    # Logistics Model
    popt_logistics = fit_logistics(df_pop)
    plot_logistics(df_pop, popt_logistics)

    # Plot with Error Bars
    _, pcovar_logistics = opt.curve_fit(
        logistics_model, df_pop["Year"], df_pop["Population"], p0=popt_logistics, method='trf', maxfev=10000)
    plot_with_error(df_pop, popt_logistics, pcovar_logistics)

    # Population in 2030
    pop_2030 = logistics_model(2030, *popt_logistics)
    sigma_2030 = err.error_prop(
        2030, logistics_model, popt_logistics, pcovar_logistics)
    print(
        "Population in 2030: {:.0f} +/- {:.0f} million".format(pop_2030/1e6, sigma_2030/1e6))


if __name__ == "__main__":
    main()
