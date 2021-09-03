from typing import List, Tuple, Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_a_mod_list_and_dict(
    df: pd.DataFrame, cluster_range: List, seed: int = 10
) -> Tuple[Dict, List]:
    """Fit multiple KMeans model and store them in both a list and a dictionary
    Args:
        df (pd.DataFrame): the preprocessed and standardized dataframe
        cluster_range (List): the range of cluster we will try to fit
        seed (int, optional): the random state for the KMeans model. Defaults to 10.
    Returns:
        Tuple: a tuple with two elements:
            mod_dict - a dictionary where each key is a str of the number of clusters, the value is the fitted value
            mod_list - a list of fitted models
    """
    mod_dict = {}
    for k in cluster_range:
        mod = KMeans(n_clusters=k, random_state=seed).fit(df)
        mod_dict[str(k)] = mod
    mod_list = list(mod_dict.values())
    return mod_dict, mod_list


def cluster_selection_plot(
    df: pd.DataFrame, mod_list: List, cluster_range: List
) -> None:
    """Make and save inertia and silhouette plot
    Args:
        df (pd.DataFrame): the preprocessed and standardized dataframe
        mod_list (List): a list of fitted KMeans model
        cluster_range (List): the range of cluster including in the list of fitted model
        output_path (str): the path to save the cluster selection plots
    """
    within_ss = [i.inertia_ for i in mod_list]
    silhouette_list = [silhouette_score(df, i.labels_) for i in mod_list]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    axs[0].plot(cluster_range, within_ss, color="royalblue")
    axs[1].plot(cluster_range, silhouette_list, color="green")

    # Format x and y label in the plot
    for i in range(2):
        axs[i].set_xlabel("number of clusters")
    for idx, name in zip([0, 1, 2], ["inertia", "silhouette score"]):
        axs[idx].set_ylabel(name)
    plt.show()


def summary_kmeans(df: pd.DataFrame, mod) -> pd.DataFrame:
    """Summary statistics for each cluster for kmeans"""
    df_summary = pd.DataFrame(mod.cluster_centers_, columns=list(df.columns))
    df_summary["count"] = pd.Series(mod.labels_).value_counts().sort_index()
    df_summary["percent"] = df_summary["count"] / df_summary["count"].sum()
    columns_ordered = ["count", "percent"] + list(df.columns)
    df_summary = df_summary[columns_ordered]
    return df_summary


def summary_plot(df: pd.DataFrame, mod) -> None:
    """Plot the summary statistics of kmeans"""
    df_summary = summary_kmeans(df, mod)
    x_max = df_summary.iloc[:, 2:].values.max()
    x_min = df_summary.iloc[:, 2:].values.min()

    fig, axs = plt.subplots(nrows=1, ncols=mod.n_clusters, figsize=(25, 5))
    for i in range(mod.n_clusters):
        if i > 0:
            axs[i].get_yaxis().set_ticklabels([])
        axs[i].scatter(
            x=df_summary.iloc[i, 2:].values, y=list(df.columns), color="royalblue"
        )
        axs[i].set_xlabel(f"cluster{str(i)}")
        axs[i].set_xlim(x_min - 0.5, x_max + 0.5)
    plt.show()


def lift_plot(df_scale: pd.DataFrame, mod, feature_names: List) -> None:
    df = df_scale.copy(deep=True)
    df["cluster_label"] = mod.predict(df)
    cluster_means = df.groupby("cluster_label")[feature_names].mean()
    population_means = df[feature_names].mean()
    lifts = cluster_means.divide(population_means)

    fig, ax = plt.subplots(figsize=(16, 10))

    xticklabels = lifts.index.tolist()
    yticklabels = lifts.columns.tolist()

    ax = sns.heatmap(
        lifts.T,
        center=1,
        vmax=2.5,
        cmap=sns.diverging_palette(10, 220, sep=80, n=7),
        xticklabels=xticklabels,
        yticklabels=yticklabels,
    )
    ax.set_xlabel("Cluster number")
    ax.set_title("Lift in cluster features (Cluster mean/population mean)")
    plt.show()
