import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random 
from scipy.stats import ttest_ind


def visual_analysis(df):
    os.makedirs("plots", exist_ok=True)

    # inspecting distribution of categories
    plt.figure()
    df['Category1'].value_counts().plot(kind='bar')
    plt.title('Distribution of Category1')
    plt.xlabel('Category1')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig("plots/category1_distribution.png")
    plt.close()

    # inspecting distribution of continuous variables
    plt.figure()
    df['Value1'].hist(bins=30, alpha=0.5, label='Value1')
    df['Value2'].hist(bins=30, alpha=0.5, label='Value2')
    plt.title('Distribution of Continuous Variables')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/continuous_variables_distribution.png")
    plt.close()

def category_probabilities(file):
    df = pd.read_csv(file, sep=";")
    category_ratios = df['Category1'].value_counts(normalize=True).sort_index()
    print("Category ratios:")
    print(category_ratios)
    return category_ratios.tolist()

def standard_deviations(file):
    df = pd.read_csv(file, sep=";")
    std_value1 = df["Value1"].std()
    std_value2 = df["Value2"].std()
    print(f"Standard deviation of Value1: {std_value1}")
    print(f"Standard deviation of Value2: {std_value2}")
    return {"Value1_std": std_value1, "Value2_std": std_value2}

def mean_values(file):
    df = pd.read_csv(file, sep=";")
    mean_value1 = df["Value1"].mean()
    mean_value2 = df["Value2"].mean()
    print(f"Mean of Value1: {mean_value1}")
    print(f"Mean of Value2: {mean_value2}")
    return {"Value1_mean": mean_value1, "Value2_mean": mean_value2}

#We will do this using standard python random
def generate_sample_dataset(
    num_samples=1000000,
    categories=None,
    category_probabilities=None,
    value1_mean=10,
    value1_std=2,
    value2_mean=20,
    value2_std=6
):
    if categories is None:
        categories = ["A", "B", "C", "D", "E"]
    if category_probabilities is None:
        category_probabilities = [0.2, 0.4, 0.2, 0.1, 0.1]

    with open("sample_dataset.csv", "w") as f:
        f.write("Category1,Value1,Value2\n")
        for _ in range(num_samples):
            category = random.choices(categories, weights=category_probabilities, k=1)[0]
            value1 = random.gauss(value1_mean, value1_std)
            value2 = random.gauss(value2_mean, value2_std)
            f.write(f"{category},{value1},{value2}\n")


# Compare two datasets using independent T-test for Value1 and Value2 columns.
def compare_datasets_ttest(file1="dataset.csv", file2="sample_dataset.csv"):
    df1 = pd.read_csv(file1, sep=";")
    df2 = pd.read_csv(file2)

    results = {}
    for col in ["Value1", "Value2"]:
        stat, pval = ttest_ind(df1[col], df2[col], equal_var=False)
        results[col] = {"t_statistic": stat, "p_value": pval}
        print(f"T-test for {col}: t={stat:.4f}, p={pval:.4e}")

    return results

def generate_data(num_samples = 1000000):
    np.random.seed(42)

    df = pd.DataFrame(
        # I can already tell here that the distributions are normal
        {
        "Category1": np.random.choice(["A", "B", "C", "D", "E"],
        num_samples, p=[0.2, 0.4, 0.2, 0.1, 0.1]), # Probabilities for which category the variable will take


        "Value1": np.random.normal(10, 2, #Standard deviation of 2 and mean of 10. 
                                #Meaning for a normal distribution most of the values will be in 
                                #the range of 6 to 14 (95% of the values will be in this range)
        num_samples), # Continuous variable 

        "Value2": np.random.normal(20, 6, #Standard deviation of 6 and mean of 20. 
                                #Meaning for a normal distribution most of the values will be in 
                                #the range of 8 to 32(95% of the values will be in this range)
        num_samples), # Continuous variable 
        }
    )
    df.to_csv("dataset.csv", sep=";")

    return df


if __name__ == "__main__":
# Generate synthetic dataset
    num_samples = 1000000
    df = generate_data(num_samples)

    #visual_analysis(df)

    means = mean_values("dataset.csv")
    std_div = standard_deviations("dataset.csv")
    category_prob = category_probabilities("dataset.csv")

    generate_sample_dataset(value1_mean=means["Value1_mean"],
                            value1_std=std_div["Value1_std"],
                            value2_mean=means["Value2_mean"],
                            value2_std=std_div["Value2_std"],
                            categories=["A", "B", "C", "D", "E"],
                            category_probabilities=category_prob)

    visual_analysis(pd.read_csv("sample_dataset.csv"))

    print(compare_datasets_ttest())

# Using the visiual analysis as well as the code analysis conducted, we can safely
# assume that the dataset is, 1. Normally distributed, 2. Has a categorical variable with 5 categories, 
# B being most common, followed by A, C being equally common, and then D and E being least common.         




