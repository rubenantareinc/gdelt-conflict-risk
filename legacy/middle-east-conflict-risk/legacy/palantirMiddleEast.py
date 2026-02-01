"""
We start the script  by importing four essential Python libraries that will help us in out
investigation into Middle East instability. 
Each import serves a specific purpose so:

- NumPy provides the numerical foundation for matrix algebra, which we use to manually implement 
  Ordinary Least Squares (OLS) regression giving us complete control over the model and transparency 
  in how coefficients are derived.

- Pandas is used to handle structured country-level panel data, allowing us to cleanly extract predictors 
  (e.g., GDP growth, refugee flows) and the dependent variable (future conflict events) from a CSV dataset.

- SciPy's `t-distribution` is imported to conduct hypothesis testing for each regression coefficient under 
  small sample assumptions — ensuring the statistical validity of our results by computing p-values accurately.

- Matplotlib is used to visualize critical relationships in our data. Specifically, we use it to plot 
  how refugee counts correlate with future conflict — turning numerical output into an interpretable narrative.

Together, these libraries allow us to transform  raw, into a  statistical model
"""
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
import matplotlib.pyplot as plt


# Here we are siply Loading the dataset as a CSV (i used the # comment instead of """ because this  line is much shorter).
df = pd.read_csv("MiddleEast.csv")


"""
Here we are Definying the independent variables and dependent variable
where we construct the matrix of independent variables (X), 
which includes all the explanatory factors believed to influence future conflict events in Middle Eastern countries.
These predictors are selected based on a combination of theoretical relevance and empirical plausibility.

Each column in X corresponds to one of the following features:

- Conflict_Events_Current: captures the autocorrelation effect since past conflict intensity is often a strong predictor 
  of future instability due to lingering tensions and unresolved grievances.

- GDP_Growth: represents economic performance. Low or negative growth may signal structural weaknesses or 
  governance issues that exacerbate conflict risk.

- Inflation_Rate: high inflation erodes purchasing power and may indicate fiscal mismanagement, which 
  can fuel public discontent.

- Unemployment_Rate: socioeconomic frustration linked to joblessness is frequently a precursor to unrest, 
  especially among youth populations.

- Negative_News_Count: a proxy for political sentiment and media-reported instability. Spikes in negative coverage 
  often precede actual conflict events.

- Refugees: measures the scale of human displacement, which can be both a symptom and a driver of regional instability.

- Election: a binary indicator capturing electoral cycles. Elections in fragile states are often flashpoints 
  for violence, particularly when contested.

- Coup: another binary indicator capturing whether a coup occurred in the current year a major institutional shock 
  that tends to destabilize societies in both the short and long term.

The `.values.tolist()` method converts the filtered DataFrame subset into a raw list-of-lists structure, 
which allows us to manually append an intercept later and apply pure NumPy-based linear algebra operations 
outside the Pandas framework.
"""
X = df[[
    "Conflict_Events_Current",
    "GDP_Growth",
    "Inflation_Rate",
    "Unemployment_Rate",
    "Negative_News_Count",
    "Refugees",
    "Election",
    "Coup"
]].values.tolist()

"""
This line defines the dependent variable `y`, representing the number of conflict events 
expected in the next year. It is the target our model aims to predict using current-year indicators. 
The `.values` method extracts the data as a NumPy array, making it compatible with the matrix 
operations used in OLS regression.
"""
y = df["Conflict_Events_Next"].values


"""
Here we manually add a column of 1 to each row of X to account for the intercept term (β₀) in the regression model. 
Then, both X and y are converted to NumPy arrays for efficient matrix operations in the OLS calculation.
"""
for row in X:
    row.insert(0, 1)

X_np = np.array(X)
y_np = np.array(y)



"""
In this section implements the closed-form OLS solution: β = (XᵀX)^(-1)Xᵀy.
We compute the transpose of X, the Gram matrix (XᵀX), its inverse, and finally multiply by Xᵀy 
& estimate the regression coefficients manually.
"""
Xt = X_np.T
XtX = np.dot(Xt, X_np)
XtX_inv = np.linalg.inv(XtX)
Xty = np.dot(Xt, y_np)
beta = np.dot(XtX_inv, Xty)



"""
We calculate the predicted values (ŷ) using our estimated coefficients, 
then compute residuals as the difference between actual and predicted values. 
We also define `n` (sample size), `k` (number of parameters), and degrees of freedom for residuals.
"""
y_pred = np.dot(X_np, beta)
residuals = y_np - y_pred
n = len(y_np)
k = len(beta)
df_resid = n - k



"""
We compute the residual sum of squares (RSS) to estimate the error variance (σ²). 
Using this, we calculate the variance-covariance matrix of the coefficients and extract 
the standard errors from its diagonal — essential for statistical inference.
"""
rss = np.sum(residuals**2)
sigma_squared = rss / df_resid
var_beta = sigma_squared * XtX_inv
std_errors = np.sqrt(np.diag(var_beta))



"""
We compute t-values by dividing each coefficient by its standard error. 
Using the Student’s t-distribution, we then calculate two-tailed p-values to assess statistical significance. 
Z-values mirror t-values here, as they are conceptually similar in large samples.
"""
t_values = beta / std_errors
p_values = [2 * (1 - t_dist.cdf(abs(t), df_resid)) for t in t_values]
z_values = beta / std_errors



"""
We calculate R-squared to measure how well the model explains the variance in the outcome. 
Adjusted R-squared refines this by penalizing for the number of predictors, giving a more accurate 
measure of model quality when multiple variables are used.
"""
ss_total = np.sum((y_np - np.mean(y_np))**2)
r_squared = 1 - (rss / ss_total)
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / df_resid



"""
This block saves the full set of OLS regression results to a human-readable text file titled 
'OLS_Manual_Results.txt'. Exporting the output serves several key purposes:

1. Reproducibility & Transparency - By saving the coefficients, diagnostics, and test statistics 
   to an external file, we create a permanent, shareable record of the model's performance and 
   assumptions. This is especially important in policy analysis or conflict forecasting, 
   where decisions may be sensitive to how variables are interpreted.

2. Model Diagnostics - The file includes R-squared and adjusted R-squared, which help assess 
   overall model fit, followed by a detailed variable-level breakdown:
   - Beta (coefficient estimates)
   - Standard errors (measure of uncertainty)
   - t-values (signal-to-noise ratio of each predictor)
   - p-values (statistical significance for hypothesis testing)

3. Structured Output - Variables are clearly labeled and aligned in columns for readability, 
   making it easy to interpret results without needing to rerun the model. This format also 
   facilitates peer review, reporting, or further use in LaTeX documents, presentations, or reports.

The inclusion of both economic and political indicators (like GDP growth, coups, and elections) 
makes this output particularly valuable for decision-makers and researchers focused on early warning 
and risk assessment in fragile regions.
"""
with open("OLS_Manual_Results.txt", "w") as f:
    f.write("Manual OLS Regression Results:\n\n")
    f.write(f"R-squared: {r_squared:.4f}\n")
    f.write(f"Adjusted R-squared: {adj_r_squared:.4f}\n\n")
    f.write("Variable\t\tBeta\t\tStd Err\t\tt-Value\t\tp-Value\n")
    var_names = ["Intercept", "Conflict_Events_Current", "GDP_Growth", "Inflation_Rate",
                 "Unemployment_Rate", "Negative_News_Count", "Refugees", "Election", "Coup"]
    for i in range(len(beta)):
        f.write(f"{var_names[i]:<24}{beta[i]:.4f}\t\t{std_errors[i]:.4f}\t\t{t_values[i]:.4f}\t\t{p_values[i]:.4f}\n")




"""
This plot visualizes the relationship between refugee flows and future conflict events. 
Meaning wee use a scatter plot to show raw data and overlay a best-fit line to highlight the trend. 
This helps translate statistical output into an intuitive, visual.
"""
plt.figure(figsize=(8, 6))
plt.scatter(df["Refugees"], y_np, alpha=0.6)
slope, intercept = np.polyfit(df["Refugees"], y_np, 1)
plt.plot(df["Refugees"], slope * df["Refugees"] + intercept, color='red')
plt.xlabel("Refugee Flow (millions)")
plt.ylabel("Future Conflict Events")
plt.title("Refugees vs Future Conflict")
plt.grid(True)
plt.tight_layout()
plt.savefig("Refugees_vs_Conflict_Manual.png")
plt.close()



# Additional Visualizations for Deeper Insight
# This histogram reveals the distribution of predicted conflict events across all observations.
# It highlights whether conflict levels are concentrated or spread out across countries/years.
plt.figure(figsize=(8, 6))
plt.hist(df["Conflict_Events_Next"], bins=15, color='skyblue', edgecolor='black')
plt.xlabel("Number of Future Conflict Events")
plt.ylabel("Frequency")
plt.title("Distribution of Future Conflict Events")
plt.grid(True)
plt.tight_layout()
plt.savefig("Histogram_Future_Conflict.png")
plt.close()

# This histogram displays how refugee flows are distributed across the dataset.
# It helps identify outliers or skewness in population displacement patterns.
plt.figure(figsize=(8, 6))
plt.hist(df["Refugees"], bins=15, color='lightgreen', edgecolor='black')
plt.xlabel("Refugees (millions)")
plt.ylabel("Frequency")
plt.title("Distribution of Refugee Flows")
plt.grid(True)
plt.tight_layout()
plt.savefig("Histogram_Refugees.png")
plt.close()

# This scatter plot explores the link between GDP growth and future conflict.
# It tests the hypothesis that economic stagnation or decline may trigger instability.
plt.figure(figsize=(8, 6))
plt.scatter(df["GDP_Growth"], df["Conflict_Events_Next"], alpha=0.6)
slope, intercept = np.polyfit(df["GDP_Growth"], df["Conflict_Events_Next"], 1)
plt.plot(df["GDP_Growth"], slope * df["GDP_Growth"] + intercept, color='red')
plt.xlabel("GDP Growth (%)")
plt.ylabel("Future Conflict Events")
plt.title("GDP Growth vs Future Conflict")
plt.grid(True)
plt.tight_layout()
plt.savefig("GDP_vs_Conflict.png")
plt.close()

# This scatter plot shows how unemployment rates relate to future conflict events.
# It evaluates whether high joblessness is a consistent predictor of unrest.
plt.figure(figsize=(8, 6))
plt.scatter(df["Unemployment_Rate"], df["Conflict_Events_Next"], alpha=0.6)
slope, intercept = np.polyfit(df["Unemployment_Rate"], df["Conflict_Events_Next"], 1)
plt.plot(df["Unemployment_Rate"], slope * df["Unemployment_Rate"] + intercept, color='red')
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Future Conflict Events")
plt.title("Unemployment Rate vs Future Conflict")
plt.grid(True)
plt.tight_layout()
plt.savefig("Unemployment_vs_Conflict.png")
plt.close()
