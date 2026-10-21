import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

output_dir = r'c:\Users\lab52\.antigravity-ide\HW2\artifacts'
os.makedirs(output_dir, exist_ok=True)

print("=== [1] Loading Data ===")
raw_path = r'c:\Users\lab52\.antigravity-ide\HW2\flood_train.csv'
df_full = pd.read_csv(raw_path)
print(f"Full dataset shape: {df_full.shape}")

# Sample 30,000 rows for rigorous statistical analysis & fast reproducible execution
np.random.seed(42)
df = df_full.sample(n=30000, random_state=42).reset_index(drop=True)
if 'id' in df.columns:
    df = df.drop(columns=['id'])

feature_cols = [c for c in df.columns if c != 'FloodProbability']
target_col = 'FloodProbability'
print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Target: {target_col}")

# Save sample dataset for submission / reproduction
df.to_csv(r'c:\Users\lab52\.antigravity-ide\HW2\flood_sample_30k.csv', index=False)
print("Saved flood_sample_30k.csv")

# === [2] Exploratory Data Analysis ===
print("=== [2] EDA & Correlation Analysis ===")
summary_stats = df.describe().T
summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))

# Correlation heatmap
corr_matrix = df.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-0.1, vmax=0.3, annot=False, linewidths=0.5)
plt.title('Correlation Heatmap of Flood Prediction Features & Target', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
plt.close()

# Correlation with target
corr_target = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
colors = sns.color_palette("Blues_r", len(corr_target))
sns.barplot(x=corr_target.values, y=corr_target.index, hue=corr_target.index, palette=colors, legend=False)
plt.title('Pearson Correlation with FloodProbability', fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'target_correlation_ranking.png'), dpi=300)
plt.close()

# === [3] Train/Test Split & Scaling ===
print("=== [3] Train/Test Split & Standardization ===")
X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

# === [4] Feature Selection (5 Algorithms, Target k=14) ===
print("=== [4] Feature Selection (5 Algorithms) ===")
K_FEATURES = 14

# 1. SelectKBest (f_regression)
skb = SelectKBest(score_func=f_regression, k=K_FEATURES)
skb.fit(X_train_scaled, y_train)
skb_selected = [feature_cols[i] for i in skb.get_support(indices=True)]
skb_scores = pd.Series(skb.scores_, index=feature_cols).sort_values(ascending=False)

# 2. RFE (LinearRegression)
rfe = RFE(estimator=LinearRegression(), n_features_to_select=K_FEATURES)
rfe.fit(X_train_scaled, y_train)
rfe_selected = [feature_cols[i] for i in rfe.get_support(indices=True)]
rfe_ranking = pd.Series(rfe.ranking_, index=feature_cols).sort_values()

# 3. Sequential Feature Selector (Forward SFS)
sfs = SequentialFeatureSelector(estimator=LinearRegression(), n_features_to_select=K_FEATURES, direction='forward', cv=3, n_jobs=-1)
sfs.fit(X_train_scaled, y_train)
sfs_selected = [feature_cols[i] for i in sfs.get_support(indices=True)]

# 4. Lasso (L1 Regularization Feature Selection)
lasso_fs = Lasso(alpha=0.0005, random_state=42)
lasso_fs.fit(X_train_scaled, y_train)
lasso_coefs = pd.Series(np.abs(lasso_fs.coef_), index=feature_cols).sort_values(ascending=False)
lasso_selected = lasso_coefs.head(K_FEATURES).index.tolist()

# 5. Permutation Importance
base_lr = LinearRegression().fit(X_train_scaled, y_train)
perm_imp = permutation_importance(base_lr, X_train_scaled, y_train, n_repeats=5, random_state=42)
perm_series = pd.Series(perm_imp.importances_mean, index=feature_cols).sort_values(ascending=False)
perm_selected = perm_series.head(K_FEATURES).index.tolist()

# Feature Selection Summary Table
fs_df = pd.DataFrame({
    'SelectKBest': [f in skb_selected for f in feature_cols],
    'RFE': [f in rfe_selected for f in feature_cols],
    'SFS (Forward)': [f in sfs_selected for f in feature_cols],
    'Lasso (L1)': [f in lasso_selected for f in feature_cols],
    'Permutation Imp': [f in perm_selected for f in feature_cols]
}, index=feature_cols)
fs_df['Total Votes'] = fs_df.sum(axis=1)
fs_df = fs_df.sort_values(by='Total Votes', ascending=False)
fs_df.to_csv(os.path.join(output_dir, 'feature_selection_comparison.csv'))

# Plot Feature Selection Votes
plt.figure(figsize=(10, 6))
sns.barplot(x=fs_df['Total Votes'], y=fs_df.index, hue=fs_df.index, palette='viridis', legend=False)
plt.title(f'Feature Selection Consensus (Top {K_FEATURES} Features across 5 Methods)', fontsize=14, fontweight='bold')
plt.xlabel('Consensus Votes (out of 5 Algorithms)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_selection_votes.png'), dpi=300)
plt.close()

# Selected 14 features by top consensus
top_14_features = fs_df.head(K_FEATURES).index.tolist()
print(f"Top 14 Consensus Features: {top_14_features}")

# === [5] Model Training & Evaluation ===
print("=== [5] Model Training & Comparison ===")
def evaluate_model(model, X_tr, X_te, y_tr, y_te, name="Model"):
    model.fit(X_tr, y_tr)
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)
    
    n_tr, p_tr = X_tr.shape
    n_te, p_te = X_te.shape
    
    r2_tr = r2_score(y_tr, y_pred_tr)
    r2_te = r2_score(y_te, y_pred_te)
    
    adj_r2_tr = 1 - (1 - r2_tr) * (n_tr - 1) / (n_tr - p_tr - 1)
    adj_r2_te = 1 - (1 - r2_te) * (n_te - 1) / (n_te - p_te - 1)
    
    return {
        'Model': name,
        'Train MAE': mean_absolute_error(y_tr, y_pred_tr),
        'Test MAE': mean_absolute_error(y_te, y_pred_te),
        'Train MSE': mean_squared_error(y_tr, y_pred_tr),
        'Test MSE': mean_squared_error(y_te, y_pred_te),
        'Train RMSE': np.sqrt(mean_squared_error(y_tr, y_pred_tr)),
        'Test RMSE': np.sqrt(mean_squared_error(y_te, y_pred_te)),
        'Train R2': r2_tr,
        'Test R2': r2_te,
        'Train Adj R2': adj_r2_tr,
        'Test Adj R2': adj_r2_te
    }

results = []

# 1. Linear Regression (All 20 features)
lr_all = LinearRegression()
results.append(evaluate_model(lr_all, X_train_scaled, X_test_scaled, y_train, y_test, "OLS Linear Regression (20 Features)"))

# 2. Linear Regression (Top 14 features)
lr_14 = LinearRegression()
results.append(evaluate_model(lr_14, X_train_scaled[top_14_features], X_test_scaled[top_14_features], y_train, y_test, "OLS Linear Regression (14 Features)"))

# 3. Ridge Regression (All 20 features)
ridge_all = Ridge(alpha=1.0)
results.append(evaluate_model(ridge_all, X_train_scaled, X_test_scaled, y_train, y_test, "Ridge Regression (20 Features, alpha=1.0)"))

# 4. Lasso Regression (All 20 features)
lasso_all = Lasso(alpha=0.0001, random_state=42)
results.append(evaluate_model(lasso_all, X_train_scaled, X_test_scaled, y_train, y_test, "Lasso Regression (20 Features, alpha=0.0001)"))

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, 'model_evaluation_metrics.csv'), index=False)
print("Model Comparison Metrics:")
print(results_df[['Model', 'Test MAE', 'Test RMSE', 'Test R2', 'Test Adj R2']])

# === [6] Statistical Regression Diagnostics (Five Assumptions) ===
print("=== [6] OLS Regression Diagnostics with statsmodels ===")
X_train_sm = sm.add_constant(X_train_scaled)
X_test_sm = sm.add_constant(X_test_scaled)
ols_sm = sm.OLS(y_train, X_train_sm).fit()

# Save OLS summary text
with open(os.path.join(output_dir, 'ols_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(str(ols_sm.summary()))

# 1. Multicollinearity (VIF)
vif_data = pd.DataFrame()
vif_data["Feature"] = feature_cols
vif_data["VIF"] = [variance_inflation_factor(X_train_scaled.values, i) for i in range(len(feature_cols))]
vif_data = vif_data.sort_values(by="VIF", ascending=False)
vif_data.to_csv(os.path.join(output_dir, 'vif_multicollinearity.csv'), index=False)
print("VIF Statistics:")
print(vif_data.head())

# 2. Residuals & Fitted Values
residuals = ols_sm.resid
fitted_values = ols_sm.fittedvalues

# 3. Durbin-Watson Test
dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson statistic: {dw_stat:.4f}")

# 4. Breusch-Pagan Test for Homoscedasticity
bp_test = het_breuschpagan(residuals, X_train_sm)
print(f"Breusch-Pagan LM Stat: {bp_test[0]:.4f}, p-value: {bp_test[1]:.4e}")

# Diagnostic Plots (4-panel figure)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel 1: Residuals vs Fitted (Homoscedasticity & Linearity)
sns.scatterplot(x=fitted_values, y=residuals, alpha=0.2, ax=axes[0, 0], color='#1f77b4')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axes[0, 0].set_title('1. Residuals vs Fitted (Linearity & Homoscedasticity)', fontweight='bold')
axes[0, 0].set_xlabel('Fitted Values (Predicted FloodProbability)')
axes[0, 0].set_ylabel('Residuals')

# Panel 2: Q-Q Plot (Normality)
sm.qqplot(residuals, line='45', fit=True, ax=axes[0, 1])
axes[0, 1].set_title('2. Normal Q-Q Plot (Residual Normality)', fontweight='bold')
axes[0, 1].get_lines()[0].set_color('#1f77b4')
axes[0, 1].get_lines()[0].set_alpha(0.3)
axes[0, 1].get_lines()[1].set_color('red')

# Panel 3: Scale-Location Plot (Square root of standardized residuals)
standardized_residuals = np.sqrt(np.abs(residuals / np.std(residuals)))
sns.scatterplot(x=fitted_values, y=standardized_residuals, alpha=0.2, ax=axes[1, 0], color='#2ca02c')
axes[1, 0].set_title('3. Scale-Location Plot (Spread-Location)', fontweight='bold')
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel(r'$\sqrt{|Standardized\ Residuals|}$')

# Panel 4: Residuals Distribution
sns.histplot(residuals, kde=True, ax=axes[1, 1], color='#9467bd', bins=40)
axes[1, 1].set_title('4. Residuals Distribution Histogram & KDE', fontweight='bold')
axes[1, 1].set_xlabel('Residual')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'five_assumptions_diagnostic_plots.png'), dpi=300)
plt.close()

# === [7] Prediction Plot with 95% Confidence & Prediction Intervals ===
print("=== [7] Plotting Prediction with 95% Confidence & Prediction Intervals ===")
# We take a test subset of 300 points sorted by predicted value for clean visualization
y_test_pred = ols_sm.get_prediction(X_test_sm)
pred_summary = y_test_pred.summary_frame(alpha=0.05) # 95% CI & PI

test_eval_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': pred_summary['mean'].values,
    'CI_Lower': pred_summary['mean_ci_lower'].values,
    'CI_Upper': pred_summary['mean_ci_upper'].values,
    'PI_Lower': pred_summary['obs_ci_lower'].values,
    'PI_Upper': pred_summary['obs_ci_upper'].values
}).sort_values(by='Predicted').reset_index(drop=True)

# 1. Sorted Curve Plot with Shaded Intervals (Sample 250 points)
sample_idx = np.linspace(0, len(test_eval_df)-1, 250, dtype=int)
sample_df = test_eval_df.iloc[sample_idx].reset_index(drop=True)

plt.figure(figsize=(12, 7))
x_axis = np.arange(len(sample_df))

# 95% Prediction Interval (shaded light blue)
plt.fill_between(x_axis, sample_df['PI_Lower'], sample_df['PI_Upper'], color='#e0f2fe', alpha=0.8, label='95% Prediction Interval (PI)')

# 95% Confidence Interval (shaded medium blue)
plt.fill_between(x_axis, sample_df['CI_Lower'], sample_df['CI_Upper'], color='#38bdf8', alpha=0.6, label='95% Confidence Interval (CI)')

# Prediction Line
plt.plot(x_axis, sample_df['Predicted'], color='#0369a1', linewidth=2.5, label='OLS Fitted Line ($\hat{y}$)')

# Actual Data Points
plt.scatter(x_axis, sample_df['Actual'], color='#f97316', alpha=0.65, s=20, label='Actual Test Observation ($y$)', zorder=5)

plt.title('Multiple Linear Regression: Actual vs Predicted with 95% CI & PI', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Sample Index (Sorted by Predicted Flood Probability)', fontsize=12)
plt.ylabel('Flood Probability', fontsize=12)
plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'prediction_with_confidence_intervals.png'), dpi=300)
plt.close()

# 2. Scatter Plot: Actual vs Predicted with 45-degree reference line
plt.figure(figsize=(9, 8))
plt.scatter(test_eval_df['Predicted'], test_eval_df['Actual'], alpha=0.25, color='#0284c7', s=25, label='Test Observations')
lims = [
    min(test_eval_df['Predicted'].min(), test_eval_df['Actual'].min()) - 0.02,
    max(test_eval_df['Predicted'].max(), test_eval_df['Actual'].max()) + 0.02
]
plt.plot(lims, lims, color='#ef4444', linestyle='--', linewidth=2, label='Perfect Prediction Line ($y = \hat{y}$)')
plt.xlim(lims)
plt.ylim(lims)
plt.title(f'Actual vs Predicted Scatter Plot ($R^2 = {results[0]["Test R2"]:.4f}$)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted FloodProbability ($\hat{y}$)', fontsize=12)
plt.ylabel('Actual FloodProbability ($y$)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted_scatter.png'), dpi=300)
plt.close()

# === [8] Model Coefficients Ranking ===
coefs = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': ols_sm.params[1:].values,
    'Std_Error': ols_sm.bse[1:].values,
    'p_value': ols_sm.pvalues[1:].values
}).sort_values(by='Coefficient', ascending=False)
coefs.to_csv(os.path.join(output_dir, 'ols_coefficients.csv'), index=False)

plt.figure(figsize=(10, 7))
sns.barplot(x='Coefficient', y='Feature', data=coefs, hue='Feature', palette='mako', legend=False)
plt.title(r'Multiple Linear Regression Standardized Coefficients ($\beta_j$)', fontsize=14, fontweight='bold')
plt.xlabel('Standardized Coefficient Impact on FloodProbability')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ols_coefficients_ranking.png'), dpi=300)
plt.close()

print("All analyses completed successfully! Artifacts generated in:", output_dir)
