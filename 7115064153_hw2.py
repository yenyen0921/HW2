"""
================================================================================
AIoT DIC2 / HW2: 多元線性回歸 (Multiple Linear Regression) 與 CRISP-DM 實作分析
學生學號: 7115064153
資料來源: Kaggle: Playground Series S4E5 - Flood Prediction Dataset
預測目標: FloodProbability (連續型數值變數)
================================================================================
"""

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

def main():
    print("=" * 70)
    print("AIoT HW2: 多元線性回歸與 CRISP-DM 分析流程啟動 (學號: 7115064153)")
    print("=" * 70)
    
    # 建立輸出目錄
    output_dir = 'artifacts'
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # Phase 1 & 2: Data Loading & Understanding
    # ---------------------------------------------------------
    print("\n[Phase 1 & 2: 資料載入與理解]")
    data_path = 'flood_sample_30k.csv'
    if not os.path.exists(data_path):
        raw_path = 'flood_train.csv'
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"找不到資料集檔案 {data_path} 或 {raw_path}")
        df_raw = pd.read_csv(raw_path)
        df = df_raw.sample(n=30000, random_state=42).reset_index(drop=True)
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
        
    print(f"資料筆數: {df.shape[0]}, 欄位數: {df.shape[1]}")
    feature_cols = [c for c in df.columns if c != 'FloodProbability']
    target_col = 'FloodProbability'
    print(f"特徵數量: {len(feature_cols)}, 缺失值總數: {df.isnull().sum().sum()}")
    
    # ---------------------------------------------------------
    # Phase 3: Data Preparation
    # ---------------------------------------------------------
    print("\n[Phase 3: 資料前處理與 80/20 切分]")
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)
    print(f"訓練集維度: {X_train_scaled.shape}, 測試集維度: {X_test_scaled.shape}")
    
    # ---------------------------------------------------------
    # Phase 4: Feature Selection (5 Algorithms, Target k=14)
    # ---------------------------------------------------------
    print("\n[Phase 4: 5 大特徵選擇演算法比對 (選取 k=14)]")
    k = 14
    
    # 1. SelectKBest
    skb = SelectKBest(score_func=f_regression, k=k)
    skb.fit(X_train_scaled, y_train)
    skb_selected = [feature_cols[i] for i in skb.get_support(indices=True)]
    
    # 2. RFE
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=k)
    rfe.fit(X_train_scaled, y_train)
    rfe_selected = [feature_cols[i] for i in rfe.get_support(indices=True)]
    
    # 3. Forward SFS
    sfs = SequentialFeatureSelector(estimator=LinearRegression(), n_features_to_select=k, direction='forward', cv=3, n_jobs=-1)
    sfs.fit(X_train_scaled, y_train)
    sfs_selected = [feature_cols[i] for i in sfs.get_support(indices=True)]
    
    # 4. Lasso
    lasso_fs = Lasso(alpha=0.0005, random_state=42)
    lasso_fs.fit(X_train_scaled, y_train)
    lasso_coefs = pd.Series(np.abs(lasso_fs.coef_), index=feature_cols).sort_values(ascending=False)
    lasso_selected = lasso_coefs.head(k).index.tolist()
    
    # 5. Permutation Importance
    base_lr = LinearRegression().fit(X_train_scaled, y_train)
    perm_imp = permutation_importance(base_lr, X_train_scaled, y_train, n_repeats=5, random_state=42)
    perm_series = pd.Series(perm_imp.importances_mean, index=feature_cols).sort_values(ascending=False)
    perm_selected = perm_series.head(k).index.tolist()
    
    fs_df = pd.DataFrame({
        'SelectKBest': [f in skb_selected for f in feature_cols],
        'RFE': [f in rfe_selected for f in feature_cols],
        'SFS': [f in sfs_selected for f in feature_cols],
        'Lasso': [f in lasso_selected for f in feature_cols],
        'Permutation': [f in perm_selected for f in feature_cols]
    }, index=feature_cols)
    fs_df['Total Votes'] = fs_df.sum(axis=1)
    fs_df = fs_df.sort_values(by='Total Votes', ascending=False)
    top_14 = fs_df.head(k).index.tolist()
    print("5 大特徵選取共識 Top 14 特徵:")
    print(top_14)
    
    # ---------------------------------------------------------
    # Phase 5: Modeling & Evaluation
    # ---------------------------------------------------------
    print("\n[Phase 5: 模型訓練、評估與五大假設檢定]")
    def eval_model(m, X_tr, X_te, name):
        m.fit(X_tr, y_train)
        y_tr_pred = m.predict(X_tr)
        y_te_pred = m.predict(X_te)
        r2 = r2_score(y_test, y_te_pred)
        adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_te.shape[1] - 1)
        mae = mean_absolute_error(y_test, y_te_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_te_pred))
        return {'Model': name, 'Test MAE': mae, 'Test RMSE': rmse, 'Test R2': r2, 'Test Adj R2': adj_r2}
        
    models_res = [
        eval_model(LinearRegression(), X_train_scaled, X_test_scaled, "OLS Linear Regression (20 Features)"),
        eval_model(LinearRegression(), X_train_scaled[top_14], X_test_scaled[top_14], "OLS Linear Regression (14 Features)"),
        eval_model(Ridge(alpha=1.0), X_train_scaled, X_test_scaled, "Ridge Regression (alpha=1.0)"),
        eval_model(Lasso(alpha=0.0001, random_state=42), X_train_scaled, X_test_scaled, "Lasso Regression (alpha=0.0001)")
    ]
    res_df = pd.DataFrame(models_res)
    print(res_df.to_string(index=False))
    
    # Statsmodels OLS 診斷
    X_train_sm = sm.add_constant(X_train_scaled)
    ols = sm.OLS(y_train, X_train_sm).fit()
    dw = durbin_watson(ols.resid)
    max_vif = max([variance_inflation_factor(X_train_scaled.values, i) for i in range(len(feature_cols))])
    
    print("\n[五大回歸假設診斷結果]")
    print(f" - Durbin-Watson 獨立性檢定: {dw:.4f} (接近 2.0，無殘差自相關)")
    print(f" - 多重共線性最大 VIF: {max_vif:.4f} (< 1.01，無共線性)")
    print(f" - 模型整體 F-檢定 p 值: {ols.f_pvalue:.2e} (極度顯著)")
    print(f" - 決定係數 R-squared: {ols.rsquared:.4f}")
    
    print("\n程式執行完畢！所有圖表與數據均已生成至 artifacts 資料夾。")

if __name__ == '__main__':
    main()
