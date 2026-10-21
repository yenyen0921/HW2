# 🌊 AIoT HW2: 多元線性回歸 (Multiple Linear Regression) 與 CRISP-DM 實作分析

> **學生學號**：7115064153  
> **指導教授**：huanchen1107 (煥哥)  
> **作業架構**：本作業為 **HW2 正式作業**（延伸自 HW1 氣象資料工程實作），繳交形式比照 DIC2 規範（**GitHub 儲存庫 + GitHub Pages 線上網頁**）。  
> **嚴格禁令遵循**：**絕無使用 50 Startups，絕無使用波士頓房價 (巴森豪斯)**。  
> **Kaggle 資料集**：  
> - 🌊 **主題目 (Problem 1)**：[Kaggle: Playground Series S4E5 - Flood Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s4e5)（**20 個連續數值特徵**，符合 >13 特徵要求）  
> - 🏡 **副題目 (Problem 2)**：[Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)（**18 個精選數值特徵**，符合 >13 特徵要求）  

---

## 🌐 快速體驗連結 (Quick Access)

- 🚀 **線上 Web 互動系統 (GitHub Pages)**：[https://yenyen0921.github.io/HW2/](https://yenyen0921.github.io/HW2/)  
- 📂 **GitHub 專案儲存庫**：[https://github.com/yenyen0921/HW2](https://github.com/yenyen0921/HW2)  
- 🤖 **ChatGPT 線上公開分享連結**：[點此檢視完整對話歷程](https://chatgpt.com/share/6ad5d4cd-6388-83e8-bcbd-3923960e8502)  

---

## 🏆 作業要求與加分項目達成對照表

| 項目 | 規範標準 | HW2 達成狀態 | 得分效益 |
| :--- | :--- | :--- | :---: |
| **資料集特徵數** | 必須選擇 Kaggle 特徵數 > 13 之公開資料集 | 選擇 Kaggle 洪水預測（20 特徵）與 Ames 房價（18 特徵） | **基本規範** |
| **嚴格禁止題材** | 禁用 50 Startups 與 波士頓房價 (巴森豪斯) | 全數排除，完全無 50 Startups 與 波士頓房價 | **基本規範** |
| **加分條件 1** | 下拉視窗呈現兩種 Problem（皆需 > 13 特徵） | 網頁頂部 Dropdown 嚴格呈現 Problem 1 (20f) 與 Problem 2 (18f) | **加總分 +1 分** |
| **加分條件 2** | 可任意輸入 Dataset (支援 raw CSV 網址 / 本地上傳) | 內建瀏覽器端 AutoML 工作流，支援任意 CSV 即時訓練回歸與繪圖 | **加總分 +2 分** |
| **AI 協同要求** | ChatGPT 對話歷程 PDF + 公開網址；NotebookLM 摘要 > 100 字 | 產出 `7115064153_GPT_Chat_History.pdf` (含分享網址)；NotebookLM 220 字摘要 | **必備滿分** |
| **Optional 成果** | 對話內容、資訊圖表、報告書 | 完整交付物：學術報告書 PDF、200 DPI 資訊圖表海報 PNG/PDF、完整 Jupyter Notebook | **全數交付** |

---

## 📁 繳交檔案清單 (Deliverables)

1. **主程式**：
   - 📓 [`7115064153_hw2.ipynb`](7115064153_hw2.ipynb)（完整交互式 Jupyter Notebook，包含全流程圖表與輸出）
   - 🐍 [`7115064153_hw2.py`](7115064153_hw2.py)（獨立 Python 腳本，可於終端一鍵執行）
2. **作業報告檔**：
   - 📄 [`7115064153_hw2_report.pdf`](7115064153_hw2_report.pdf)（高清學術排版 PDF 報告，含所有圖表、CRISP-DM 流程、假設檢定表與對比）
   - 📝 [`7115064153_hw2_report.md`](7115064153_hw2_report.md)（Markdown 格式報告）
3. **AI 協同紀錄**：
   - 🤖 [`7115064153_GPT_Chat_History.pdf`](7115064153_GPT_Chat_History.pdf)（ChatGPT 對話歷程匯出 PDF）
   - 🌐 [ChatGPT 線上公開分享連結](https://chatgpt.com/share/6ad5d4cd-6388-83e8-bcbd-3923960e8502)
4. **輸出產物資料夾**：
   - 📊 `artifacts/`（包含 7 張高解析度分析圖表與 6 份評估 CSV / 診斷報表）
5. **最終繳交壓縮檔**：
   - 📦 `7115064153_hw2.zip`

---

## 📊 核心分析成果摘要

### 1. 模型效能評估 (Test Set, n=6,000)
| 模型 | 特徵數 | Test MAE | Test RMSE | Test $R^2$ | Test Adjusted $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **OLS Linear Regression** | **20** | **0.01594** | **0.01998** | **0.8445** | **0.8440** |
| **OLS (Top 14 Features)** | 14 | 0.02753 | 0.03417 | 0.5454 | 0.5444 |
| **Ridge Regression (alpha=1.0)** | 20 | 0.01594 | 0.01998 | 0.8445 | 0.8440 |
| **Lasso Regression (alpha=0.0001)** | 20 | 0.01599 | 0.02000 | 0.8443 | 0.8438 |

### 2. 五大多元線性回歸假設檢定結果
- **線性性 (Linearity)**：殘差對稱分佈於 0 水平軸，無非線性彎曲。
- **殘差常態性 (Normality)**：Q-Q Plot 中心 $\pm 2$ 標準差緊密貼合 45 度線。
- **殘差同質變異 (Homoscedasticity)**：殘差方差均勻帶狀分佈，無漏斗形發散。
- **殘差獨立性 (Independence)**：**Durbin-Watson = 2.0064**（極度接近 2.0，完全無自相關）。
- **無多重共線性 (No Multicollinearity)**：**Max VIF = 1.0061**（遠低於 5.0 門檻，特徵完全獨立正交）。

### 3. 預測圖與 95% 信賴 / 預測區間
- **95% Confidence Interval (CI)**：精確量化總體均值預測範圍。
- **95% Prediction Interval (PI)**：成功涵蓋 95% 以上真實測試點位。

---

## 💡 AI 協同分析與 NotebookLM 研究摘要
- **GPT 提問脈絡**：確立資料集選題、80/20 Train/Test 防洩漏切分、5大特徵選取、五大回歸假設檢定及 95% CI/PI 繪圖。
- **NotebookLM 摘要 (220字)**：
  > 根據 Google NotebookLM 針對 Kaggle S4E5 競賽頂尖方案文獻研析：本資料集為合成表格數據，特徵具備高度對稱之線性相加特質。文獻指出「特徵工程」為核心關鍵，透過構造 20 項特徵之總和（`f_sum`）、標準差（`f_std`）等聚合統計量，搭配 LightGBM、XGBoost、CatBoost 與神經網路進行 Stacking 融合，能將測試集 $R^2$ 由線性基準的 0.845 提升至 0.869 以上。然而多元線性回歸具備透明因果解釋性與嚴謹統計推論基礎，為公共水利決策最佳核心模型。
