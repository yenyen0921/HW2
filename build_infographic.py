import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np

# Set high-DPI font styling
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def create_infographic():
    fig = plt.figure(figsize=(18, 24), dpi=200, facecolor='#0f172a')
    
    # Grid layout: Header, 3 rows of content, Footer
    gs = gridspec.GridSpec(5, 2, height_ratios=[1.6, 3.5, 4.2, 4.2, 1.0], hspace=0.28, wspace=0.20,
                           left=0.05, right=0.95, top=0.95, bottom=0.03)

    # ----------------------------------------------------
    # Header Banner
    # ----------------------------------------------------
    ax_head = fig.add_subplot(gs[0, :])
    ax_head.set_facecolor('#1e293b')
    ax_head.axis('off')
    
    # Outer border
    for spine in ax_head.spines.values():
        spine.set_visible(True)
        spine.set_color('#38bdf8')
        spine.set_linewidth(2)

    ax_head.text(0.5, 0.72, '🌊 AIoT HW2 多元線性回歸實作分析 ── 核心成果資訊圖表', 
                 ha='center', va='center', color='#38bdf8', fontsize=24, fontweight='bold')
    ax_head.text(0.5, 0.45, 'Kaggle S4E5: Flood Prediction Dataset (20 Features) & CRISP-DM 機器學習全流程', 
                 ha='center', va='center', color='#f8fafc', fontsize=15, fontweight='bold')
    
    meta_text = '學生學號：7115064153  |  指導教授：huanchen1107 (煥哥)  |  課程主題：多元線性回歸、5大特徵選取、五大假設檢定、95% CI/PI'
    ax_head.text(0.5, 0.18, meta_text, 
                 ha='center', va='center', color='#94a3b8', fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f172a', edgecolor='#334155'))

    # ----------------------------------------------------
    # Row 1 Left: CRISP-DM 6大階段核心指標卡
    # ----------------------------------------------------
    ax_crisp = fig.add_subplot(gs[1, 0])
    ax_crisp.set_facecolor('#1e293b')
    ax_crisp.set_xticks([])
    ax_crisp.set_yticks([])
    for spine in ax_crisp.spines.values():
        spine.set_color('#334155')

    ax_crisp.text(0.04, 0.92, '📌 1. CRISP-DM 六大階段標準架構', color='#38bdf8', fontsize=16, fontweight='bold')

    steps = [
        ('1. Business Understanding', '量化流域防汛風險，預測連續機率目標 FloodProbability (0.35~0.70)'),
        ('2. Data Understanding', '20 項環境/治理特徵，30,000 筆樣本，0 遺失值，特徵間高度正交獨立'),
        ('3. Data Preparation', '80/20 嚴格防洩漏切分 (24k/6k)，StandardScaler 標準化前處理'),
        ('4. Modeling', '5 大特徵篩選 (SelectKBest, RFE, SFS, Lasso, Permutation) & OLS 擬合'),
        ('5. Evaluation', '全特徵 OLS: R² = 0.8445, RMSE = 0.01998；五大回歸假設全數通過'),
        ('6. Deployment', 'Scikit-Learn Pipeline 部署，REST API 推論端點，GitHub Pages 互動系統')
    ]

    for idx, (title, desc) in enumerate(steps):
        y_pos = 0.77 - idx * 0.13
        ax_crisp.text(0.04, y_pos, f'Step {idx+1}  {title}', color='#34d399', fontsize=12, fontweight='bold')
        ax_crisp.text(0.04, y_pos - 0.05, desc, color='#cbd5e1', fontsize=10.5)

    # ----------------------------------------------------
    # Row 1 Right: 模型綜合效能雷達/對比指標
    # ----------------------------------------------------
    ax_models = fig.add_subplot(gs[1, 1])
    ax_models.set_facecolor('#1e293b')
    ax_models.set_xticks([])
    ax_models.set_yticks([])
    for spine in ax_models.spines.values():
        spine.set_color('#334155')

    ax_models.text(0.04, 0.92, '📊 2. 模型效能對比 (Test Set, n=6,000)', color='#38bdf8', fontsize=16, fontweight='bold')

    # Draw mini bar chart inside
    models = ['OLS (20 Feat.)', 'Ridge (alpha=1)', 'Lasso (alpha=1e-4)', 'OLS (Top 14 Feat.)']
    r2_vals = [0.8445, 0.8445, 0.8443, 0.5454]
    rmse_vals = [0.01998, 0.01998, 0.02000, 0.03417]
    colors = ['#38bdf8', '#818cf8', '#a78bfa', '#f59e0b']

    for i in range(4):
        y_bar = 0.72 - i * 0.17
        ax_models.text(0.04, y_bar + 0.04, models[i], color='#f8fafc', fontsize=11.5, fontweight='bold')
        ax_models.text(0.70, y_bar + 0.04, f'R² = {r2_vals[i]:.4f} | RMSE = {rmse_vals[i]:.5f}', 
                       color=colors[i], fontsize=11, fontweight='bold')
        # Background bar
        ax_models.barh(y_bar, 0.90, height=0.04, left=0.04, color='#334155', alpha=0.5)
        # Value bar
        ax_models.barh(y_bar, 0.90 * (r2_vals[i] / 1.0), height=0.04, left=0.04, color=colors[i], alpha=0.9)

    ax_models.text(0.04, 0.06, '💡 結論：20 項特徵具均勻線性可加性，全特徵 OLS 展現最高決定係數與最低測試集誤差。', 
                   color='#94a3b8', fontsize=10.5, style='italic')

    # ----------------------------------------------------
    # Row 2 Left: 5 大特徵選擇投票圖
    # ----------------------------------------------------
    ax_fs = fig.add_subplot(gs[2, 0])
    ax_fs.set_facecolor('#1e293b')
    ax_fs.axis('off')
    fs_img_path = 'c:/Users/lab52/.antigravity-ide/HW2/artifacts/feature_selection_votes.png'
    if os.path.exists(fs_img_path):
        img_fs = Image.open(fs_img_path)
        ax_fs.imshow(img_fs)
    ax_fs.set_title('🔍 3. 五大特徵選擇演算法共識投票 (Top 14 Features)', color='#38bdf8', fontsize=14, fontweight='bold', pad=10)

    # ----------------------------------------------------
    # Row 2 Right: 預測圖與 95% 信賴/預測區間 (CI/PI)
    # ----------------------------------------------------
    ax_cipi = fig.add_subplot(gs[2, 1])
    ax_cipi.set_facecolor('#1e293b')
    ax_cipi.axis('off')
    cipi_img_path = 'c:/Users/lab52/.antigravity-ide/HW2/artifacts/prediction_with_confidence_intervals.png'
    if os.path.exists(cipi_img_path):
        img_cipi = Image.open(cipi_img_path)
        ax_cipi.imshow(img_cipi)
    ax_cipi.set_title('📈 4. 預測圖與 95% 信賴區間 (CI) / 預測區間 (PI)', color='#38bdf8', fontsize=14, fontweight='bold', pad=10)

    # ----------------------------------------------------
    # Row 3 Left: 五大多元線性回歸基本假設檢定圖
    # ----------------------------------------------------
    ax_diag = fig.add_subplot(gs[3, 0])
    ax_diag.set_facecolor('#1e293b')
    ax_diag.axis('off')
    diag_img_path = 'c:/Users/lab52/.antigravity-ide/HW2/artifacts/five_assumptions_diagnostic_plots.png'
    if os.path.exists(diag_img_path):
        img_diag = Image.open(diag_img_path)
        ax_diag.imshow(img_diag)
    ax_diag.set_title('🔬 5. 五大回歸基本假設統計診斷 (五圖全數通過)', color='#38bdf8', fontsize=14, fontweight='bold', pad=10)

    # ----------------------------------------------------
    # Row 3 Right: 回歸假設數值摘要與 NotebookLM 文獻研析
    # ----------------------------------------------------
    ax_assump = fig.add_subplot(gs[3, 1])
    ax_assump.set_facecolor('#1e293b')
    ax_assump.set_xticks([])
    ax_assump.set_yticks([])
    for spine in ax_assump.spines.values():
        spine.set_color('#334155')

    ax_assump.text(0.04, 0.94, '📋 6. 回歸假設檢定指標 & AI 文獻研析', color='#38bdf8', fontsize=15, fontweight='bold')

    assumptions = [
        ('1. 線性性 (Linearity)', '殘差均值 = 0.0000，無任何非線性彎曲'),
        ('2. 殘差常態性 (Normality)', 'Q-Q 圖在 ±2 標準差內緊貼 45 度線，呈完美常態分佈'),
        ('3. 同質變異性 (Homoscedasticity)', 'Scale-Location 水平帶狀均勻分佈，無漏斗形發散'),
        ('4. 殘差獨立性 (Independence)', 'Durbin-Watson = 2.0064 (極度接近 2.0，完全無自相關)'),
        ('5. 無多重共線性 (Multicollinearity)', '最大 VIF = 1.0061 (遠低於門檻 5.0，特徵正交獨立)')
    ]

    for idx, (h, detail) in enumerate(assumptions):
        y_pos = 0.84 - idx * 0.075
        ax_assump.text(0.04, y_pos, f'✔ {h}：', color='#34d399', fontsize=11, fontweight='bold')
        ax_assump.text(0.04, y_pos - 0.032, detail, color='#cbd5e1', fontsize=10)

    # NotebookLM Section Box
    ax_assump.text(0.04, 0.38, '🤖 Google NotebookLM 同主題文獻研析摘要 (220字)', color='#a78bfa', fontsize=12, fontweight='bold')
    nb_text = (
        "「根據 Google NotebookLM 針對 Kaggle S4E5 競賽頂尖方案文獻研析：本資料集為合成表格數據，"
        "特徵具備高度對稱之線性相加特質。文獻指出『特徵工程』為核心關鍵，透過構造 20 項特徵之總和 (f_sum)、"
        "標準差 (f_std) 等聚合統計量，搭配 LightGBM、XGBoost、CatBoost 進行 Stacking 融合，能將 R² 由線性基準"
        "的 0.845 提升至 0.869 以上。然而多元線性回歸具備透明因果解釋性與嚴謹統計推論基礎，為公共水利防汛最佳核心模型。」"
    )
    ax_assump.text(0.04, 0.16, nb_text, color='#94a3b8', fontsize=9.5, wrap=True,
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='#0f172a', edgecolor='#6366f1', alpha=0.9))

    # ----------------------------------------------------
    # Footer Banner
    # ----------------------------------------------------
    ax_foot = fig.add_subplot(gs[4, :])
    ax_foot.set_facecolor('#1e293b')
    ax_foot.axis('off')
    for spine in ax_foot.spines.values():
        spine.set_color('#334155')

    foot_text = (
        '🌐 GitHub 儲存庫：https://github.com/yenyen0921/HW2  |  '
        '🚀 線上 Web 展示：https://yenyen0921.github.io/HW2/\n'
        '📄 報告書：7115064153_hw2_report.pdf  |  '
        '🤖 ChatGPT 對話歷程：https://chatgpt.com/share/6ad5d4cd-6388-83e8-bcbd-3923960e8502'
    )
    ax_foot.text(0.5, 0.5, foot_text, ha='center', va='center', color='#38bdf8', fontsize=11, fontweight='bold')

    # Save Infographic PNG & PDF
    out_png_hw2 = 'c:/Users/lab52/.antigravity-ide/HW2/7115064153_hw2_infographic.png'
    out_pdf_hw2 = 'c:/Users/lab52/.antigravity-ide/HW2/7115064153_hw2_infographic.pdf'

    plt.savefig(out_png_hw2, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.savefig(out_pdf_hw2, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    print(f"Infographic generated successfully at {out_png_hw2} and {out_pdf_hw2}!")

if __name__ == '__main__':
    create_infographic()
