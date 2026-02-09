import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os


# デバッグ情報を表示
st.sidebar.title("🔍 フォントデバッグ情報")

# 1. フォントファイルの存在確認
font_paths = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:\\Windows\\Fonts\\msgothic.ttc",
]

st.sidebar.write("### フォントファイル確認")
for font_path in font_paths:
    if os.path.exists(font_path):
        st.sidebar.success(f"✅ {font_path}")
    else:
        st.sidebar.error(f"❌ {font_path}")

# 2. インストール済みフォント一覧
st.sidebar.write("### 利用可能なフォント")
available_fonts = [
    f.name
    for f in fm.fontManager.ttflist
    if "CJK" in f.name or "Noto" in f.name or "Gothic" in f.name
]
st.sidebar.write(available_fonts[:10])  # 最初の10個を表示


# 3. フォント設定を試行
def set_japanese_font():
    """日本語フォントを設定する"""
    # Noto Sans CJK を探す
    for font in fm.fontManager.ttflist:
        if "Noto Sans CJK" in font.name or "NotoSansCJK" in font.name:
            plt.rcParams["font.family"] = font.name
            plt.rcParams["axes.unicode_minus"] = False
            st.sidebar.success(f"✅ 使用フォント: {font.name}")
            return True

    # 見つからない場合は手動でパス指定
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                fm.fontManager.addfont(font_path)
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams["font.family"] = font_prop.get_name()
                plt.rcParams["axes.unicode_minus"] = False
                st.sidebar.success(f"✅ パスから読込: {font_path}")
                return True
            except Exception as e:
                st.sidebar.error(f"エラー: {e}")

    st.sidebar.warning("⚠️ 日本語フォントが見つかりません")
    return False


set_japanese_font()

# 現在の設定を表示
st.sidebar.write("### 現在のmatplotlib設定")
st.sidebar.write(f"font.family: {plt.rcParams['font.family']}")

st.markdown(
    """
    <style>
    /* ボタンのスタイル */
    div.stButton {
        display: flex;
        justify-content: center;
    }
    div.stButton > button:first-child {
        background-color: #9b59b6;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        border: none;
        font-weight: 500;
    }
    div.stButton > button:first-child:hover {
        background-color: #8e44ad;
        border: none;
    }
    
    /* selectboxの行間調整 */
    div[data-testid="stSelectbox"] {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* 要素間の間隔 */
    .element-container {
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams["font.family"] = ["MS Gothic", "DejaVu Sans"]
# csvファイルをDataFrameに格納した。
train_data = pd.read_csv("seven_train.csv")

# 欠損値を一括で補完する
train_data_filled = train_data.fillna(train_data.mean(numeric_only=True))

st.title("相関関係アプリ")

mapping1 = {
    "セブン-イレブン総店舗数": "Total_Stores_Seven_Eleven",
    "既存店売上成長率": "Existing_Store_Sales_Growth_Rate",
    "セブン-イレブン客数": "Customer_Count_Seven_Eleven_Million",
    "海外売上高（百万米ドル）": "International_Revenue_USD_Million",
    "日本のGDP成長率（%）": "GDP_Growth_Rate",
    "消費者信頼感指数": "Consumer_Confidence_Index",
    "デジタル投資額（百万円）": "Digital_Investment_Million_Yen",
    "顧客満足度スコア": "Customer_Satisfaction_Score",
}
selected_jp1 = st.selectbox("変数を選択してください", list(mapping1.keys()))

# 辞書アクセスは 変数の中身をキーとして使うので、"" は不要。
selected_data1 = mapping1[selected_jp1]

mapping2 = {
    "売上高（百万円）": "Revenue_Million_Yen",
    "営業利益（百万円）": "Operating_Income_Million_Yen",
    "純利益（百万円）": "Net_Income_Million_Yen",
    "総資産（百万円）": "Total_Assets_Million_Yen",
    "海外売上高（百万米ドル）": "International_Revenue_USD_Million",
}

selected_jp2 = st.selectbox(
    "相関関係をみたい対象を選択してください", list(mapping2.keys())
)

selected_data2 = mapping2[selected_jp2]

# ボタンを中央に配置
st.markdown('<div class="center-button">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    button_clicked = st.button("相関関係を見る")
st.markdown("</div>", unsafe_allow_html=True)

# 抜き出したデータを相関関数にかける
if button_clicked:
    correlation_stores_revenue, p_value = pearsonr(
        train_data_filled[selected_data1],
        train_data_filled[selected_data2],
    )
    if p_value < 0.05:
        if correlation_stores_revenue > 0:
            sei = "正"
            st.markdown(
                f"<h3 style='text-align: center; color: blue; font-size: 1.5rem;'>統計的に有意な{sei}の相関関係が認められます</h3>",
                unsafe_allow_html=True,
            )
        else:
            fu = "負"
            st.markdown(
                f"<h3 style='text-align: center; color: blue; font-size: 1.5rem;'>統計的に有意な{fu}の相関関係が認められます</h3>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<h3 style='text-align: center; color: #e74c3c; font-size: 1.5rem;'>統計的に有意な相関関係は認められません</h3>",
            unsafe_allow_html=True,
        )

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(
        train_data_filled[selected_data1],
        train_data_filled[selected_data2] / 100,
    )
    ax1.set_xlabel(selected_jp1)
    ax1.set_ylabel(selected_jp2)
    ax1.set_title(f"{selected_jp1}と{selected_jp2}の相関関係")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    st.write(f"店舗数と売上の相関係数: {correlation_stores_revenue:.4f}")
    st.write(f"p値: {p_value:.4f}")
