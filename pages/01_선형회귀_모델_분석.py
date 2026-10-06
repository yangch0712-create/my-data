import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="선형회귀 모델 분석 - 서울 기온 예측",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 연평균 기온 선형회귀 모델 분석 및 비교")
st.write("서울의 연평균 기온 데이터를 활용하여 **전체 데이터 모델**, **최근 50년 학습 모델(1956~2005)**, **최근 100년 학습 모델(1906~2005)**을 구축하고 최근 20년(2006~2025) 테스트 데이터에 대한 예측 성능을 비교합니다.")

@st.cache_data
def load_yearly_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"
    try:
        df = pd.read_csv(url, encoding='cp949')
    except Exception:
        df = pd.read_csv(url, encoding='utf-8')
    
    df.columns = df.columns.str.strip()
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    avg_col = [c for c in df.columns if '평균' in c][0]
    df[avg_col] = pd.to_numeric(df[avg_col], errors='coerce')
    
    yearly_df = df.groupby('연도').agg(
        연평균기온=(avg_col, 'mean'),
        관측일수=(avg_col, 'count')
    ).reset_index()
    
    yearly_df = yearly_df[yearly_df['관측일수'] >= 300].copy()
    return yearly_df

try:
    yearly_df = load_yearly_data()
    
    # -------------------------------------------------------------
    # 1) 전체 데이터 모델 학습 및 평가
    # -------------------------------------------------------------
    X_full = yearly_df[['연도']]
    y_full = yearly_df['연평균기온']
    
    model_full = LinearRegression()
    model_full.fit(X_full, y_full)
    pred_full = model_full.predict(X_full)
    
    mae_full = mean_absolute_error(y_full, pred_full)
    mse_full = mean_squared_error(y_full, pred_full)
    r2_full = r2_score(y_full, pred_full)
    slope_full = model_full.coef_[0]
    
    # -------------------------------------------------------------
    # 2) 훈련/테스트 데이터 분할 및 50년 vs 100년 비교 모델 학습
    # -------------------------------------------------------------
    # 공통 테스트 데이터 (최근 20년: 2006 ~ 2025)
    test_df = yearly_df[(yearly_df['연도'] >= 2006) & (yearly_df['연도'] <= 2025)]
    X_test = test_df[['연도']]
    y_test = test_df['연평균기온']
    
    # [모델 A] 최근 50년 학습 (1956 ~ 2005)
    train_50 = yearly_df[(yearly_df['연도'] >= 1956) & (yearly_df['연도'] <= 2005)]
    X_train_50 = train_50[['연도']]
    y_train_50 = train_50['연평균기온']
    
    model_50 = LinearRegression()
    model_50.fit(X_train_50, y_train_50)
    pred_test_50 = model_50.predict(X_test)
    
    mae_50 = mean_absolute_error(y_test, pred_test_50)
    mse_50 = mean_squared_error(y_test, pred_test_50)
    r2_50 = r2_score(y_test, pred_test_50)
    slope_50 = model_50.coef_[0]
    
    # [모델 B] 최근 100년 학습 (1906 ~ 2005)
    train_100 = yearly_df[(yearly_df['연도'] >= 1906) & (yearly_df['연도'] <= 2005)]
    X_train_100 = train_100[['연도']]
    y_train_100 = train_100['연평균기온']
    
    model_100 = LinearRegression()
    model_100.fit(X_train_100, y_train_100)
    pred_test_100 = model_100.predict(X_test)
    
    mae_100 = mean_absolute_error(y_test, pred_test_100)
    mse_100 = mean_squared_error(y_test, pred_test_100)
    r2_100 = r2_score(y_test, pred_test_100)
    slope_100 = model_100.coef_[0]
    
    # -------------------------------------------------------------
    # 화면 구성
    # -------------------------------------------------------------
    # 섹션 1: 전체 데이터 모델 평가
    st.subheader("1. 전체 데이터(전체 기간) 선형회귀 모델 평가")
    st.write(f"전체 관측 기간({int(yearly_df['연도'].min())}년 ~ {int(yearly_df['연도'].max())}년) 데이터 전체에 적합한 모델 지표입니다.")
    
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    f_col1.metric("기울기 (10년당 상승폭)", f"{slope_full * 10:+.3f} ℃")
    f_col2.metric("MAE (평균 절대 오차)", f"{mae_full:.3f} ℃")
    f_col3.metric("MSE (평균 제곱 오차)", f"{mse_full:.3f} ℃")
    f_col4.metric("R² (결정계수)", f"{r2_full:.3f}")
    
    st.divider()
    
    # 섹션 2: 50년 vs 100년 학습 모델 비교
    st.subheader("2. 학습 기간별 모델 비교 (공통 테스트 데이터: 2006년~2025년)")
    st.write("최근 20년간의 실제 기온을 예측할 때 과거 학습 기간(50년 vs 100년)에 따른 회귀선 기울기 및 예측 성능 평가 표입니다.")
    
    metrics_data = {
        '비교 항목': ['학습 기간 (Train Years)', '테스트 기간 (Test Years)', '기울기 (10년당 기온 변화량)', 'MAE (평균 절대 오차)', 'MSE (평균 제곱 오차)', 'R² (결정계수)'],
        '최근 50년 학습 모델': ['1956년 ~ 2005년', '2006년 ~ 2025년', f"{slope_50 * 10:+.3f} ℃", f"{mae_50:.3f} ℃", f"{mse_50:.3f} ℃", f"{r2_50:.3f}"],
        '최근 100년 학습 모델': ['1906년 ~ 2005년', '2006년 ~ 2025년', f"{slope_100 * 10:+.3f} ℃", f"{mae_100:.3f} ℃", f"{mse_100:.3f} ℃", f"{r2_100:.3f}"]
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df)
    
    # 섹션 3: 회귀선 시각화 비교
    st.subheader("📈 학습 기간별 회귀선 및 테스트 데이터 시각화")
    
    fig = go.Figure()
    
    # 전체 관측점 (배경)
    fig.add_trace(go.Scatter(
        x=yearly_df['연도'], y=yearly_df['연평균기온'],
        mode='markers', name='과거 관측 데이터',
        marker=dict(color='gray', opacity=0.4, size=6)
    ))
    
    # 테스트 데이터 (2006~2025)
    fig.add_trace(go.Scatter(
        x=test_df['연도'], y=test_df['연평균기온'],
        mode='markers+lines', name='테스트 데이터 (2006~2025)',
        marker=dict(color='red', size=8),
        line=dict(color='red', width=2)
    ))
    
    # 50년 학습 회귀선
    years_plot = np.arange(1906, 2026).reshape(-1, 1)
    pred_50_line = model_50.predict(years_plot)
    fig.add_trace(go.Scatter(
        x=years_plot.flatten(), y=pred_50_line,
        mode='lines', name=f'50년 학습 회귀선 (기울기: +{slope_50*10:.2f}℃/10년)',
        line=dict(color='blue', width=3, dash='dash')
    ))
    
    # 100년 학습 회귀선
    pred_100_line = model_100.predict(years_plot)
    fig.add_trace(go.Scatter(
        x=years_plot.flatten(), y=pred_100_line,
        mode='lines', name=f'100년 학습 회귀선 (기울기: +{slope_100*10:.2f}℃/10년)',
        line=dict(color='green', width=3)
    ))
    
    fig.update_layout(
        title="서울 연평균 기온 회귀선 및 2006~2025년 예측 성능 비교",
        xaxis_title="연도",
        yaxis_title="연평균 기온 (℃)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 섹션 4: 인사이트
    st.subheader("💡 회귀선 기울기 및 성능 비교 분석 결과")
    st.markdown(f"""
    1. **회귀선 기울기 비교**:
       - **50년 학습 모델(1956~2005)**의 기울기는 **10년당 약 {slope_50*10:+.2f}℃ 상승**입니다.
       - **100년 학습 모델(1906~2005)**의 기울기는 **10년당 약 {slope_100*10:+.2f}℃ 상승**입니다.
       - **인사이트**: 최근 50년 모델의 기울기가 더 급격하며, 이는 후반부로 갈수록 산업화/도시화에 따른 지구 온난화 및 기온 상승 속도가 빨라졌음을 의미합니다.

    2. **테스트 데이터(2006~2025) 예측 성능 평가**:
       - **MAE / MSE 관점**: 최근 온난화 가속 경향을 더 잘 반영한 **50년 학습 모델**이 더 작은 오차(MAE: {mae_50:.3f}℃)를 기록하여 예측력이 우수합니다.
       - **100년 학습 모델의 한계**: 100년 전 과거 데이터를 포함하면 완만한 기울기가 형성되어, 최근 20년 동안 급격히 상승한 실제 기온을 과소평가(Underestimation)하게 됩니다.
    """)

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
