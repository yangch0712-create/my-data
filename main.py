import streamlit as st
import pandas as pd
import plotly.express as px

# 페이지 기본 설정
st.set_page_config(
    page_title="서울 100년 기온 변화 분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 지난 100년간 연평균 기온 변화")
st.write("기상청 서울 관측 데이터(`seoul.csv`)를 활용하여 지난 100년간의 기온 상승 추이를 분석합니다.")

# 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"
    
    # 한글 인코딩 예외 처리
    try:
        df = pd.read_csv(url, encoding='cp949')
    except Exception:
        df = pd.read_csv(url, encoding='utf-8')
    
    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()
    
    # 날짜 처리 및 연도 추출
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    # 주요 기온 컬럼 매핑 (단위 포함 대응)
    avg_col = [c for c in df.columns if '평균' in c][0]
    min_col = [c for c in df.columns if '최저' in c][0]
    max_col = [c for c in df.columns if '최고' in c][0]
    
    # 수치형 데이터 변환 및 결측치 처리
    df[avg_col] = pd.to_numeric(df[avg_col], errors='coerce')
    df[min_col] = pd.to_numeric(df[min_col], errors='coerce')
    df[max_col] = pd.to_numeric(df[max_col], errors='coerce')
    
    # 연도별 집계
    yearly_df = df.groupby('연도').agg(
        연평균기온=(avg_col, 'mean'),
        연평균최저기온=(min_col, 'mean'),
        연평균최고기온=(max_col, 'mean'),
        관측일수=(avg_col, 'count')
    ).reset_index()
    
    # 관측 데이터가 충분한 연도만 필터링 (연간 300일 이상)
    yearly_df = yearly_df[yearly_df['관측일수'] >= 300].copy()
    
    # 10년 이동평균 계산
    yearly_df['10년이동평균'] = yearly_df['연평균기온'].rolling(window=10, min_periods=1).mean()
    
    return df, yearly_df

# 메인 실행 영역
try:
    with st.spinner("데이터를 불러오는 중입니다..."):
        raw_df, yearly_df = load_data()
    
    # 주요 요약 지표 (Metrics)
    min_year = int(yearly_df['연도'].min())
    max_year = int(yearly_df['연도'].max())
    first_avg = yearly_df.iloc[0]['연평균기온']
    last_avg = yearly_df.iloc[-1]['연평균기온']
    temp_diff = last_avg - first_avg
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("분석 기간", f"{min_year}년 ~ {max_year}년")
    col2.metric("관측 초기 연평균", f"{first_avg:.1f} ℃")
    col3.metric("최근 연평균", f"{last_avg:.1f} ℃")
    col4.metric("기온 변화량", f"{temp_diff:+.1f} ℃", delta=f"{temp_diff:.1f} ℃")
    
    st.divider()
    
    # 시각화 그래프 (Plotly 사용 - 한글 깨짐 방지 및 인터랙티브 호버 지원)
    st.subheader("📈 연도별 평균 기온 추이 및 10년 이동평균선")
    
    fig = px.line(
        yearly_df, 
        x='연도', 
        y='연평균기온', 
        title=f'서울 연평균 기온 변화 ({min_year} - {max_year})',
        labels={'연도': '연도', '연평균기온': '연평균 기온 (℃)'},
        markers=True
    )
    
    # 이동평균 추세선 추가
    fig.add_scatter(
        x=yearly_df['연도'], 
        y=yearly_df['10년이동평균'], 
        mode='lines', 
        name='10년 이동평균선',
        line=dict(color='orange', width=3, dash='dash')
    )
    
    fig.update_traces(hovertemplate='<b>%{x}년</b><br>연평균 기온: %{y:.2f}℃')
    fig.update_layout(
        hovermode="x unified",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True, title="기온 (℃)"),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 상세 데이터표 보기
    with st.expander("📊 연도별 요약 데이터 보기"):
        st.dataframe(
            yearly_df[['연도', '연평균기온', '연평균최저기온', '연평균최고기온']].style.format({
                '연평균기온': '{:.2f} ℃',
                '연평균최저기온': '{:.2f} ℃',
                '연평균최고기온': '{:.2f} ℃'
            }), 
            use_container_width=True
        )

except Exception as e:
    st.error(f"데이터 로드 및 처리 중 오류 발생: {e}")
