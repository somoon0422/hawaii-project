import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns

# CSV 파일 읽기
data = pd.read_csv('Total_News.csv')
df = pd.DataFrame(data)

# 사이드바에 페이지 선택 상자 추가
page = st.sidebar.selectbox(
    "페이지 선택",
    ["뉴스 데이터", "Data visualization"]
)

# 뉴스 데이터 페이지
if page == "뉴스 데이터":
    # 대시보드 제목 및 설명
    st.title("🏝️ 하와이 프로젝트 - 뉴스 크롤링")

    st.write(
        "안녕하세요👋 "
        "하와이 프로젝트에 오신 것을 환영합니다."
    )

    st.write(
        "아래는 크롤링에 사용된 data 입니다." 
        "크롤링이 진행되면 여기서 확인이 가능합니다."
        "이제 원하는 분석과 시각화를 확인 하세요."
    )

    # 데이터프레임 표시
    st.write(df)

# Data visualization 페이지
elif page == "Data visualization":
    st.title("📊 Data visualization")

    st.write(
        "여기에서 다양한 데이터 시각화를 확인할 수 있습니다."
    )

    # 예시: 워드 클라우드 시각화
    st.subheader("Word Cloud of Text Data")
    text_column = st.selectbox("텍스트 데이터 컬럼 선택", df.columns)
    text_data = df[text_column].dropna().astype(str).values
    text = " ".join(text_data)
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # 예시: 특정 단어의 빈도를 막대 그래프로 시각화
    st.subheader("Top 10 Frequent Words")
    words = text.split()
    word_counts = Counter(words)
    common_words = word_counts.most_common(10)
    
    word_df = pd.DataFrame(common_words, columns=['word', 'count'])
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x='count', y='word', data=word_df, palette='viridis')
    st.pyplot(plt)
