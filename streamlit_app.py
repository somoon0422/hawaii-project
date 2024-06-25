# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# from collections import Counter
# import seaborn as sns

# # CSV 파일 읽기
# data = pd.read_csv('Total_News.csv')
# df = pd.DataFrame(data)

# # 사이드바에 페이지 선택 상자 추가
# page = st.sidebar.selectbox(
#     "페이지 선택",
#     ["뉴스 데이터", "Data visualization"]
# )

# # 뉴스 데이터 페이지
# if page == "뉴스 데이터":
#     # 대시보드 제목 및 설명
#     st.title("🏝️ 하와이 프로젝트 - 뉴스 크롤링")

#     st.write(
#         "안녕하세요👋 "
#         "하와이 프로젝트에 오신 것을 환영합니다."
#     )

#     st.write(
#         "아래는 크롤링에 사용된 data 입니다." 
#         "크롤링이 진행되면 여기서 확인이 가능합니다."
#         "이제 원하는 분석과 시각화를 확인 하세요."
#     )

#     # 데이터프레임 표시
#     st.write(df)

# # Data visualization 페이지
# elif page == "Data visualization":
#     st.title("📊 Data visualization")

#     st.write(
#         "여기에서 다양한 데이터 시각화를 확인할 수 있습니다."
#     )

#     # 예시: 워드 클라우드 시각화
#     st.subheader("Word Cloud of Text Data")
#     text_column = st.selectbox("텍스트 데이터 컬럼 선택", df.columns)
#     text_data = df[text_column].dropna().astype(str).values
#     text = " ".join(text_data)
    
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     st.pyplot(plt)

#     # 예시: 특정 단어의 빈도를 막대 그래프로 시각화
#     st.subheader("Top 10 Frequent Words")
#     words = text.split()
#     word_counts = Counter(words)
#     common_words = word_counts.most_common(10)
    
#     word_df = pd.DataFrame(common_words, columns=['word', 'count'])
    
#     plt.figure(figsize=(10, 5))
#     sns.barplot(x='count', y='word', data=word_df, palette='viridis')
#     st.pyplot(plt)


import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Pandas 설정 변경: 최대 열 수와 최대 행 수 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 페이지 제목
st.title("뉴스 기사 데이터 분석 대시보드")

# 데이터 업로드
uploaded_file = st.file_uploader("뉴스 기사 데이터 파일을 업로드하세요 (CSV 형식)")

if uploaded_file is not None:
    # 데이터 읽기
    df = pd.read_csv(uploaded_file)
    st.write("업로드된 데이터:")
    
    # 데이터 프레임을 스크롤 가능하게 표시
    st.dataframe(df, height=600, width=800)

    # 데이터 프레임의 열 이름 출력
    st.write("데이터 프레임의 열 이름:")
    st.write(df.columns)

    # 'Country', 'Article Title', 'Article Body' 열이 있는지 확인
    if 'Country' in df.columns and 'Article Title' in df.columns and 'Article Body' in df.columns:
        # 국가별로 데이터 분류
        countries = df['Country'].unique()
        selected_country = st.selectbox("국가를 선택하세요", countries)

        # 선택된 국가의 데이터 필터링
        country_data = df[df['Country'] == selected_country]

        # 데이터 요약
        st.header(f"{selected_country} 데이터 요약")
        num_articles = len(country_data)
        st.write(f"총 기사 수: {num_articles}")

        # 기사 제목과 본문 내용을 하나의 텍스트로 결합
        country_data['text'] = country_data['Article Title'] + " " + country_data['Article Body']

        # 빈도 분석
        st.header("빈도 분석")
        vectorizer = CountVectorizer(stop_words='english')
        word_count = vectorizer.fit_transform(country_data['text'])
        word_count_sum = word_count.sum(axis=0)
        words_freq = [(word, word_count_sum[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        st.write(words_freq[:10])

        # 감정 분석
        st.header("감정 분석")
        country_data['polarity'] = country_data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        st.write(country_data[['Article Title', 'polarity']].head())
        st.bar_chart(country_data['polarity'])

        # 워드 클라우드
        st.header("워드 클라우드")
        wordcloud = WordCloud(width=800, height=400, max_words=100).generate(' '.join(country_data['text']))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # 토픽 모델링
        st.header("토픽 모델링")
        lda = LatentDirichletAllocation(n_components=5, random_state=0)
        lda.fit(word_count)
        terms = vectorizer.get_feature_names_out()
        for idx, topic in enumerate(lda.components_):
            st.write(f"토픽 {idx+1}:")
            st.write(" ".join([terms[i] for i in topic.argsort()[:-11:-1]]))
    else:
        st.write("'Country', 'Article Title', 'Article Body' 열을 찾을 수 없습니다. 데이터 파일에 이 열들이 포함되어 있는지 확인해주세요.")
else:
    st.write("파일을 업로드해주세요.")
