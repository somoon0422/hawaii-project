import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def app():
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
            # 분석 유형 선택
            analysis_type = st.radio("분석 유형을 선택하세요", ["빠른분석", "심도분석"])

            # 사용자 입력 키워드
            keyword = st.text_input("키워드를 입력하세요")

            if analysis_type == "빠른분석":
                # 빠른분석: Article Title 열에서 키워드 분석
                st.header("빠른분석 결과")
                if keyword:
                    df['Keyword Frequency'] = df['Article Title'].apply(lambda x: str(x).lower().count(keyword.lower()) if isinstance(x, str) else 0)
                    country_keyword_freq = df.groupby('Country')['Keyword Frequency'].sum()
                    st.bar_chart(country_keyword_freq)
            elif analysis_type == "심도분석":
                # 심도분석: Article Body 열에서 키워드 분석
                st.header("심도분석 결과")
                if keyword:
                    df['Keyword Frequency'] = df['Article Body'].apply(lambda x: str(x).lower().count(keyword.lower()) if isinstance(x, str) else 0)
                    country_keyword_freq = df.groupby('Country')['Keyword Frequency'].sum()
                    st.bar_chart(country_keyword_freq)

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

            # 빈 값 또는 NaN 값을 제거
            country_data = country_data.dropna(subset=['text'])
            country_data = country_data[country_data['text'].str.strip() != '']

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
