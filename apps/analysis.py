import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scattertext import CorpusFromPandas, produce_scattertext_explorer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from matplotlib import font_manager, rc
import plotly.graph_objects as go
from nltk.corpus import stopwords
import matplotlib
from textblob import TextBlob
import base64
import nltk
import re
import plotly.express as px
import networkx as nx

# Matplotlib에서 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'NanumGothic'

# NLTK 불용어 다운로드
nltk.download('stopwords')

def remove_stopwords(text, stop_words):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def parse_date(date_string):
    if pd.isna(date_string) or date_string == 0:
        return None
    
    # 날짜 형식을 지원하는 포맷들
    supported_formats = ['%Y%m%d', '%Y-%m-%d', '%Y/%m/%d', '%Y%m%d%H%M%S']
    parsed_date = None
    for fmt in supported_formats:
        try:
            parsed_date = pd.to_datetime(date_string, format=fmt)
            break
        except ValueError:
            continue
    return parsed_date

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # CSV 파일을 base64 인코딩하여 다운로드 링크 생성
    href = f'<a href="data:file/csv;base64,{b64}" download="news_data.csv">데이터 다운로드</a>'
    return href


def extract_year(date):
    if pd.isna(date):
        return None
    return date.year

def get_most_significant_word_per_year(data):
    significant_words = []
    for year in sorted(data['Year'].dropna().unique()):
        year_data = data[data['Year'] == year]
        vectorizer = CountVectorizer(stop_words='english')
        word_count = vectorizer.fit_transform(year_data['Article Body'])
        words = vectorizer.get_feature_names_out()
        word_count_sum = word_count.sum(axis=0).A1
        most_significant_index = word_count_sum.argmax()
        most_significant_word = words[most_significant_index]
        significant_words.append({'Year': year, 'Most Significant Word': most_significant_word})
    return pd.DataFrame(significant_words)



def app():
    st.title("뉴스 기사 데이터 분석 대시보드")

    # 데이터 업로드
    uploaded_file = st.file_uploader("뉴스 기사 데이터 파일을 업로드하세요 (CSV 형식)")

    if uploaded_file is not None:
        # 데이터 읽기
        df = pd.read_csv(uploaded_file)

        # 'Submission Date'와 'Last Edited Date' 열 중 하나가 있는지 확인
        date_columns = ['Submission Date', 'Last Edited Date']
        valid_date_column = None
        for col in date_columns:
            if col in df.columns:
                valid_date_column = col
                break

        if valid_date_column is None:
            st.error("Submission Date 또는 Last Edited Date 열이 필요합니다.")
            return

        # 날짜 컬럼 병합 및 포맷 변환
        try:
            df['Date'] = df[date_columns].apply(lambda row: parse_date(row[0]) if pd.notna(row[0]) and row[0] != 0 else parse_date(row[1]) if pd.notna(row[1]) and row[1] != 0 else None, axis=1)
            df = df.dropna(subset=['Date'])  # NaN 값이 있는 행 제거
            df['Date'] = df['Date'].dt.date  # datetime 객체에서 날짜만 추출
        except Exception as e:
            st.error(f"날짜 데이터를 처리하는 동안 오류가 발생했습니다: {str(e)}")
            return

         # 가장 예전 날짜와 가장 최신 날짜 가져오기
        min_date = df['Date'].min()
        max_date = df['Date'].max()

        # 날짜 필터 설정
        start_date = st.date_input("시작 날짜", min_value=min_date, value=min_date)
        end_date = st.date_input("종료 날짜", max_value=max_date, value=max_date)


        if start_date > end_date:
            st.error("종료 날짜는 시작 날짜 이후여야 합니다.")
            return

        # 날짜 필터 적용
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        df = df[mask]


        
        # 분석 유형 선택
        analysis_type = st.radio("분석 유형을 선택하세요", ["빠른분석", "심도분석"])

        # 사용자 입력 키워드
        keyword = st.text_input("키워드를 입력하세요")

        # NLTK 불용어 목록 가져오기 및 사용자 정의 불용어 추가
        stop_words = set(stopwords.words('english'))
        custom_stop_words = {'said', 'will', 'new'}
        stop_words = stop_words.union(custom_stop_words)

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

            # 감정 분석을 위한 'polarity' 열 추가
            df['polarity'] = df['Article Body'].apply(lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else None)

            # 선택된 국가의 데이터 필터링
            selected_country = st.selectbox("국가를 선택하세요", df['Country'].unique())
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

            # 불용어 제거
            country_data['text'] = country_data['text'].apply(lambda x: remove_stopwords(str(x), stop_words))

            # 감정 분석을 위한 데이터 분리
            X_train, X_test, y_train, y_test = train_test_split(country_data['text'], country_data['polarity'] > 0, test_size=0.2, random_state=0)

            # 분석 결과 출력
            st.subheader("감정 분류 결과")
            classifier = MultinomialNB()
            vectorizer = TfidfVectorizer(stop_words=stop_words)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            classifier.fit(X_train_vec, y_train)
            y_pred = classifier.predict(X_test_vec)
            st.write(classification_report(y_test, y_pred))

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

        # 불용어 제거
        country_data['text'] = country_data['text'].apply(lambda x: remove_stopwords(str(x), stop_words))

        # 분석 결과 선택
        st.header("분석 결과")
        options = ["빈도 분석", "감정 분석", "워드 클라우드", "토픽 모델링", "감정 분류", "주제 분류", "TF-IDF 분석", "단어 빈도 분포", "네트워크 그래프"]
        selected_options = st.multiselect("원하는 분석을 선택하세요", options)

        if "빈도 분석" in selected_options:
            st.subheader("빈도 분석")
            vectorizer = CountVectorizer(stop_words=list(stop_words))
            word_count = vectorizer.fit_transform(country_data['text'])
            word_count_sum = word_count.sum(axis=0)
            words_freq = [(word, word_count_sum[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
            words_df = pd.DataFrame(words_freq, columns=['Word', 'Frequency'])
            st.write(words_df.head(10))

        if "감정 분석" in selected_options:
            st.subheader("감정 분석")
            country_data['polarity'] = country_data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
            st.write(country_data[['Date', 'polarity']].describe())
            plt.hist(country_data['polarity'], bins=20)
            st.pyplot(plt)

        if "워드 클라우드" in selected_options:
            st.subheader("워드 클라우드")
            text = ' '.join(country_data['text'])
            wordcloud = WordCloud(stopwords=stop_words, background_color='white').generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

        if "토픽 모델링" in selected_options:
            st.subheader("토픽 모델링")
            vectorizer = CountVectorizer(stop_words=list(stop_words))
            dtm = vectorizer.fit_transform(country_data['text'])
            lda = LatentDirichletAllocation(n_components=5, random_state=0)
            lda.fit(dtm)
            for i, topic in enumerate(lda.components_):
                st.write(f"토픽 {i}:")
                st.write([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])

        if "감정 분류" in selected_options:
            st.subheader("감정 분류")

            # 감정 분석을 위한 'polarity' 열 추가
            country_data['polarity'] = country_data['text'].apply(lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else None)

            # NaN 값 제거
            country_data = country_data.dropna(subset=['polarity'])

            # 감정 분류 레이블 추가
            country_data['label'] = (country_data['polarity'] > 0).astype(int)

            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(country_data['text'], country_data['label'], test_size=0.2, random_state=0)

            # TF-IDF 벡터화
            vectorizer = TfidfVectorizer(stop_words=list(stop_words))
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            # 모델 학습 및 예측
            clf = MultinomialNB()
            clf.fit(X_train_vec, y_train)
            y_pred = clf.predict(X_test_vec)

            # 분류 성능 지표 출력
            st.write("### 감정 분류 성능 지표")
            st.text("Precision, Recall, F1-score를 포함한 분류 성능 지표")
            
            report = classification_report(y_test, y_pred, target_names=["부정적", "긍정적"], output_dict=True)
            report_df = pd.DataFrame(report).transpose()

            st.dataframe(report_df)

            st.write("### 감정 분류 결과 요약")
            st.text(f"전체 샘플 수: {len(y_test)}")
            st.text(f"정확도(Accuracy): {report['accuracy']:.2f}")
            st.text(f"부정적(0) 감정 기사 수: {sum(y_test == 0)}")
            st.text(f"긍정적(1) 감정 기사 수: {sum(y_test == 1)}")

            st.text("감정 분류 성능 요약:")
            st.text(f"부정적 Precision: {report['부정적']['precision']:.2f}")
            st.text(f"긍정적 Precision: {report['긍정적']['precision']:.2f}")
            st.text(f"부정적 Recall: {report['부정적']['recall']:.2f}")
            st.text(f"긍정적 Recall: {report['긍정적']['recall']:.2f}")
            st.text(f"부정적 F1-score: {report['부정적']['f1-score']:.2f}")
            st.text(f"긍정적 F1-score: {report['긍정적']['f1-score']:.2f}")


        if "주제 분류" in selected_options:
            st.subheader("주제 분류")

            # 데이터 전처리: Category 열에서 null 값이 있는 경우 제거
            country_data = country_data.dropna(subset=['Category'])

            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(country_data['text'], country_data['Category'], test_size=0.2, random_state=0)

            # TF-IDF 벡터화
            vectorizer = TfidfVectorizer(stop_words=list(stop_words))
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            # 모델 학습 및 예측
            classifier = MultinomialNB()
            classifier.fit(X_train_vec, y_train)
            y_pred = classifier.predict(X_test_vec)

            # 분류 성능 지표 출력
            st.write("### 주제 분류 성능 지표")
            st.text("Precision, Recall, F1-score를 포함한 분류 성능 지표")
            
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()

            st.dataframe(report_df)

            st.write("### 주제 분류 결과 요약")
            st.text(f"전체 샘플 수: {len(y_test)}")
            st.text(f"정확도(Accuracy): {report['accuracy']:.2f}")

            # 각 카테고리 별 샘플 수 계산
            category_counts = y_test.value_counts()

            st.write("#### 각 카테고리별 기사 수")
            category_count_table = pd.DataFrame(category_counts).reset_index()
            category_count_table.columns = ['Category', '기사 수']
            st.table(category_count_table)

            st.write("#### 주제 분류 성능 요약")
            performance_summary = {
                "Precision": report['weighted avg']['precision'],
                "Recall": report['weighted avg']['recall'],
                "F1-score": report['weighted avg']['f1-score']
            }
            st.table(pd.DataFrame(performance_summary, index=["주제 분류 성능"]).T)


        if "TF-IDF 분석" in selected_options:
            st.subheader("TF-IDF 분석")
            vectorizer = TfidfVectorizer(stop_words=list(stop_words))
            tfidf_matrix = vectorizer.fit_transform(country_data['text'])
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())


            st.write("### 각 문서의 주요 단어")
            for i, text in enumerate(country_data['text'].head(10)):
                expander = st.expander(f"문서 {i+1}: {text[:50]}...")
                with expander:
                    st.markdown("**주요 단어**:")
                    indices = np.argsort(tfidf_matrix[i].toarray().flatten())[::-1][:5]
                    for idx in indices:
                        st.info(f"- **{vectorizer.get_feature_names_out()[idx]}**: {tfidf_matrix[i, idx]:.3f}")
                    # 전체 내용 보기 버튼 추가
                    if len(text) > 50:
                        st.write("전체 내용 보기:")
                        st.text(text)



        if "단어 빈도 분포" in selected_options:
            st.subheader("단어 빈도 분포")
            
            # CountVectorizer를 사용하여 단어 빈도 계산
            vectorizer = CountVectorizer(stop_words=list(stop_words))
            word_count = vectorizer.fit_transform(country_data['text'])
            word_count_sum = word_count.sum(axis=0)
            words_freq = [(word, word_count_sum[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
            words_df = pd.DataFrame(words_freq, columns=['Word', 'Frequency'])
            
            # Plotly 막대 차트 생성
            fig = go.Figure(go.Bar(
                x=words_df['Word'].head(10),
                y=words_df['Frequency'].head(10),
                text=words_df['Frequency'].head(10),
                textposition='outside',  # 빈도수를 막대 바깥에 표시
                marker_color=words_df['Frequency'].head(10),  # 빈도수에 따라 다른 색상 설정
                hovertemplate="단어: %{x}<br>빈도수: %{y}",
            ))
            
            fig.update_layout(
                title='상위 10개 단어의 빈도 분포',
                xaxis_title='단어',
                yaxis_title='빈도수',
                xaxis={'categoryorder': 'total descending'}  # 단어 빈도 기준 내림차순 정렬
            )

            # Plotly 그래프를 Streamlit에 렌더링
            st.plotly_chart(fig)


        if "네트워크 그래프" in selected_options:
            # 네트워크 그래프
            st.subheader("네트워크 그래프")
            vectorizer = CountVectorizer(stop_words=list(stop_words))
            word_count = vectorizer.fit_transform(country_data['text'])
            words = vectorizer.get_feature_names_out()
            word_freq = dict(zip(words, word_count.sum(axis=0).A1))

            # Co-occurrence matrix 계산
            co_occurrence_matrix = (word_count.T * word_count)
            co_occurrence_matrix.setdiag(0)
            co_occurrence_df = pd.DataFrame(co_occurrence_matrix.toarray(), index=words, columns=words)

            # 그래프 생성
            G = nx.from_pandas_adjacency(co_occurrence_df)
            pos = nx.spring_layout(G, k=0.1)
            edge_trace = []
            node_trace = go.Scatter(
                x=[], y=[], text=[], mode='markers+text',
                textposition="bottom center",
                hoverinfo='text', marker=dict(
                    showscale=True, colorscale='YlGnBu', color=[], size=10,
                    colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right')))

            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace.append(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None], line=dict(width=0.5, color='#888'),
                    hoverinfo='none', mode='lines'))

            for node in G.nodes():
                x, y = pos[node]
                node_trace['x'] += (x,)
                node_trace['y'] += (y,)
                node_trace['marker']['color'] += (G.degree(node),)
                node_info = f"{node} (# of connections: {G.degree(node)})"
                node_trace['text'] += (node_info,)

            fig = go.Figure(data=edge_trace + [node_trace], layout=go.Layout(
                title='단어 네트워크 그래프', titlefont_size=16, showlegend=False, hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(text="단어 간의 연관성을 보여주는 네트워크 그래프입니다.", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)],
                xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False)))

            st.plotly_chart(fig)



        # CSV 다운로드 링크 제공
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)

if __name__ == "__main__":
    app()
