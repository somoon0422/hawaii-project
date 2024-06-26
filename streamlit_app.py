import streamlit as st
from apps import analysis
from apps import crawling

def main():
    st.title('Welcome To Hawaii Project!')

    st.sidebar.title('Menu')
    page = st.sidebar.selectbox('선택하세요', ['Main Page','visualization', 'Crawling Page' ])

    if page == 'Main Page':
        st.write("""
            ## 데이터 크롤링 및 분석 애플리케이션
            이 애플리케이션은 데이터 크롤링과 분석을 위한 간단한 툴입니다. 
            왼쪽의 사이드바에서 원하는 페이지를 선택하세요.
        """)
    elif page == 'visualization':
        analysis.app()

    elif page == 'Crawling Page':
        crawling.app()


if __name__ == '__main__':
    main()
