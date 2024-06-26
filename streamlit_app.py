import streamlit as st
from apps import analysis
from apps import crawling

def main():
    st.title('🎡Welcome To Hawaii Project!')

    st.sidebar.title('Menu')
    page = st.sidebar.selectbox('선택하세요', ['Main Page','Visualization', 'Crawling Page' ])

    if page == 'Main Page':
        st.write(
            """
            #### This is Data crawling and analytics applications

            This application is a simple tool for data crawling and analysis.

            Please select the desired page from the sidebar on the left.
            """
        )




    elif page == 'Visualization':
        analysis.app()

    elif page == 'Crawling Page':
        crawling.app()


if __name__ == '__main__':
    main()
