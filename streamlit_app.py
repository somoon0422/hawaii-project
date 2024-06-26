import streamlit as st
from apps import crawling, analysis

def main():
    st.title('Main Page')

    option = st.sidebar.selectbox(
        'Select an app',
        ('Crawling App', 'Analysis Dashboard')
    )

    if option == 'Crawling App':
        crawling.app()
    elif option == 'Analysis Dashboard':
        analysis.app()

if __name__ == '__main__':
    main()
