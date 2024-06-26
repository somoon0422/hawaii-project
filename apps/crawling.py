import logging
import pandas as pd
import streamlit as st
from china.crawl_china import crawl_china
from china.crawl_vietnam import crawl_vietnam
from china.crawl_korea import crawl_korea
from china.crawl_usa import crawl_usa
from china.crawl_taiwan import crawl_taiwan
from china.crawl_readable import crawl_readable

def app():
    st.title('Crawling App')

    countries = ['China', 'Vietnam', 'Korea', 'USA', 'Taiwan', 'The readable']
    selected_country = st.selectbox("Select a country for crawling", countries)

    if st.button('Crawling Start!'):
        st.write(f"Starting crawling for {selected_country}...")
        if selected_country == 'China':
            try:
                crawl_data = crawl_china()
                if crawl_data is not None:
                    st.success('Crawling completed!')
                    st.write(crawl_data)
                    st.markdown(get_table_download_link(crawl_data, 'China_News.csv'), unsafe_allow_html=True)
                else:
                    st.error('Error occurred during crawling. Please check the logs for details.')
            except Exception as e:
                logging.error(f"Error occurred during crawling for China: {str(e)}")
                st.error('Error occurred during crawling. Please check the logs for details.')
        elif selected_country == 'Vietnam':
            crawl_vietnam()
        elif selected_country == 'Korea':
            crawl_korea()
        elif selected_country == 'USA':
            crawl_usa()
        elif selected_country == 'Taiwan':
            crawl_taiwan()
        elif selected_country == 'The readable':
            crawl_readable()

    if st.button('Merge All Data!'):
        st.write("Merging data from all countries...")
        try:
            merge_data()
            st.success("Data merging completed!")
        except Exception as e:
            logging.error(f"Error occurred during data merging: {str(e)}")
            st.error(f"Error occurred during data merging: {str(e)}")

def merge_data():
    # Merge all CSV files
    eco = pd.read_csv('China_Economy.csv')
    soc = pd.read_csv('China_Society.csv')
    wor = pd.read_csv('China_World.csv')
    cul = pd.read_csv('China_Culture.csv')
    spo = pd.read_csv('China_Sports.csv')
    mil = pd.read_csv('China_Military.csv')

    df = pd.concat([eco, soc, wor, cul, spo, mil])
    df.to_csv('China_News.csv', index=False)

def get_table_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename} CSV File</a>'
    return href

if __name__ == '__main__':
    app()
