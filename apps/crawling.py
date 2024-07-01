import logging
import os
import pandas as pd
import streamlit as st
from crawling.crawl_china import crawl_china
from crawling.crawl_vietnam import crawl_vietnam
from crawling.crawl_korea import crawl_korea
from crawling.crawl_usa import crawl_usa
from crawling.crawl_taiwan import crawl_taiwan
from crawling.crawl_readable import crawl_readable
import base64

def app():
    st.title('Crawling App')

    countries = ['China', 'Vietnam', 'SouthKorea', 'USA', 'Taiwan', 'The readable']
    selected_country = st.selectbox("Select a country for crawling", countries)

    if st.button('Crawling Start!'):
        st.write(f"Starting crawling for {selected_country}...")
        if selected_country == 'China':
            try:
                crawl_data = crawl_china()
                if crawl_data is not None:
                    st.success('Crawling completed!')
                    st.write(crawl_data)
                    st.markdown(get_table_download_link(crawl_data, 'China_articles.csv'), unsafe_allow_html=True)
                else:
                    st.error('Error occurred during crawling. Please check the logs for details.')
            except Exception as e:
                logging.error(f"Error occurred during crawling for China: {str(e)}")
                st.error('Error occurred during crawling. Please check the logs for details.')
        
        elif selected_country == 'Vietnam':
            crawl_vietnam()
            # 나머지 국가에 대한 크롤링 함수 호출 추가 필요
        
        elif selected_country == 'South Korea':
            try:
                crawl_data = crawl_korea()
                if crawl_data is not None:
                    st.success('Crawling completed!')
                    st.write(crawl_data)
                    st.markdown(get_table_download_link(crawl_data, 'SouthKorea_articles.csv'), unsafe_allow_html=True)
                else:
                    st.error('Error occurred during crawling. Please check the logs for details.')
            except Exception as e:
                logging.error(f"Error occurred during crawling for SouthKorea: {str(e)}")
                st.error('Error occurred during crawling. Please check the logs for details.')
        
        elif selected_country == 'USA':
            crawl_usa()
            # 나머지 국가에 대한 크롤링 함수 호출 추가 필요
        
        elif selected_country == 'Taiwan':
            crawl_taiwan()
            # 나머지 국가에 대한 크롤링 함수 호출 추가 필요
        
        elif selected_country == 'The readable':
            crawl_readable()
            # 나머지 국가에 대한 크롤링 함수 호출 추가 필요

    if st.button('Merge All Data'):
        st.write("Merging data from all countries...")
        try:
            merged_filename = merge_data(countries)
            if merged_filename:
                st.success("Data merging completed!")
                st.write(f"Merged file: {merged_filename}")
                st.markdown(get_table_download_link(pd.read_csv(merged_filename), 'all_articles.csv'), unsafe_allow_html=True)
            else:
                st.error("No data files found for merging.")
        except Exception as e:
            logging.error(f"Error occurred during data merging: {str(e)}")
            st.error('Error occurred during data merging. Please check the logs for details.')


def get_table_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename} CSV File</a>'
    return href


def merge_data(countries):
    all_countries_data = []

    for country in countries:
        if country == 'Others':
            continue
        filename = f'{country}_articles.csv'
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            all_countries_data.append(df)

    if all_countries_data:
        merged_df = pd.concat(all_countries_data, ignore_index=True)
        merged_filename = 'all_articles.csv'
        merged_df.to_csv(merged_filename, index=False, encoding='utf-8-sig')
        return merged_filename
    else:
        return None






if __name__ == '__main__':
    app()