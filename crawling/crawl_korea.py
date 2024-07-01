# 뉴스 기사 크롤링
import requests
import re
import pandas as pd
import csv
import os
import time
import random
import sys
import json
import logging
from datetime import datetime
from bs4 import BeautifulSoup as soup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException , WebDriverException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import logging
from mtranslate import translate
import time

def initialize_webdriver():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    return driver

def initialize_logging():
    logging.basicConfig(level=logging.ERROR, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

def extract_article_data(driver, url, category):
    driver.get(url)
    time.sleep(1)
    
    try:
        # 기자 이름
        journalist = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="newsWriterCarousel01"]/div/div/div/div/div/a')))
        reporter = journalist.text
    except Exception as e:
        logging.error(f"Error extracting journalist name: {str(e)}")
        reporter = 'unknown'
        
    try:
        # 송고 날짜와 시간
        submission_element = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, 'newsUpdateTime01'))
        )
        data_published_time = submission_element.get_attribute('data-published-time')
        submission_date = data_published_time[:8]
        submission_time = data_published_time[8:12]
    except Exception as e:
        logging.error(f"Error extracting submission date/time: {str(e)}")
        submission_date = 'unknown'
        submission_time = 'unknown'
        
    try:
        # 기사 제목
        article_title = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="articleWrap"]/div[1]/header/h1')))
        title = article_title.text
    except Exception as e:
        logging.error(f"Error extracting article title: {str(e)}")
        title = 'unknown'
        
    try:
        # 기사 내용
        article_body = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'container')))
        paragraphs = article_body.find_elements(By.TAG_NAME, 'p')
        body = ' '.join([p.text for p in paragraphs])
    except Exception as e:
        logging.error(f"Error extracting article body: {str(e)}")
        body = 'unknown'
        
    return {
        'Country': 'South Korea',
        'Press': 'Yonhap News Agency',
        'Reporter': reporter,
        'Category': category,
        'Submission Date': submission_date,
        'Submission Time': submission_time,
        'Last Edited Date': 'null',
        'Last Edited Time': 'null',
        'Article Title': title,
        'Article Body': body,
        'Size': len(body),
        'url': url
    }

def scrape_category(driver, base_url, category, page_count, article_count):
    articles_data = []
    for i in range(1, page_count + 1):
        driver.get(f'{base_url}/all/{i}')
        for j in range(1, article_count + 1):
            try:
                element_xpath = f'//*[@id="container"]/div/div/div[2]/section/div[1]/ul/li[{j}]/div/div[2]/a'
                element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, element_xpath))
                )
                article_url = element.get_attribute('href')
                article_data = extract_article_data(driver, article_url, category)
                articles_data.append(article_data)
                driver.back()
            except Exception as e:
                logging.error(f"Error reading element at index {j} on page {i}: {str(e)}")
    return articles_data

def translate_text(text):
    try:
        translation = translate(text, 'en')
        print(f"원본 텍스트: {text}")
        print(f"번역된 텍스트: {translation}")
        print("="*50)
        return translation
    except Exception as e:
        print(f"번역 중 오류 발생: {e}")
        return None  # 번역 중 오류 발생시 'none' 으로 대체

def crawl_korea():
    driver = initialize_webdriver()
    initialize_logging()

    all_articles = []
    
    # 정치면 크롤링
    politics_base_url = 'https://www.yna.co.kr/politics'
    politics_articles = scrape_category(driver, politics_base_url, 'politics', 12, 27)
    all_articles.extend(politics_articles)
    print("정치면이 완료 되었습니다.")
    
    # 경제면 크롤링
    economy_base_url = 'https://www.yna.co.kr/economy'
    economy_articles = scrape_category(driver, economy_base_url, 'economy', 21, 14)
    all_articles.extend(economy_articles)
    print("경제면이 완료 되었습니다.")

    # 사회면 크롤링
    society_base_url = 'https://www.yna.co.kr/society'
    society_articles = scrape_category(driver, society_base_url, 'society', 21, 28)
    all_articles.extend(society_articles)
    print("사회면이 완료 되었습니다.")

    # 세계면 크롤링
    world_base_url = 'https://www.yna.co.kr/international'
    world_articles = scrape_category(driver, world_base_url, 'world', 20, 27)
    all_articles.extend(world_articles)
    print("세계면이 완료 되었습니다.")
    
    # 산업면 크롤링
    industry_base_url = 'https://www.yna.co.kr/industry'
    industry_articles = scrape_category(driver, industry_base_url, 'industry', 21, 27)
    all_articles.extend(industry_articles)
    print("산업면이 완료 되었습니다.")
    
    # 문화면 크롤링
    culture_base_url = 'https://www.yna.co.kr/culture'
    culture_articles = scrape_category(driver, culture_base_url, 'culture', 17, 27)
    all_articles.extend(culture_articles)
    print("문화면이 완료 되었습니다.")
    
    # 스포츠면 크롤링
    sports_base_url = 'https://www.yna.co.kr/sports'
    sports_articles = scrape_category(driver, sports_base_url, 'sports', 10, 27)
    all_articles.extend(sports_articles)
    print("스포츠면이 완료 되었습니다.")

    # 데이터 프레임 생성
    df = pd.DataFrame(all_articles)
    df.to_csv('Yonhap_articles.csv', index=False, encoding='utf-8-sig')
    print("Yonhap_articles.csv 파일이 생성되었습니다.")
    
    driver.quit()

    # 이전에 번역된 데이터가 저장된 파일명
    translated_file_name = 'SouthKorea_articles.csv'

    try:
        # 이전에 번역된 데이터 로드
        translated_df = pd.read_csv(translated_file_name)
        print("이전에 번역된 데이터 불러오기 완료")
    except FileNotFoundError:
        print("이전에 번역된 데이터가 없습니다. 새로운 번역을 시작합니다.")
        translated_df = None

    # CSV 파일 읽기
    df = pd.read_csv('Yonhap_articles.csv')

    # 번역할 컬럼 선택
    columns_to_translate = ['Reporter', 'Article Title', 'Article Body']

    # 이전에 번역된 데이터가 있는 경우 해당 부분을 제외하고 번역 진행
    if translated_df is not None:
        for column in columns_to_translate:
            df[column] = df[column].where(df[column].notnull(), translated_df[column])
        start_index = translated_df.index[-1] + 1
    else:
        start_index = 0

    # 나머지 데이터에 대해 번역 진행
    for i in range(start_index, len(df)):
        for column in columns_to_translate:
            df.at[i, column] = translate_text(str(df.at[i, column]))

    # # 컬럼 순서 재정렬
    # columns_order = ['Country', 'Press', 'Reporter', 'Category', 'Submission Date', 'Submission Time', 'Last Edited Date', 'Last Edited Time', 'Article Title', 'Article Body', 'Size', 'url']
    # df = df[columns_order]

    # 번역된 데이터프레임을 새로운 CSV 파일로 저장
    df.to_csv(translated_file_name, index=False, encoding='utf-8-sig')
    print(f"{translated_file_name} 파일이 생성되었습니다.")

if __name__ == "__main__":
    crawl_korea()
