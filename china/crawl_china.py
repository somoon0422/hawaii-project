from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import logging
import pandas as pd
import re
import time
from datetime import datetime


# 카테고리별 URL 설정
categories = {
    'politics': 'http://en.people.cn/90785/index{}.html', 
    'economy': 'http://en.people.cn/business/index{}.html',
    'society': 'http://en.people.cn/90882/index{}.html',
    'world': 'http://en.people.cn/90777/index{}.html',
    'culture': 'http://en.people.cn/90782/index{}.html',
    'sports': 'http://en.people.cn/90779/index{}.html',
    'military': 'http://en.people.cn/90786/index{}.html'
}

def crawl_articles(category):
    Press = []
    Reporter = []
    Category = []
    SubmissionDate = [] 
    SubmissionTime = []
    UpdataDate = []
    UpdataTime = []
    titles = []
    articles = []
    Size = []
    urls = []

    url_format = categories[category]

    # 정규식 패턴
    time_pattern = re.compile(r'(\d{2}:\d{2}), (\w+ \d{2}, \d{4})')

    # 웹드라이버 설정
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    for i in range(1, 11):  # 페이지 무한대 설정 가능
        page = url_format.format(i)
        driver.get(page)
        
        try:
            for j in range(1, 21):
                # 안정화
                time.sleep(3)
                # 기사 페이지
                headline = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, f'/html/body/div/div[5]/div[1]/ul/li[{j}]/a'))
                )
                # 큰 기사 링크로 이동
                headline_href = headline.get_attribute('href')
                
                # 새 창 열기
                driver.execute_script("window.open(arguments[0]);", headline_href)
                
                # 새 창으로 전환
                driver.switch_to.window(driver.window_handles[1])

                # 탭 개수 확인 및 닫기
                if len(driver.window_handles) > 3:
                    driver.close()
                    driver.switch_to.window(driver.window_handles[1])
                
                # Press
                Press.append('People.cn')
                
                # reporter
                try: 
                    reporter = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, 'editor'))
                    )
                    # 기자 이름 추출
                    reporter_text = reporter.text
                    if '(' in reporter_text and ')' in reporter_text:
                        names = reporter_text.split('(')[1].split(')')[0]
                        names = names.replace('Web editor:', '').strip()
                        reporter_names = [name.strip() for name in names.split(',')]
                        reporter_result = ', '.join(reporter_names)
                    else:
                        reporter_result = ''
                    Reporter.append(reporter_result)
                    
                except Exception as e:
                    logging.error(e)
                    print('ERROR : Crawling : get reporter :', e)
                    Reporter.append(None)
                    
                # Category  
                Category.append(category.capitalize())
                
                # Submission Date
                try:
                    submission_time_element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, '//div[@class="origin cf"]/span'))
                    )
                    submission_time_text = submission_time_element.text
                    
                    match = time_pattern.search(submission_time_text)
                    
                    if match:
                        time_str = match.group(1)
                        date_str = match.group(2)
                        date = datetime.strptime(date_str, '%B %d, %Y').strftime('%Y%m%d')
                        
                        SubmissionDate.append(int(date))
                        SubmissionTime.append(int(time_str.replace(':', '')))
                    else:
                        SubmissionDate.append(None)
                        SubmissionTime.append(None)
                except Exception as e :
                    logging.error(e)
                    print('ERROR : Crawling : get submission date / time :', e)
                    SubmissionDate.append(None)
                    SubmissionTime.append(None)
                    
                # Last Edited Date
                UpdataDate.append(None)
                
                # Last Edited Time
                UpdataTime.append(None)
                
                # titles
                try:
                    title = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, '/html/body/div/div[3]/h1'))
                    )
                    titles.append(title.text)
                except Exception as e:
                    print('ERROR : Crawling : get title :', e)
                    titles.append(None)
                
                # articles
                try:
                    article_box = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, '//div[@class="w860 d2txtCon cf"]'))
                    )
                    paragraphs = article_box.find_elements(By.TAG_NAME, 'p')
                    article = ''
                    for paragraph in paragraphs:
                        article += paragraph.text + '\n'
                    articles.append(article)
                except Exception as e:
                    print('ERROR : Crawling : get articles :', e)
                    articles.append(None)
                
                # Size
                try:
                    Size.append(len(article))
                except Exception as e:
                    print('ERROR : Crawling : get size :', e)
                    articles.append(None)
                
                # url
                urls.append(headline_href)
                
                print(f'{j}번째 기사 크롤링 완료', end='\r')
                
                # 새 창 닫기
                driver.close()
                
                # 원래 창으로 돌아가기
                driver.switch_to.window(driver.window_handles[0])
                
        except Exception as e:
            logging.error(e)
            print('ERROR : Crawling : ',e)
            # 다음 단계로 넘어감 
            continue
        
        print(f'{i}번째 페이지 크롤링 완료')

    # 데이터 프레임 생성
    df = pd.DataFrame({
        'Press': Press,
        'Reporter': Reporter,
        'Category': Category,
        'Submission Date': SubmissionDate,
        'Submission Time': SubmissionTime,
        'Last Edited Date': UpdataDate,
        'Last Edited Time': UpdataTime,
        'Article Title': titles,
        'Article Body': articles,
        'Size': Size,
        'url': urls
    })    
    
    return df

def crawl_china():

    dfs = []    
    for category in categories.keys():
        try :
            category_data = crawl_articles(category)
            category_data.to_csv(f'China_{category.capitalize()}.csv', index=False)
            dfs.append(category_data)
            print(f'{category} 카테고리 크롤링 완료')
        except Exception as e:
            print(f"Error occurred during China crawling: {str(e)}")

    # 6개 csv 파일을 하나로 합치기
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv('China_News.csv', index=False)
    print('전체 카테고리 크롤링 완료')

    return combined_df




if __name__ == '__main__' :
    crawl_china()