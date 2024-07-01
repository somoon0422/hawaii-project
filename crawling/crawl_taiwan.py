import pandas as pd

def crawl_taiwan():
    try:
        # Taiwan 크롤링 코드 작성
        crawl_data = pd.DataFrame({'Country': ['Taiwan'], 'Data': ['Sample data']})  # 예시 데이터
        # 여기서 데이터베이스에 저장하는 코드를 추가할 수 있음
        save_to_database(crawl_data)
    except Exception as e:
        print(f"Error occurred during Taiwan crawling: {str(e)}")

def save_to_database(data):
    # 여기에 데이터베이스에 저장하는 코드를 추가하세요
    # 예: data.to_sql('table_name', con=your_database_connection, if_exists='append')
    print("Saving data to database...")
    print(data)  # 예시로 데이터 출력
