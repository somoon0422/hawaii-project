# 📊 Data evaluation template

A simple Streamlit app showing how to evaluate and annotate data, using dataframes
and charts. 

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hawaii-project-isui36gmipn.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```


### 폴더 구조 및 파일 배치

- apps/
   - analysis.py - 분석페이지
   - crawling.py - 크롤링 페이지
- china/
  - __init__.py
  - crawl_china.py
  - crawl_vietnam.py
  - crawl_korea.py
  - crawl_usa.py
  - crawl_taiwan.py
  - crawl_readable.py
- multiapp.py
- streamlit_app.py (main코드)
- requirements.txt
- README.md



