import streamlit as st
import pandas as pd


st.title("🏝️ Hawaii Project- News Crawling")

st.write(
    "안녕하세요👋"
    "하와이 프로젝트에 오신 것을 환영 합니다."
)

st.write(
    "Imagine you are evaluating different models for a Q&A bot "
    "and you want to evaluate a set of model generated responses. "
    "You have collected some user data. "
    "Here is a sample question and response set."
)

data = pd.read_csv('Total_News.csv')

df = pd.DataFrame(data)

st.write(df)

st.write(
    "Now I want to evaluate the responses from my model. "
    "One way to achieve this is to use the very powerful `st.data_editor` feature. "
    "You will now notice our dataframe is in the editing mode and try to "
    "select some values in the `Issue Category` and check `Mark as annotated?` once finished 👇"
)

# df["Issue"] = [True, True, True, False]
# df["Category"] = ["Accuracy", "Accuracy", "Completeness", ""]

# new_df = st.data_editor(
#     df,
#     column_config={
#         "Questions": st.column_config.TextColumn(width="medium", disabled=True),
#         "Answers": st.column_config.TextColumn(width="medium", disabled=True),
#         "Issue": st.column_config.CheckboxColumn("Mark as annotated?", default=False),
#         "Category": st.column_config.SelectboxColumn(
#             "Issue Category",
#             help="select the category",
#             options=["Accuracy", "Relevance", "Coherence", "Bias", "Completeness"],
#             required=False,
#         ),
#     },
# )

# st.write(
#     "You will notice that we changed our dataframe and added new data. "
#     "Now it is time to visualize what we have annotated!"
# )

# st.divider()

# st.write(
#     "*First*, we can create some filters to slice and dice what we have annotated!"
# )

# col1, col2 = st.columns([1, 1])
# with col1:
#     issue_filter = st.selectbox("Issues or Non-issues", options=new_df.Issue.unique())
# with col2:
#     category_filter = st.selectbox(
#         "Choose a category",
#         options=new_df[new_df["Issue"] == issue_filter].Category.unique(),
#     )

# st.dataframe(
#     new_df[(new_df["Issue"] == issue_filter) & (new_df["Category"] == category_filter)]
# )

# st.markdown("")
# st.write(
#     "*Next*, we can visualize our data quickly using `st.metrics` and `st.bar_plot`"
# )

# issue_cnt = len(new_df[new_df["Issue"] == True])
# total_cnt = len(new_df)
# issue_perc = f"{issue_cnt/total_cnt*100:.0f}%"

# col1, col2 = st.columns([1, 1])
# with col1:
#     st.metric("Number of responses", issue_cnt)
# with col2:
#     st.metric("Annotation Progress", issue_perc)

# df_plot = new_df[new_df["Category"] != ""].Category.value_counts().reset_index()

# st.bar_chart(df_plot, x="Category", y="count")

st.write(
    "Here we are at the end of getting started with streamlit! Happy Streamlit-ing! :balloon:"
)

