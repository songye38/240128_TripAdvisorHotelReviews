import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.markdown("# Page 2 ❄️")
st.sidebar.markdown("# Page 2 ❄️")

df = pd.read_csv("eda_ver_dataset.csv")

#st.dataframe(df)



# ratings_로 시작하는 모든 칼럼에 대한 히스토그램 그리기
ratings_columns = df.filter(regex='^ratings_')

# 서브플롯 설정
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15),sharey=True)

# 각 칼럼에 대한 히스토그램 그리기
for i, col in enumerate(ratings_columns.columns):
    sns.histplot(df[col], bins=[0,1, 2, 3, 4, 5, 6], kde=False, color='skyblue', edgecolor='black', ax=axes[i//3, i%3])
    axes[i//3, i%3].set_title(f'Distribution of {col}')
    axes[i//3, i%3].set_xlabel('Rating')
    axes[i//3, i%3].set_ylabel('Frequency')
    axes[i//3, i%3].set_xticks(range(6))
    axes[i//3, i%3].set_xticklabels(['0', '1', '2', '3', '4', '5'])

# 레이아웃 조정
plt.tight_layout()
plt.show()

st.pyplot(fig)

