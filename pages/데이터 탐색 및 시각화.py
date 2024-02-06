import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.markdown("# 데이터 탐색 및 시각화")
st.sidebar.markdown("# 데이터 탐색 및 시각화")

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


# st.title('This is a title')
# st.title('_Streamlit_ is :blue[cool] :sunglasses:')

st.header('This is a header with a divider', divider='rainbow')
st.text('헌법재판소 재판관은 정당에 가입하거나 정치에 관여할 수 없다. 대통령·국무총리·국무위원·행정각부의 장·헌법재판소 재판관·법관·중앙선거관리위원회 위원·감사원장·감사위원 기타 법률이 정한 공무원이 그 직무집행에 있어서 헌법이나 법률을 위배한 때에는 국회는 탄핵의 소추를 의결할 수 있다.정당의 목적이나 활동이 민주적 기본질서에 위배될 때에는 정부는 헌법재판소에 그 해산을 제소할 수 있고, 정당은 헌법재판소의 심판에 의하여 해산된다.')
st.header('_Streamlit_ is :blue[cool] :sunglasses:')

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk의 불용어 다운로드
import nltk
nltk.download('stopwords')

# 불용어 설정 (rating_overall에 따라서 어떤 불용어를 사용할지는 상황에 따라 조정해야 합니다.)

additional_stop_words = ['hotel', 'stay', 'would', 'could', 'even', 'one','stay','stayed','room','rooms']

# 불용어 설정 (rating_overall에 따라서 어떤 불용어를 사용할지는 상황에 따라 조정해야 합니다.)
stop_words = set(stopwords.words('english') + additional_stop_words)

# ratings_overall 값에 따른 Word Cloud 생성 함수
def generate_wordcloud_for_rating(rating, text_column):
    # 해당 rating에 해당하는 리뷰들 선택
    subset = df[df['ratings_overall'] == rating]
    
    # 해당 리뷰들의 text 컬럼을 합치기
    text = ' '.join(subset[text_column].astype(str))
    
    # 불용어 제거 및 토큰화
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    
    # Word Cloud 생성
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_tokens))
    
    # 시각화
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Overall Rating {rating}')
    plt.show()

# ratings_overall 값에 따라 Word Cloud 생성
for rating in range(1, 6):
    generate_wordcloud_for_rating(rating, 'text')


# 레이아웃 조정
plt.tight_layout()
plt.show()

st.pyplot(fig2)
