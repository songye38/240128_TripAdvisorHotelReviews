import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


st.markdown("# text column ❄️")
st.sidebar.markdown("# text column ❄️")

# Assume df is loaded from a CSV file
df = pd.read_csv("eda_ver_dataset.csv")

# Define initial keywords (can be empty or a default list)
keywords = ['staff', 'room', 'location', 'service', 'breakfast', 'expensive']

# Get user input for keywords
user_keywords = st.multiselect("Select Keywords:", keywords)

# Update the keywords list
if user_keywords:
    keywords = user_keywords

# Lists to store ratings and corresponding ratios for each keyword
ratings = {keyword: [] for keyword in keywords}
ratios = {keyword: [] for keyword in keywords}

# Iterate over each rating
for rating in [1.0, 2.0, 3.0, 4.0, 5.0]:
    # Iterate over each keyword
    for keyword in keywords:
        # Count the number of rows where ratings_overall matches the current rating and the text contains the keyword
        positive_count = df[(df['ratings_overall'] == rating) & (df['text'].str.contains(keyword, case=False))].shape[0]

        # Count the total number of rows where ratings_overall matches the current rating
        total_count = df[df['ratings_overall'] == rating].shape[0]

        # Calculate the ratio
        positive_ratio = positive_count / total_count * 100 if total_count > 0 else 0

        # Append to lists
        ratings[keyword].append(rating)
        ratios[keyword].append(positive_ratio)

# Plotting the bar graph for each keyword
for keyword in keywords:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ratings[keyword], ratios[keyword], color='skyblue')
    ax.set_xlabel('Ratings')
    ax.set_ylabel(f"Ratio of '{keyword}' Mentions (%)")
    ax.set_title(f"Ratio of '{keyword}' mentions among different ratings")
    ax.set_ylim(0, 100)  # Set the y-axis limit to ensure a proper scale

    # Display the percentages on top of the bars
    for i, ratio in enumerate(ratios[keyword]):
        ax.text(ratings[keyword][i], ratio + 1, f'{ratio:.2f}%', ha='center', va='bottom')

    st.pyplot(fig)
