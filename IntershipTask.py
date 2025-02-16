#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import webbrowser 
import os


# In[2]:


# load the datasets
apps_df=pd.read_csv(r"C:\Users\Dell\Downloads\Play Store Data.csv")
reviews_df=pd.read_csv(r"C:\Users\Dell\Downloads\User Reviews.csv")


# In[3]:


apps_df.head()


# In[4]:


reviews_df.head()


# In[5]:


apps_df.tail()


# In[6]:


reviews_df.tail()


# In[7]:


# Filter Play Store data for "Health & Fitness" apps
health_fitness_apps = apps_df[apps_df["Category"] == "HEALTH_AND_FITNESS"]["App"].unique()


# In[8]:


# Filter 5-star reviews for apps in the "Health & Fitness" category
filtered_reviews = reviews_df[
    (reviews_df["App"].isin(health_fitness_apps)) & (reviews_df["Sentiment"] == "Positive")
]


# In[9]:


filtered_reviews.head()


# In[10]:


filtered_reviews.tail()


# In[11]:


# Combine all review texts into a single string
five_star_reviews = filtered_reviews[filtered_reviews['Sentiment'].str.lower() == 'positive']
reviews_text = ' '.join(five_star_reviews['Translated_Review'].dropna().tolist())


# In[12]:


from wordcloud import WordCloud
import numpy as np
# Generate the word cloud using the provided settings
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    stopwords="english"
).generate(reviews_text)  # Using sample text for now

# Convert word cloud into a dataframe for Plotly visualization
word_freq_dict = wordcloud.words_
word_freq_df = pd.DataFrame(word_freq_dict.items(), columns=["word", "count"])

# Generate random x and y positions for word placement
np.random.seed(42)
word_freq_df["x"] = np.random.uniform(0, 1, size=len(word_freq_df))
word_freq_df["y"] = np.random.uniform(0, 1, size=len(word_freq_df))

# Create a scatter plot to mimic a word cloud in Plotly
fig1 = px.scatter(
    word_freq_df,
    x="x",
    y="y",
    text="word",
    size="count",
    title="Word Cloud for 5-Star Reviews of Health & Fitness Apps",
    color="count",
    color_continuous_scale=px.colors.sequential.Blues,
    width=600,
    height=400
  )

# Customize layout to resemble a word cloud
fig1.update_traces(textfont_size=word_freq_df["count"] / max(word_freq_df["count"]) * 40, textposition="middle center")
fig1.update_layout(
    plot_bgcolor="black",
    paper_bgcolor="black",
    font_color="white",
    title_font={"size": 16},
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    margin=dict(l=10, r=10, t=30, b=10)
)

# Show the figure
fig1.show()


# In[13]:


import numpy as np
import seaborn as sns
from datetime import datetime


# In[14]:


# Convert columns to appropriate data types
apps_df["Rating"] = pd.to_numeric(apps_df["Rating"], errors="coerce")
apps_df["Reviews"] = pd.to_numeric(apps_df["Reviews"], errors="coerce")

# Clean and convert Installs to numeric
apps_df["Installs"] = apps_df["Installs"].astype(str).str.replace("[+,]", "", regex=True)
apps_df["Installs"] = pd.to_numeric(apps_df["Installs"], errors="coerce")

# Clean and convert Size to numeric (in MB)
apps_df["Size"] = apps_df["Size"].astype(str).str.replace("M", "").replace("Varies with device", np.nan)
apps_df["Size"] = pd.to_numeric(apps_df["Size"], errors="coerce")

# Convert "Last Updated" to datetime
apps_df["Last Updated"] = pd.to_datetime(apps_df["Last Updated"], errors="coerce")


# In[15]:


apps_df[["Rating", "Reviews", "Installs", "Size", "Last Updated"]].head()


# In[16]:


apps_df[["Rating", "Reviews", "Installs", "Size", "Last Updated"]].tail()


# In[17]:


# Apply filtering conditions
filtered_data = apps_df[
    (apps_df["Rating"] >= 4.0) &  # Filter for Rating >= 4.0
    (apps_df["Size"] >= 10) &  # Filter for Size >= 10 MB
    (apps_df["Last Updated"].dt.month == 1)  # Filter for January updates
]


# In[18]:


filtered_data[["App", "Rating", "Size", "Last Updated"]].head()


# In[19]:


# Group by Category and compute statistics
category_stats = filtered_data.groupby("Category").agg(
    avg_rating=("Rating", "mean"),
    total_reviews=("Reviews", "sum"),
    total_installs=("Installs", "sum")
).sort_values(by="total_installs", ascending=False).head(10)


# In[20]:


# Get current IST hour (IST is UTC+5:30)
current_hour = datetime.utcnow().hour + 5.5


# In[21]:


# Fix the missing import and correct the code snippet
import plotly.graph_objects as go

# Check if the current time is within the 3 PM - 5 PM IST window (IST is UTC+5:30, adjust accordingly if needed)
if 15 <= current_hour < 17:
    # Ensure 'category_stats' is available and contains necessary columns
    if 'category_stats' in locals() and all(col in category_stats.columns for col in ["avg_rating", "total_reviews"]):
        
        # Create a grouped bar chart using Plotly
        fig2 = go.Figure()

        # Add bars for Average Rating
        fig2.add_trace(go.Bar(
            x=category_stats.index,
            y=category_stats["avg_rating"],
            name="Average Rating",
            marker_color="skyblue"
        ))

        # Add bars for Total Reviews
        fig2.add_trace(go.Bar(
            x=category_stats.index,
            y=category_stats["total_reviews"],
            name="Total Reviews",
            marker_color="orange"
        ))

        # Update layout for better visualization
        fig2.update_layout(
            title="Average Rating and Total Reviews by Category (Top 10 by Installs)",
            xaxis_title="Category",
            yaxis_title="Value",
            barmode="group",
            plot_bgcolor="black",
            paper_bgcolor="black",
            font_color="white",
            title_font=dict(size=16),
            xaxis=dict(title_font=dict(size=12)),
            yaxis=dict(title_font=dict(size=12)),
            margin=dict(l=10, r=10, t=30, b=10)
        )

        # Show the figure
        fig2.show()

    else:
        print("Error: 'category_stats' is missing or does not contain the required columns.")
else:
    print("This chart is available only between 3 PM IST and 5 PM IST.")


# In[22]:


# Filter data based on conditions
filtered_data = apps_df[
    (apps_df["Content Rating"] == "Teen") &  # Content Rating is 'Teen'
    (apps_df["App"].str.startswith("E")) &  # App name starts with 'E'
    (apps_df["Installs"] > 10000)  # Installs > 10K
]


# In[23]:


filtered_data[["Content Rating", "App", "Installs"]].head()


# In[24]:


filtered_data[["Content Rating", "App", "Installs"]].tail()


# In[25]:


# Extract year-month for aggregation
filtered_data["YearMonth"] = filtered_data["Last Updated"].dt.to_period("M")


# In[26]:


# Group by category and time, then sum installs
time_series_data = filtered_data.groupby(["Category", "YearMonth"]).agg(
    total_installs=("Installs", "sum")
).reset_index()


# In[27]:


time_series_data.head()


# In[28]:


time_series_data.tail()


# In[29]:


# Convert period to datetime for plotting
time_series_data["YearMonth"] = time_series_data["YearMonth"].dt.to_timestamp()

# Calculate month-over-month percentage change
time_series_data["pct_change"] = time_series_data.groupby("Category")["total_installs"].pct_change() * 100


# In[30]:


time_series_data[["YearMonth", "pct_change"]].head()


# In[31]:


# Get current IST hour (UTC+5:30)
current_hour = datetime.utcnow().hour + 5.5


# In[32]:


from datetime import datetime
import pytz
import plotly.express as px

# Get the current time in IST
ist = pytz.timezone('Asia/Kolkata')
current_hour = datetime.now(ist).hour

# Check if within 6 PM - 9 PM IST
if 18 <= current_hour < 21:
    # Ensure 'time_series_data' exists and contains necessary columns
    if 'time_series_data' in locals() and all(col in time_series_data.columns for col in ["YearMonth", "total_installs", "Category", "pct_change"]):
        
        # Create a line chart in Plotly
        fig_line = px.line(
            time_series_data,
            x="YearMonth",
            y="total_installs",
            color="Category",
            markers=True,
            title="Total Installs Trend Over Time by Category (Highlighted: Growth > 20%)",
            labels={"YearMonth": "Time", "total_installs": "Total Installs"},
            width=800,
            height=500
        )

        # Highlight areas where MoM increase > 20%
        highlight_data = time_series_data[time_series_data["pct_change"] > 20]
        highlight_traces = px.area(
            highlight_data,
            x="YearMonth",
            y="total_installs",
            color="Category"
        )

        # Set opacity separately for highlighted areas
        for trace in highlight_traces.data:
            trace.update(opacity=0.3)
            fig_line.add_trace(trace)

        # Update layout with dark theme
        fig_line.update_layout(
            plot_bgcolor="black",
            paper_bgcolor="black",
            font_color="white",
            title_font={"size": 16},
            xaxis=dict(title_font={"size": 12}),
            yaxis=dict(title_font={"size": 12}),
            margin=dict(l=10, r=10, t=30, b=10)
        )

        # Show the figure
        fig_line.show()

    else:
        print("Error: 'time_series_data' is missing or does not contain the required columns.")
else:
    print("This chart is available only between 6 PM and 9 PM IST.")


# In[ ]:




