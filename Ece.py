import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


df1 = pd.read_csv("Reviews.csv")
df2 = pd.read_csv("Apps.csv")


df1.rename(columns={"score": "avg_score"}, inplace=True)
df2.rename(columns={"score": "rated_score"}, inplace=True)
df1.rename(columns={"app_Id": "appId"}, inplace=True)
merged_df = pd.DataFrame(pd.merge(df1, df2, on="appId", how="inner"))



merged_df.drop("userName", axis=1, inplace=True)
merged_df.drop("userImage", axis=1, inplace=True)
merged_df.drop("reviewCreatedVersion", axis=1, inplace=True)
merged_df.drop("repliedAt", axis=1, inplace=True)
merged_df.drop("url", axis=1, inplace=True)
merged_df.drop("adSupported", axis=1, inplace=True)
merged_df.drop("containsAds", axis=1, inplace=True)
merged_df.drop("free", axis=1, inplace=True)
merged_df.drop("contentRatingDescription", axis=1, inplace=True)
merged_df.drop("reviewId", axis=1, inplace=True)



merged_df["content"] = merged_df["content"].astype(str)
merged_df["avg_score"] = merged_df["avg_score"].astype(float)
merged_df["thumbsUpCount"] = merged_df["thumbsUpCount"].astype(float)
merged_df["replyContent"] = merged_df["replyContent"].astype(str)
merged_df["appId"] = merged_df["appId"].astype(str)
merged_df["title"] = merged_df["title"].astype(str)
merged_df["rated_score"] = merged_df["rated_score"].astype(float)
merged_df["reviews"] = merged_df["reviews"].astype(float)
merged_df["price"] = merged_df["price"].astype(float)
merged_df["genre"] = merged_df["genre"].astype(str)
merged_df["ratings"] = merged_df["ratings"].astype(float)
merged_df["contentRating"] = merged_df["contentRating"].astype(str)



merged_df["installs"] = merged_df["installs"].str.replace("+","")
merged_df["installs"] = merged_df["installs"].str.replace(",","")
merged_df["installs"] = merged_df["installs"].astype(float)


filtered_rows = merged_df.loc[merged_df["size"].str.contains("Varies with device")]
filtered_rows_1 = merged_df.loc[merged_df["size"].str.contains("M")]
filtered_rows_2 = merged_df.loc[merged_df["size"].str.contains("k")]


if not filtered_rows.empty:
    merged_df.loc[filtered_rows.index, "size"] = None
if not filtered_rows_1.empty:
    merged_df.loc[filtered_rows_1.index, "size"] = filtered_rows_1["size"].str.replace("M", "")
if not filtered_rows_2.empty:
    merged_df.loc[filtered_rows_2.index, "size"] = filtered_rows_2["size"].str.replace("k", "")

merged_df["size"] = merged_df["size"].astype(float)
merged_df["size"] = merged_df["size"] * 1024


merged_df["size"] = merged_df["size"].fillna(merged_df["size"].mean())
merged_df["replyContent"] = merged_df["replyContent"].fillna("None")
merged_df["rated_score"] = merged_df["rated_score"].fillna(merged_df["rated_score"].mean())
merged_df["reviews"] = merged_df["reviews"].fillna(0).astype(float)
merged_df["ratings"] = merged_df["ratings"].fillna(merged_df["ratings"].mean())
merged_df["released"] = merged_df["released"].fillna("None")


print((merged_df["title"].unique()))