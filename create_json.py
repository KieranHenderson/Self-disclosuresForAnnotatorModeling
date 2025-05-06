import pandas as pd
import json

"""
This short snippet was used to create the social_norms.json file
that was later used to create the user_embeddings.pkl file. This
file was processed in lines 87 - 90 of users_sbert_embeddings.py

This json file was meant to be a hacky workaround to the missing 
json / history files just to get the embedding script running.
"""

df = pd.read_csv(r"C:\Users\User\PycharmProjects\perspectivism-personalization\data\social_norms_clean.csv")

json_df = df[['id', 'body', 'parent_id', 'author_name']].copy()
json_df = json_df.rename(columns={'author_name': 'author'})
json_data = json_df.to_dict('records')

with open('social_norms.json', 'w') as f:
    json.dump(json_data, f)