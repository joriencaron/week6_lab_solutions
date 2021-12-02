#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
import pickle
import random


# In[2]:


top100 = pd.read_csv('top100.csv').drop(['Unnamed: 0'],axis=1)


# In[3]:


cluster_df = pd.read_csv('df_withclusters.csv').drop(['Unnamed: 0'],axis=1)


# In[4]:


song_features_std = pd.read_csv('std_features.csv').drop(['Unnamed: 0'],axis=1)


# In[10]:


cid = input('Please input your Spotify Client ID:')
s_id = input('Please input your Spotify Secret ID:')


# In[11]:


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=cid,
                                                           client_secret=s_id))


# In[12]:


kmeans = pickle.load(open('kmean.pkl','rb'))


# In[13]:


scaler = pickle.load(open('scaler.pkl','rb'))


# In[14]:


def features(track, artist):
    track_id = sp.search(q='artist:' + artist + ' track:' + track, type='track')
    uri = track_id["tracks"]["items"][0]['id']
    features = sp.audio_features(uri)
    return features


# In[15]:


print('Welcome Jan, Erin & Fred. This is the GNOD song recommender!')
answer = input('Do you want me to recommend a song?(yes/no) ').lower()

if answer == 'no':
    print('Too bad, you missed out on a great song!')

else:

    new_song = input("Enter song: ").lower()
    new_artist = input("Enter artist: ").lower()

    test_entry = sp.search(q=new_song, type='track')
    wrong_entry = test_entry["tracks"]["total"] 

    while wrong_entry == 0:
        print("Song does not exists, try another")
        new_song = input("Enter song: ").lower()
        new_artist = input("Enter artist: ").lower()
        test_entry = sp.search(q=new_song, type='track')
        wrong_entry = test_entry["tracks"]["total"]

    if new_song in list(top100['song_title']):
        recommendation_hot = random.choice(list(top100['song_title']))
        while recommendation_hot == new_song:
                recommendation_hot = random.choice(list(top100['song_title']))
                print('Your recommendation:', recommendation_hot)
    else:
        feature = features(new_song, new_artist)
        column = list(feature[0].keys())
        values = [list(feature[0].values())]
        df_new_song = pd.DataFrame(data = feature, columns = column)
        df_new_song = df_new_song.drop(['type','id','uri','track_href','analysis_url','time_signature'],axis=1)
        std_new_song = scaler.transform(df_new_song)
        new_cluster = kmeans.predict(std_new_song)
        df_cluster = cluster_df[cluster_df['cluster'] == list(new_cluster)[0]]
        
        recommendation = random.choice(list(df_cluster['song_and_artist']))
        url = df_cluster['url'][df_cluster['song_and_artist'] == recommendation].values[0]

        print('Your recommendation:',recommendation)
        print('URL:',url)


# In[ ]:




