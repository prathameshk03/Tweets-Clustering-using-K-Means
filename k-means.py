import pandas as pd
import re

def lemmatize_text(text):
    words = text.split()
    lemmatized_words = []
    for word in words:
        lemmatized_word = word
        lemmatized_words.append(lemmatized_word)
    return lemmatized_words

def preprocess(data):
    # Removing timestamp and tweet id
    data = data.drop(data.columns[[0, 1]], axis=1)
    
    # Removin any word that starts with any symbol
    data['tweet'] = data['tweet'].str.replace('(@\w+.*?)', "")
    
    # Removing any hashtag symbols 
    data['tweet'] = data['tweet'].replace({'#': ''}, regex=True)
    
    # Removing any URLs
    data['tweet'] = data['tweet'].str.rsplit('http').str[0]
    
    # Lemmatizing and tokenizing the tweets
    data['tweet'] = data['tweet'].apply(lemmatize_text)

    return data

# Calculating the Jaccard distance
def calc_jaccard(arg1, arg2):
    arg1 = set(arg1)
    arg2 = set(arg2)
    union = len(list((arg1 | arg2)))
    intersection = len(list((arg1 & arg2)))
    return 1 - (intersection / union)

#Calculating Sum of Squared Error
def calc_sse(k_clusters, centroids):
    sse = 0
    for i in centroids.keys():
        for j in list(k_clusters[i]):
            sse += calc_jaccard(centroids[i], j) ** 2
    return sse

# k-means
def kmeans(K, data, centroids=None):
    # Shuffling the rows
    data = data.sample(frac=1).reset_index(drop=True)
    
    # Initializing the centroids
    if centroids == None:
        centroids = {}
        for i in range(K):
            if data['tweet'].iloc[i] not in list(centroids.keys()):
                centroids[i] = data['tweet'].iloc[i]
    
    # Forming K Clusters
    keys = range(K)
    k_clusters = dict(zip(keys, ([] for _ in keys)))
    
    # Get distance of each tweet from all centroids in the cluster and put it in the nearest cluster
    for item in data['tweet']:
        dist = [calc_jaccard(item, centroids[i]) for i in centroids]
        min_dist = dist.index(min(dist))
        k_clusters[min_dist].append(item)
    
    # Calculate new centroids based on the minimum distance of sum of all tweets with respect to each other in a cluster
    recomputed_centroids = dict(zip(keys, ([] for _ in keys)))
    for i in k_clusters:
        cluster = k_clusters[i]
        dists_in_cluster = []
        for j in cluster:
            if j != []:
                dist = [calc_jaccard(j, c) for c in cluster]
                dists_in_cluster.append(sum(dist))
        index = dists_in_cluster.index(min(dists_in_cluster))
        recomputed_centroids[i] = cluster[index]
    
    change_flag = True
    
    # Comparing the old centroids and the recomputed centroids to check if they are the same
    for i in range(K):
        if list(centroids.values())[i] != list(recomputed_centroids.values())[i]:
            change_flag = True
            break
        else:
            change_flag = False
    
    if change_flag:
        print("Centroids changed. Centroids recomputed.")
        centroids = recomputed_centroids.copy()
        kmeans(K, data, centroids)
    else:
        sse = calc_sse(k_clusters, centroids)
        sse = '{:.2f}'.format(sse)
        print("Converged.\nSSE is ", sse)
        print("Kth Cluster : Number of tweets")
        for i in range(K):
            print(i+1, ":", sse, ":", len(k_clusters[i]), "tweets")
    
    return None

dataset = "https://raw.githubusercontent.com/prathameshk03/Tweets-Clustering-using-K-Means/master/usnewshealth.txt"
data = pd.read_csv(dataset, names=['tweet_id', 'data_time', 'tweet'], sep='|')
data = preprocess(data)

k_values = [5, 10, 15, 20, 25, 30]
for i in k_values:
    kmeans(i, data, centroids=None)