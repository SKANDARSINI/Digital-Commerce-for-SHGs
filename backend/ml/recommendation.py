# import pymongo
# import pandas as pd
# import numpy as np
# import seaborn as sns
# from matplotlib import pyplot as plt
# import warnings
# import os
# import importlib
# warnings.filterwarnings('ignore')
# sns.set_theme(color_codes=True)

# # user-defined function to check library is installed or not, if not installed then it will install automatically at runtime.
# def check_and_install_library(library_name):
#     try:
#         importlib.import_module(library_name)
#         print(f"{library_name} is already installed.")
#     except ImportError:
#         print(f"{library_name} is not installed. Installing...")
#         try:
#             import pip
#             pip.main(['install', library_name])
#         except:
#             print("Error: Failed to install the library. Please install it manually.")

# # Connect to MongoDB Atlas
# client = pymongo.MongoClient("mongodb+srv://Skandarsini:darsh%402312@cluster0.scqmoj3.mongodb.net/thrivetogether")
# db = client["thrivetogether"]
# collection = db["recommendations"]

# # Retrieve data from MongoDB
# cursor = collection.find({}, {"_id": 0, "__v": 0})  # Exclude _id field
# mongo_data = list(cursor)

# # Convert data to DataFrame
# df = pd.DataFrame(mongo_data)

# df.head()

# df.shape

# df.columns

# df.info()

# df.drop('timestamp',axis=1,inplace=True)

# df.describe()

# #handle missing values
# df.isnull().sum()

# #handling duplicate records
# df[df.duplicated()].shape[0]

# df.head()

# plt.figure(figsize=(8,4))
# sns.countplot(x='rating',data=df)
# plt.title('Rating Distribution')
# plt.xlabel('Rating')
# plt.ylabel('Count')
# plt.grid()
# plt.show()

# print('Total rating : ',df.shape[0])
# print('Total unique users : ',df['userId'].unique().shape[0])
# print('Total unique products : ',df['productId'].unique().shape[0])

# no_of_rated_products_per_user = df.groupby(by='userId')['rating'].count().sort_values(ascending=False)
# no_of_rated_products_per_user.head()

# print('No of rated product more than 50 per user : {} '.format(sum(no_of_rated_products_per_user >= 1)))

# """#Popularity Based Recommendation

# Popularity based recommendation system works with the trend. It basically uses the items which are in trend right now. For example, if any product which is usually bought by every new user then there are chances that it may suggest that item to the user who just signed up.

# The problems with popularity based recommendation system is that the personalization is not available with this method i.e. even though you know the behaviour of the user but you cannot recommend items accordingly.
# """

# data=df.groupby('productId').filter(lambda x:x['rating'].count()>=2)

# data.head()

# no_of_rating_per_product=data.groupby('productId')['rating'].count().sort_values(ascending=False)

# no_of_rating_per_product.head()

# #top 4 product
# no_of_rating_per_product.head(4).plot(kind='bar')
# plt.xlabel('Product ID')
# plt.ylabel('num of rating')
# plt.title('top 20 procduct')
# plt.show()

# #average rating product
# mean_rating_product_count=pd.DataFrame(data.groupby('productId')['rating'].mean())

# mean_rating_product_count.head()

# #plot the rating distribution of average rating product
# plt.hist(mean_rating_product_count['rating'],bins=100)
# plt.title('Mean Rating distribution')
# plt.show()

# #check the skewness of the mean rating data
# mean_rating_product_count['rating'].skew()

# mean_rating_product_count['rating_counts'] = pd.DataFrame(data.groupby('productId')['rating'].count())

# mean_rating_product_count.head()

# #highest mean rating product
# mean_rating_product_count[mean_rating_product_count['rating_counts']==mean_rating_product_count['rating_counts'].max()]

# #min mean rating product
# print('min average rating product : ',mean_rating_product_count['rating_counts'].min())
# print('total min average rating products : ',mean_rating_product_count[mean_rating_product_count['rating_counts']==mean_rating_product_count['rating_counts'].min()].shape[0])

# #plot the rating count of mean_rating_product_count
# plt.hist(mean_rating_product_count['rating_counts'],bins=100)
# plt.title('rating count distribution')
# plt.show()

# #joint plot of rating and rating counts
# sns.jointplot(x='rating',y='rating_counts',data=mean_rating_product_count)
# plt.title('Joint Plot of rating and rating counts')
# plt.tight_layout()
# plt.show()

# plt.scatter(x=mean_rating_product_count['rating'],y=mean_rating_product_count['rating_counts'])
# plt.show()

# print('Correlation between Rating and Rating Counts is : {} '.format(mean_rating_product_count['rating'].corr(mean_rating_product_count['rating_counts'])))

# """##Collaberative filtering (Item-Item recommedation)

# Collaborative filtering is commonly used for recommender systems. These techniques aim to fill in the missing entries of a user-item association matrix. We are going to use collaborative filtering (CF) approach. CF is based on the idea that the best recommendations come from people who have similar tastes. In other words, it uses historical item ratings of like-minded people to predict how someone would rate an item.Collaborative filtering has two sub-categories that are generally called memory based and model-based approaches.
# """

# #import surprise library for collebrative filtering
# #check_and_install_library('surprise')
# from surprise import KNNWithMeans
# from surprise import Dataset
# from surprise import accuracy
# from surprise import Reader
# from surprise import train_test_split

# #Reading the dataset
# reader = Reader(rating_scale=(1, 5))
# surprise_data = Dataset.load_from_df(data,reader)

# #Splitting surprise the dataset into 80,20 ratio using train_test_split
# trainset, testset = train_test_split(surprise_data, test_size=0.3,random_state=42)

# # Use user_based true/false to switch between user-based or item-based collaborative filtering
# algo = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})
# algo.fit(trainset)

# #make prediction using testset
# test_pred=algo.test(testset)

# #print RMSE
# print("Item-based Model : Test Set")
# accuracy.rmse(test_pred ,verbose=True)

# """#Model-based collaborative filtering system

# These methods are based on machine learning and data mining techniques. The goal is to train models to be able to make predictions. For example, we could use existing user-item interactions to train a model to predict the top-5 items that a user might like the most. One advantage of these methods is that they are able to recommend a larger number of items to a larger number of users, compared to other methods like memory based approach. They have large coverage, even when working with large sparse matrices.
# """

# data2=data
# ratings_matrix = data2.pivot_table(values='rating', index='userId', columns='productId', fill_value=0)
# ratings_matrix.head()

# #check the shape of the rating_matrix
# ratings_matrix.shape

# #transpose the metrix to make column (productId) as index and index as column (userId)
# x_ratings_matrix=ratings_matrix.T
# x_ratings_matrix.head()

# x_ratings_matrix.shape

# #Decomposition of the matrix using Singular Value Decomposition technique
# from sklearn.decomposition import TruncatedSVD
# SVD = TruncatedSVD(n_components=2)
# decomposed_matrix = SVD.fit_transform(x_ratings_matrix)
# decomposed_matrix.shape

# #Correlation Matrix
# correlation_matrix = np.corrcoef(decomposed_matrix)
# correlation_matrix.shape

# x_ratings_matrix.index[1]

# i=2
# product_names=list(x_ratings_matrix.index)
# product_id=product_names.index(i)
# print(product_id)

# correlation_product_ID = correlation_matrix[product_id]
# correlation_product_ID.shape

# correlation_matrix[correlation_product_ID>0.75].shape

# #Recommending top 4 highly correlated products in sequence
# recommend = list(x_ratings_matrix.index[correlation_product_ID > 0.75])
# recommend[:4]

# def get_recommendations(user_id):
#     # Logic to generate recommendations based on user ID
#     # This could involve loading a model, processing data, and generating recommendations
#     recommendations = ['Product A', 'Product B', 'Product C']
#     return recommendations

import numpy as np

class RecommendationModel:

    def __init__(self, user_data, product_data):
        """
        Initialize the RecommendationModel with user data and product data

        :param user_data: dict of user_id to dict of product_id to rating
        :param product_data: dict of product_id to product_name
        """
        self.user_data = user_data
        self.product_data = product_data

    def _similarity(self, user1, user2, norm=False):
        """
        Compute similarity between two users

        :param user1: dict of product_id to rating
        :param user2: dict of product_id to rating
        :param norm: If True, divide by vector length to normalize
        """
        if norm:
            return np.dot(np.array(list(user1.values())), np.array(list(user2.values()))) / \
                   (np.linalg.norm(np.array(list(user1.values()))) * np.linalg.norm(np.array(list(user2.values()))))
        else:
            return np.dot(np.array(list(user1.values())), np.array(list(user2.values())))

    def predict(self, user_id):
        """
        Predict the top N products that the user would like

        :param user_id: str or int, user for which to make prediction
        :return: list of tuples in format (product_id, product_name, similarity)
        """

        user_data = self.user_data

        # Create dict of all users' ratings minus the target user
        users = user_data.copy()
        users.pop(user_id, None)

        # Calculate pairwise similarities between users
        user_similarities = {}
        for user_id1, user_dict1 in users.items():
            for user_id2, user_dict2 in users.items():
                if user_id1 == user_id2:
                    continue
                key = (user_id1, user_id2)
                if key in user_similarities:
                    user_similarities[key] += self._similarity(user_dict1, user_dict2)
                else:
                    user_similarities[key] = self._similarity(user_dict1, user_dict2)

        # Find the top N most similar users
        similar_users = sorted(user_similarities.items(), key=lambda x: x[1], reverse=True)[:5]

        # Calculate predicted ratings for each product
        predicted_ratings = {}
        for product_id, _ in self.user_data[user_id].items():
            ratings = [v[1] for k, v in similar_users if product_id in user_data[k]]
            if not ratings:
                continue
            total_similarity = sum(ratings)
            predicted_rating = sum(r * sim for r, sim in zip(user_data[user_id][product_id], ratings)) / total_similarity
            predicted_ratings[product_id] = predicted_rating

        # Sort by predicted rating to get top recommendations
        sorted_ratings = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)

        # Format results as tuples for Flask response
        recommendations = [(self.product_data[product_id], product_id, round(rating, 1)) for product_id, rating in
                         sorted_ratings[:10]]

        return recommendations