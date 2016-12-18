import pandas as pd
import graphlab

# Files and their schema, using panda
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('data/u.user', sep='|', names=u_cols, encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('data/u.data', sep='\t', names=r_cols, encoding='latin-1')

i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('data/u.item', sep='|', names=i_cols, encoding='latin-1')


# Setting up training and test data
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('data/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('data/ua.test', sep='\t', names=r_cols, encoding='latin-1')

# Convert into SFrames to use with GraphLab
train_data = graph.SFrame(ratings_base)
test_data = graph.SFrame(ratings_test)


# Training models

# Popularity Based Model:
#   all users have the same recommendation model based on the global ratings
popularity_model = graphlab.popularity_recommmender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')

# Collaborative Filtering Model
# Jaccard Similarity:
#   computes the ratio between the number of users who have rated A or B, and the distinct users who have rated said movies
# Cosine Similarity:
#   given precomputed vectors representing movie type and number of ratings, compute cosine between A and B
# Pearson Similarity:
#   computes the linear dependence between two variables X and Y
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='pearson')


# Evaluating models
model_performance = graphlab.compare(test_data, [popularity_model, item_sim_model])
graphlab.show_comparison(model_performance,[popularity_model, item_sim_model])
