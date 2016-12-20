import pandas as pd
from numpy import exp, array, random, dot

def find_offset_genre_index(e):
    for i in xrange(5,len(e)):
        if e[i] == 1:
            return i-5

# returns only action movies
def get_movies_data_by_genre():
    i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
     'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv('data/u.item', sep='|', names=i_cols, encoding='latin-1').values

    genres = ['Unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    genres_dict = {
        'Unknown': [],
        'Action': [],
        'Adventure': [],
        'Animation': [],
        'Children': [],
        'Comedy': [],
        'Crime': [],
        'Documentary': [],
        'Drama': [],
        'Fantasy': [],
        'Film-Noir': [],
        'Horror': [],
        'Musical': [],
        'Mystery': [],
        'Romance': [],
        'Sci-Fi': [],
        'Thriller': [],
        'War': [],
        'Western': []
    }
    for e in items:
        genre = genres[find_offset_genre_index(e)]
        genres_dict[genre].append(e)

    return genres_dict

# returns [[user_id, age, sex, occupation]]
def get_users_data():
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('data/u.user', sep='|', names=u_cols, encoding='latin-1').values
    return users[:,[0,3]]

# returns [[movie_id, rating]]
def get_ratings_data():
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('data/u.data', sep='\t', names=r_cols, encoding='latin-1').values
    return ratings[:,[1,2]]

# # Setting up training and test data
# r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
# ratings_base = pd.read_csv('data/ua.base', sep='\t', names=r_cols, encoding='latin-1').values
# ratings_test = pd.read_csv('data/ua.test', sep='\t', names=r_cols, encoding='latin-1').values
