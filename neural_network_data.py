import pandas as pd
from numpy import append, array

# # # # # #
# Helpers #
# # # # # #
# returns only action movies
def get_movies_data_by_genre():
    i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
     'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv('data/u.item', sep='|', names=i_cols, encoding='latin-1').values

    genres = ['Unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    genres_dict = {
        'Unknown': [], 'Action': [], 'Adventure': [], 'Animation': [], 'Children': [],
        'Comedy': [], 'Crime': [], 'Documentary': [], 'Drama': [], 'Fantasy': [],
        'Film-Noir': [], 'Horror': [], 'Musical': [], 'Mystery': [], 'Romance': [],
        'Sci-Fi': [], 'Thriller': [], 'War': [], 'Western': []
    }

    def find_offset_genre_index(e):
        for i in xrange(5,len(e)):
            if e[i] == 1:
                return i-5

    for e in items:
        genre = genres[find_offset_genre_index(e)]
        genres_dict[genre].append(e)

    return genres_dict

# returns [[user_id, age, sex, occupation]]
def get_users_data():
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('data/u.user', sep='|', names=u_cols, encoding='latin-1').values
    return users[:,[0,1,2,3]]

# returns [[user_id, movie_id, rating]]
def get_ratings_data():
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('data/u.data', sep='\t', names=r_cols, encoding='latin-1').values
    return ratings[:,[0,1,2]]

# equivalent of a ratings left outer join with users on id
def join_ratings_and_users():
    ratings = get_ratings_data()
    users = get_users_data()

    result = []
    for r in ratings:
        for u in users:
            if r[0] == u[0]:
                result.append(append(r[1:],u[1:]))
                break
            continue

    # [[movie_id, rating, age, sex, occupation]]
    return result

# # # # # # # # #
# Training Data #
# # # # # # # # #
def get_training_data_by_genre(genre):
    movies = get_movies_data_by_genre()[genre]

    # left join on join_ratings_and_users
    ratings_and_users = join_ratings_and_users()

    training_input = []
    training_output = []
    for m in movies:
        for ru in ratings_and_users:
            if m[0] == ru[0]:
                training_input.append(ru[2:])
                training_output.append(ru[1])
                break
            continue

    return [training_input, training_output]
