import joblib
import pandas as pd
import scipy.sparse as sparse

class Recommender(object):
    def __init__(self):
        self.classifier = joblib.load("rf_classifier.joblib")
        self.user_history = joblib.load('read_history_dict.joblib')
        self.most_popular_books = joblib.load('most_popular_book.joblib')
        self.als_model = joblib.load('als_model.joblib')
        self.matrix = joblib.load('user_item_matrix.joblib')
        self.books_index = joblib.load('book_index.joblib')
        self.user_index = joblib.load('user_index.joblib')
        self.books_profile = joblib.load('books_profile.joblib') 
        self.users_profile = joblib.load('user_profile.joblib')
        self.cols_for_using = joblib.load('cols_for_using.joblib')


    def get_most_popular_books_recommendations(self, user_id): 
        books = [book for book in self.most_popular_books if book not in self.user_history[user_id]]
        return books    
    
    def get_hybrid_recommendations(self, user_id, usr_ind):
        already_seen = self.user_history[user_id]
        als_recs = list(self.als_model.recommend(usr_ind, sparse.csr_matrix(self.matrix[usr_ind]), N=30, filter_already_liked_items=True)[0])
        recos_books_for_user = []
        for ind in als_recs: 
            if self.books_index[ind] not in already_seen: 
                books_feat = self.books_profile[self.books_profile.book_id_x==self.books_index[ind]]
                user_feat = self.users_profile[self.users_profile.user_id==self.user_index[usr_ind]]
                prob_of_like = self.classifier.predict_proba(user_feat.merge(books_feat, how='cross')[self.cols_for_using])[0][1]
                recos_books_for_user.append((self.books_index[ind], prob_of_like))

        recos_books_for_user = [int(item[0]) for item in sorted(recos_books_for_user, key=lambda x: x[1], reverse=True)]
        return recos_books_for_user
