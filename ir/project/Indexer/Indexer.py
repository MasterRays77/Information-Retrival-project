import os
import pickle
import nltk
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import warnings

warnings.filterwarnings("ignore")

nltk.download('wordnet')
nltk.download('omw-1.4')

class Indexer:
    def __init__(self, html_file_path):
        self.html_file_path = html_file_path
        self.vectorizer = TfidfVectorizer(stop_words=None, tokenizer=self.custom_tokenizer, token_pattern=None)
        self.documents, self.filenames = self.retrieve_documents()
        self.inverted_index = {}  
        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)  # Fit the vectorizer here
            self.inverted_index = self.build_inverted_index()
        else:
            self.tfidf_matrix = None
            print("No documents found.")
        self.save_inverted_index_to_pickle()  # Print inverted index when initializing
            
    def retrieve_documents(self):
        documents = []
        filenames = []
        with open(self.html_file_path, 'r', encoding="utf-8", errors="ignore") as html_file:
            text = html_file.read()  # Read the entire content of the HTML file
            documents.append(text)
            filenames.append(self.html_file_path)  # Using the HTML file path as the filename
        return documents, filenames

    
    def build_inverted_index(self):
        inverted_index = {}
        terms = self.vectorizer.get_feature_names_out()
        term_indices = {term: i for i, term in enumerate(terms)}
        for i, doc in enumerate(self.documents):
            for term in terms:
                if term in doc:
                    if term not in inverted_index:
                        inverted_index[term] = []
                    inverted_index[term].append(i)
        return inverted_index
    
    def custom_tokenizer(self, text):
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        tokens = word_tokenize(text)
        tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return tokens

    def search(self, query, top_k=5):
        if self.tfidf_matrix is None or self.tfidf_matrix.shape[0] == 0:
            return []
        query_vector = self.vectorizer.transform([query])
        query_terms = query.split()
        query_terms_set = set(query_terms)
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix)
        scores = cosine_similarities.flatten()
        doc_indices = scores.argsort()[::-1]
        result = []
        for doc_index in doc_indices[:top_k]:  # Select top-K results
            filename = self.filenames[doc_index]
            score = scores[doc_index]
            if score > 0:
                tfidf_scores = {term: self.tfidf_matrix[doc_index, self.vectorizer.vocabulary_.get(term, -1)] for term in query_terms_set}
                result.append((filename, score, tfidf_scores))
        return result

    def save_search_results_to_pickle(self, results, filename):
        with open(filename, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Search results:")
        for filename, similarity_score, tfidf_scores in results:
            print(f" Cosine Similarity Score: {similarity_score}, TF-IDF Scores: {tfidf_scores}")

    def save_inverted_index(self):
        print("Inverted index:")
        print("{")
        for term, indices in self.inverted_index.items():
            print(f"'{term}': {indices}, ", end="")
        print("\n}")

    def save_inverted_index_to_pickle(self):
        print("Inverted index content (Pickle format):")
        print(pickle.dumps(self.inverted_index))  # Print content in pickle format directly to terminal
        self.save_inverted_index()


# Usage example:
def main():
    html_file_path = 'C:\\Users\\Gurukiran S\\OneDrive\\Desktop\\ir\\project\\Scrapy\\posts\\output.html'
    indexer = Indexer(html_file_path)
    query = input("Enter your query: ")
    results = indexer.search(query)
    indexer.save_search_results_to_pickle(results, "search_results.pickle")

if __name__ == "__main__":
    main()