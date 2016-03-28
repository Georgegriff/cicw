import read_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import numpy as np

stop_words = ['a','the','and', 'of', 'or', 'then', 'an']
pattern = '(?u)\\b[A-Za-z]{3,}'
tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=stop_words,token_pattern=pattern,ngram_range=(1,3))
#read the training_data data
training_tweets, training_labels = read_data.read_training_data("./training_data/negative", "./training_data/positive")

training_matrix = tfidf.fit_transform(training_tweets)
svc_classifier = svm.LinearSVC()
svc_classifier.fit(training_matrix, training_labels)


pos_testing, pos_testing_labels = read_data.read_testing_data("./test_data/positive", False)
neg_testing, neg_testing_labels = read_data.read_testing_data("./test_data/negative", True)

#Testing Features
testing_positive_features = tfidf.transform(pos_testing)
testing_negative_features = tfidf.transform(neg_testing)

#Testing Results
results_negative = svc_classifier.predict(testing_negative_features)
results_positive = svc_classifier.predict(testing_positive_features)

def calculate_accuracy():
	pos_total = pos_testing.__len__()
	total =  pos_total + neg_testing.__len__()
	incorrect_negative = np.flatnonzero(results_negative)
	total_incorrect_neg = incorrect_negative.__len__()

	correct_positive = np.flatnonzero(results_positive)
	total_incorrect_pos = pos_total - correct_positive.__len__()

	total_incorrect = total_incorrect_neg + total_incorrect_pos

	return (float(total - total_incorrect) / float(total)) * 100

print "Accuracy: %s %%" % calculate_accuracy()
