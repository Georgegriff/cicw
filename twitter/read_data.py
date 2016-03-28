import glob
import os

training_tweets = []
training_tweet_labels = []

testing_pos_tweets = []
testing_pos_labels = []
testing_neg_tweets = []
testing_neg_labels = []
POSITIVE = 1
NEGATIVE = 0


def read_training_data(negative_path, positive_path):
	neg_files = glob.glob(os.path.join(negative_path, '*'))
	pos_files = glob.glob(os.path.join(positive_path, '*'))
	read_neg(neg_files, training_tweets, training_tweet_labels)
	read_pos(pos_files, training_tweets, training_tweet_labels)
	return training_tweets, training_tweet_labels


def read_testing_data(path, isNeg):
	files = glob.glob(os.path.join(path, '*'))
	if isNeg == True:
		read_neg(files, testing_pos_tweets, testing_pos_labels)
		return testing_pos_tweets, testing_pos_labels
	else:
		read_pos(files, testing_neg_tweets, testing_neg_labels)
		return testing_neg_tweets, testing_neg_labels


def read_neg(neg_files, tweets, tweet_labels):
	for negative in neg_files:
		with open(negative, 'r')as file:
			text = file.read()
			text = unicode(text, "utf-8", "ignore")
			tweets.append(text)
			tweet_labels.append(NEGATIVE)


def read_pos(pos_files, tweets, tweet_labels):
	for positive in pos_files:
		with open(positive, 'r')as file:
			text = file.read()
			text = unicode(text, "utf-8", "ignore")
			tweets.append(text)
			tweet_labels.append(POSITIVE)
