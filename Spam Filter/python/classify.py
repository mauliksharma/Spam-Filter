import csv

from textblob import TextBlob

import pandas
import sklearn
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve

#Loading dataset
messages = pandas.read_csv('../data/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["label", "message"])
print '\nFirst few Messages in Corpus: '
print messages.head(), '\n'

print 'Number of messages: ', len(messages), '\n'

#Data Preprocessing
def split_into_tokens(message): #Tokenisation
    message = unicode(message, 'utf8')

print 'Part of Speech tags:'
print TextBlob("Hello world, how is it going?").tags, '\n'  # list of (word, POS) pairs

def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

print 'Tokenised and Lemmatised Corpus Head:'
print messages.message.head().apply(split_into_lemmas), '\n'

#Data to Vectors (Feature Extraction)
bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
print 'Number of features (Vocabulary)', len(bow_transformer.vocabulary_), '\n'

message4 = messages['message'][3] #Example of vectorisation
print 'Example message: '
print message4
bow4 = bow_transformer.transform([message4])
print 'Bag of Words Conversion of Example: '
print bow4
print bow4.shape, '\n'
print 'Words that appear twice:'
print bow_transformer.get_feature_names()[6736]
print bow_transformer.get_feature_names()[8013], '\n'

messages_bow = bow_transformer.transform(messages['message'])
print 'Complete vectorised corpus sparse matrix :- '
print 'sparse matrix shape:', messages_bow.shape
print 'number of non-zeros:', messages_bow.nnz
print 'sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])), '\n'

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print 'TF-IDF Conversion of Example'
print tfidf4, '\n'

messages_tfidf = tfidf_transformer.transform(messages_bow)
print 'Shape of TF-IDF Converted feature vectorised corpus matrix:'
print messages_tfidf.shape, '\n'

#Model Training, Classification
spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])

print 'Prediction of our Example message: '
print 'predicted:', spam_detector.predict(tfidf4)[0]
print 'expected:', messages.label[3], '\n'

all_predictions = spam_detector.predict(messages_tfidf)
print 'All Predictions: '
print all_predictions, '\n'

print 'Accuracy', accuracy_score(messages['label'], all_predictions), '\n'
print 'Confusion matrix\n', confusion_matrix(messages['label'], all_predictions)
print '(row=expected, col=predicted)\n'

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

msg_train, msg_test, label_train, label_test = \
    train_test_split(messages['message'], messages['label'], test_size=0.2)

scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )
print scores
