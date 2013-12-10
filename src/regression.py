import ast
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
import pylab as pl
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
#from sklearn.linear_model import BayesianRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import os

def insertInDict(input, user_num_words_dict, user_num_chats_dict, user, words):
  word_dict = ast.literal_eval(words)
  if (len(word_dict.keys()) == 0):
    return
  if user not in input:
    input[user] = {}
  if user not in user_num_words_dict:
    user_num_words_dict[user] = 0
    user_num_chats_dict[user] = 0
  count = 0
  for word in word_dict:
    count += word_dict[word]
    if word not in input[user]:
      input[user][word] = word_dict[word]
    else:
      input[user][word] += word_dict[word]
  user_num_chats_dict[user] += 1
  user_num_words_dict[user] += count


def convertToInputDict():
  f = open('../data/new_chats_dataset.csv')
  training_dict = {}
  user_num_chats_dict = {}
  user_num_words_dict = {}
  user_quality = {}
  for line in f.readlines():
    words = line.split(';')
    assert len(words) == 14, words
    insertInDict(training_dict, user_num_words_dict, user_num_chats_dict, words[1], words[12])
    insertInDict(training_dict, user_num_words_dict, user_num_chats_dict, words[2], words[13])
  print 'file read'
  for user in user_num_words_dict:
    # if user_num_chats_dict[user] < 2:
    #   del training_dict[user]
    #   continue
    user_quality[user] = user_num_words_dict[user] / float(user_num_chats_dict[user])
  print 'target made'
  return (training_dict, user_quality)


def vectorize(input_dict, target_dict):
  input_list = []
  target_list = []
  for key in sorted(input_dict.keys()):
    input_list.append(input_dict[key])
    target_list.append(target_dict[key])
  v = DictVectorizer()
  input_vector = v.fit_transform(input_list)
  return (input_vector, target_list)

def regressionUsingSVM(input_train, input_test, result_train, result_test):
  filename = '../models/LinearSVM.pkl'
  if os.path.exists(filename):
    classifier = joblib.load(filename)
  else:
    classifier = SVR(cache_size=1000, kernel='linear')
    classifier.fit(input_train, result_train)
    joblib.dump(classifier, filename, compress=9)
  print 'Fitting done'
  score(classifier, input_test, result_test, 'LinearSVM')

def regressionUsingNonLinearSVM(input_train, input_test, result_train, result_test):
  filename = '../models/NonLinearSVM.pkl'
  if os.path.exists(filename):
    classifier = joblib.load(filename)
  else:
    classifier = SVR(cache_size=1000)
    classifier.fit(input_train, result_train)
    joblib.dump(classifier, filename, compress=9)
  print 'Fitting done'
  score(classifier, input_test, result_test, 'NonLinearSVM')

def regressionUsingLinearRegression(input_train, input_test, result_train, result_test):
  filename = '../models/LinearRegression.pkl'
  if os.path.exists(filename):
    classifier = joblib.load(filename)
  else:
    classifier = LinearRegression()
    classifier.fit(input_train, result_train)
    joblib.dump(classifier, filename, compress=9)
  print 'Fitting done'
  print score(classifier, input_test, result_test, 'LinearRegression')

def score(classifier, input_test, result_test, title):
  predictions = classifier.predict(input_test)
  print 'Mean absolute error for %s: %.2f' % (title, mean_absolute_error(result_test, predictions))
  print 'R2 Score for %s: %.2f' % (title, r2_score(result_test, predictions))
  print 'Mean Squared error for %s: %.2f' % (title, mean_squared_error(result_test, predictions))
  print 'Explained variance score for %s: %.2f' % (title, explained_variance_score(result_test, predictions))


(input, user_quality) = convertToInputDict()
print 'Number of users:', len(user_quality.keys())
print 'Number of words:', len(input.keys())
(input_vector, result_vector) = vectorize(input, user_quality)
input_train, input_test, result_train, result_test = cross_validation.train_test_split(input_vector, result_vector, test_size=0.7, random_state=0)
print 'Train test data split done'
regressionUsingSVM(input_train, input_test, result_train, result_test)
regressionUsingLinearRegression(input_train, input_test, result_train, result_test)
regressionUsingNonLinearSVM(input_train, input_test, result_train, result_test)