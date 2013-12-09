import ast
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
import pylab as pl
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def convertToTargetDict(lines):
  target_enum = {'clean': 0, 'dirty': 1, 'bot': 1}
  target_dict = {}
  for line in lines:
    words = map(lambda x: x.strip(), line.split(';'))
    target_dict[words[0]] = target_enum[words[1]]
    assert len(words) == 2
  return target_dict

def insertInTrainingDict(data, user, words):
  if user not in data:
    data[user] = {}
  word_dict = ast.literal_eval(words)
  for word in word_dict:
    if word not in data[user]:
      data[user][word] = word_dict[word]
    else:
      data[user][word] += word_dict[word]



def convertToInputDict(target_dict):
  f = open('../data/new_chats_dataset.csv')
  training_dict = {}
  for line in f.readlines():
    words = line.split(';')
    assert len(words) == 14, words
    if words[1] in target_dict:
      insertInTrainingDict(training_dict, words[1], words[12])
    if words[2] in target_dict:
      insertInTrainingDict(training_dict, words[2], words[13])
  return training_dict


def vectorize(input_dict, target_dict):
  input_list = []
  target_list = []
  for key in sorted(input_dict.keys()):
    input_list.append(input_dict[key])
    target_list.append(target_dict[key])
  v = DictVectorizer()
  input_vector = v.fit_transform(input_list)
  return (input_vector, target_list)

def classifyUsingDecisionTree(input_train, input_test, result_train, result_test):
  return None  


def classifyAfterFeatureSelectionUsingRandomForest(input_train, input_test, result_train, result_test):
  classifier = Pipeline([
  ('feature_selection', LinearSVC(penalty="l1",dual=False)),
  ('classification', RandomForestClassifier())
  ])
  classifier.fit(input_train.toarray(), result_train)
  score(classifier, input_test.toarray(), result_test, 'Pipeline: LinearSVC feature_selection -> RandomForestClassifier')

def classifyUsingNaiveBayes(input_train, input_test, result_train, result_test):
  classifier = MultinomialNB()
  classifier.fit(input_train, result_train)
  score(classifier, input_test, result_test, 'MultinomialNB')


def classifyUsingSvc(input_train, input_test, result_train, result_test):
  classifier = SVC(probability=True, class_weight='auto', C=2)
  classifier.fit(input_train, result_train)
  score(classifier, input_test, result_test, 'SVC')


def classifyUsingKNeighbour(input_train, input_test, result_train, result_test):
  classifier = KNeighborsClassifier(n_neighbors=5)
  classifier.fit(input_train, result_train)  
  score(classifier, input_test, result_test, 'KNeighborsClassifier')

def score(classifier, input, result, title):
  print 'For %s' % title
  predictions = classifier.predict_proba(input)
  precision, recall, thresholds = precision_recall_curve(result, predictions[:, 1])
  area = auc(recall, precision)
  print("Area Under Curve for %s: %0.2f" % (title, area))

  pl.clf()
  pl.plot(recall, precision, label='Precision-Recall curve')
  pl.xlabel('Recall')
  pl.ylabel('Precision')
  pl.ylim([0.0, 1.05])
  pl.xlim([0.0, 1.0])
  pl.title('%s AUC=%0.2f' % (title, area))
  pl.legend(loc="lower left")
  pl.show()
  scorePrimitive(classifier, input, result)
  

def scorePrimitive(classifier, input, result):
  predictions = classifier.predict(input)
  (precision, recall, fscore, threshold) = precision_recall_fscore_support(result, predictions, average='micro')
  print 'Precision: %0.2f ; Recall: %0.2f ; F-Score: %0.2f' % (precision, recall, fscore)
  num_true_positives = 0
  num_true_negatives = 0
  num_false_positives = 0
  num_false_negatives = 0
  for i, prediction in enumerate(predictions):
    if prediction == 0:
      if result[i] == 0:
        num_true_negatives += 1
      else:
        num_false_negatives += 1
    else:
      if result[i] == 1:
        num_true_positives += 1
      else:
        num_false_positives += 1
  print 'Num True Positives: ', num_true_positives
  print 'Num False Positives: ', num_false_positives
  print 'Num True Negatives: ', num_true_negatives
  print 'Num False Negatives: ', num_false_negatives

f = open('../data/new_users_dataset.csv')
lines = f.readlines()
result = convertToTargetDict(lines)
input = convertToInputDict(result)

(input_vector, result_vector) = vectorize(input, result)
input_train, input_test, result_train, result_test = cross_validation.train_test_split(input_vector, result_vector, test_size=0.1, random_state=0)

classifyUsingNaiveBayes(input_train, input_test, result_train, result_test)
classifyUsingSvc(input_train, input_test, result_train, result_test)
classifyUsingKNeighbour(input_train, input_test, result_train, result_test)
classifyUsingDecisionTree(input_train, input_test, result_train, result_test)
classifyAfterFeatureSelectionUsingRandomForest(input_train, input_test, result_train, result_test)




