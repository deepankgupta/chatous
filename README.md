chatous
=======

This contains project work which uses chatous.com dataset for predicting quality of users and helps in predicting a better matching algorithm.

There are three different source files:

1. kmeans.py : This does the analysis of the dataset and tries to cluster the profiles into categories based on age and gender.

2. classification.py : This runs through various different classification algorithms and applies them on the dataset to find bad users based on their conversation patterns.

3. regression.py : This applies regression algorithms on conversation word vectors to predict the length of the conversation per user.


User Statistics
---------------
We ran a k-means clustering algorithm on the profile data keeping age and gender as variables and got the table: ![Cluster Table](results/cluster.png) 
Specifically males outnumber females in the network and the demographics also point to more teenage boys than girls in the networks. 

Some other statistics about the user profiles based on their country are given here:
![Country distribution](results/country_dist.png). 
This points out that the demographic for the network is global with most number of members either in US, India or UK.


User Classification
-------------------
We got a hand-made manual set of 1034 users which were classified as either clean, dirty or bot. Using this data set and dividing it into training and test set; we tried to apply various classification algorithms in order to find out if we could learn to predict dirty/bot users and penalize them in the chat matching algorithm. We tried various different algorithms starting with k-neighbours to a pipeline combining more than one algorithm.

In the data set, the distribution is quite skewed; so instead of using the normal accuracy measure for calculating an algorithm's effectiveness, we used the Precision-Recall curves. We calculated the following metrics for every algorithm:

1. Precision recall curves

2. Number of false positives, false negatives, true positives, true negatives

3. Precision, Recall and F-score.

All the data about the different algorithms is present in the file: results/classification. The various plots are:

K-Means Neighbour Precision Recall

![K Means Neighbour](results/KNeighboursClassifierPrecisionRecall.png)

Multinomial Naive Bayes Precision Recall

![Multinomial Naive Bayes](results/MultinomialNaiveBayesPrecisionRecall.png)

Support Vector Machine Precision Recall

![Support Vector Machine](results/SVCPrecisionRecall.png)

Pipeline Precision Recall

![Pipeline Precision Recall](results/PipelinePrecisionRecall.png)

User Quality Regression
-----------------------
Besides this the other key parameter in matching users will be the quality of the user. The quality of the user is defined as the average length of conversations that a user does. If the user starts and maintains longer conversations, the user will get a better score. For doing this we also eliminated users who have not done a single conversation yet and were thus left with over 39727 users.

The signal that we used for learning was the word vectors that they had spoken till now. Based on the word vectors of their previous conversations till now we tried to learn a model with 70% of the user set and then use that to predict the quality on the rest of the 30% user base. We tried three different models: linear regresison, linear SVC and non-linear SVC with a exponential kernel.

The results were pretty much as expected with the non-linear support vector machine with exponential kernel performing really well. All the results are documented in results/regression.
