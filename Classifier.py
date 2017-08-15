import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import Word2Vec
import gensim

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.
train = pd.read_csv( "/home/vivek/Desktop/Kaggle/Sentiment Analysis/labeledTrainData.tsv", header=0,
 delimiter="\t", quoting=3 )
test = pd.read_csv( "/home/vivek/Desktop/Kaggle/Sentiment Analysis/testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "/home/vivek/Desktop/Kaggle/Sentiment Analysis/unlabeledTrainData.tsv", header=0,
 delimiter="\t", quoting=3 )
model = models = gensim.models.Word2Vec.load('300features_40minwords_10context')
clean_train_reviews = []

num_features=300
for review in train["review"]:
    clean_train_reviews.append( Word2Vec.review_to_wordlist( review,remove_stopwords=True ))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )

print "Creating average feature vecs for test reviews"
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( Word2Vec.review_to_wordlist( review,remove_stopwords=True ))
testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )


# Fit a random forest to the training data, using 100 trees

forest = RandomForestClassifier( n_estimators = 100 )

print "Fitting a random forest to labeled training data..."
forest = forest.fit( trainDataVecs, train["sentiment"] )

# Test & extract results
result = forest.predict( testDataVecs )

# Write the test results
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3)