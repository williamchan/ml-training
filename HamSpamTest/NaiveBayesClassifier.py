import os
import re
import math

root_path = '/Users/skshetra/Workspace/AI_academy/data/HamSpamTest'
ham_path = root_path + '/ham'
spam_path = root_path + '/spam'
test_path = root_path + '/test'

# constants
alpha = 1
vocabularySize = 200000

# returns words (features) from a file
def extractFeaturesFromFile(file_path):
    delimiters = "(", ")", " ", ",", "'", "\n", "."
    regexPattern = '|'.join(map(re.escape, delimiters))

    features = []
    f = open(file_path, "r")
    for line in f:
        tokens = re.split(regexPattern, line)
        # skip from and to lines
        if (tokens[0] == 'email_from' or tokens[0] == 'email_to'):
            continue
        # for some reason 5th token is the word
        if (len(tokens) >= 5):
            features.append(tokens[4])
        
    return features

# process training data files uder path
# construct a dictionary of word counts
def processTrainingDataFiles(path, data_dict):
    totalEmails = 0
    totalWords = 0 
    for filename in os.listdir(path):
        totalEmails += 1
        # process data from each file
        f = path+'/'+filename
        words = extractFeaturesFromFile(f)           
        totalWords += len(words)
        for word in words:
            if (word in data_dict):
                data_dict[word] = data_dict[word] +1
            else:
                data_dict[word] = 1
    return totalEmails, totalWords
            
# compute class log probablity using formula: log(prior) + SUM (log(fp))
def computeClassProbablity(prior, smoothed_dict, unseen_feature_prob, features):
    
    prob = math.log(prior)    
    for f in features:
        if f in smoothed_dict:
            prob += math.log(smoothed_dict[f])
        else:
            prob += math.log(unseen_feature_prob)
    
    return prob
        
    
# classify: Given a list of features returns a class
def classify(features):
    
    global alpha, totalHamEmails, totalSpamEmails, smoothing_denominator_ham, smoothing_denominator_spam
        
    # compute Ham probability
    hamPrior = totalHamEmails / float(totalHamEmails + totalSpamEmails)
    ham_unseen_feature_prob = alpha / float(smoothing_denominator_ham)
    hamProb = computeClassProbablity(hamPrior, smoothed_ham_dict, ham_unseen_feature_prob, features)
    
    # compute Spam probability
    spamPrior = totalSpamEmails / float(totalHamEmails + totalSpamEmails)
    spam_unseen_feature_prob = alpha / float(smoothing_denominator_spam)
    spamProb = computeClassProbablity(spamPrior, smoothed_spam_dict, spam_unseen_feature_prob, features)
    
    if (hamProb > spamProb):
        return "Ham"
    else:
        return "Spam"


# Ham data variables
totalHamEmails = 0
totalHamWords = 0
ham_dict = {}
smoothed_ham_dict = {}

# Spam data variables
totalSpamEmails = 0
totalSpamWords = 0
spam_dict = {}
smoothed_spam_dict = {}

# process Ham training data
totalHamEmails, totalHamWords = processTrainingDataFiles(ham_path, ham_dict)

# apply smoothing
smoothing_denominator_ham = totalHamWords + (alpha * vocabularySize)
for key in ham_dict:
    smoothed_ham_dict[key] = (float)(ham_dict[key] + alpha) / smoothing_denominator_ham

# process Spam training data
totalSpamEmails, totalSpamWords = processTrainingDataFiles(spam_path, spam_dict)

# apply smoothing
smoothing_denominator_spam = totalSpamWords + (alpha * vocabularySize)
for key in spam_dict:
    smoothed_spam_dict[key] = (float)(spam_dict[key] + alpha) / smoothing_denominator_spam
    

# Read the true labels for Test data
truth_table = {}
f_truth = open(test_path+'/'+'truth', "r")
for line in f_truth:
    tokens = re.split(' |\/|\n', line)
    truth_table[tokens[1]] = tokens[2] # key: filename, value: label
       

# P - Spam, N - Ham
TP = 0
FP = 0
TN = 0
FN = 0

# Classify Test data
for filename in os.listdir(test_path):
    
    if (filename == 'truth'):
        continue
        
    # process data from each test file
    f = test_path+'/'+filename
    words = extractFeaturesFromFile(f)
    category = classify(words)
    if (category == 'Spam'):
        if (category == truth_table[filename]):
            TP += 1
        else:
            FP += 1
    else:
        if (category == truth_table[filename]):
            TN += 1
        else:
            FN += 1

Precision = TP/float(TP+FP)
Recall = TP/float(TP+FN)
F_score = 2 * Precision * Recall / float(Precision+Recall)
Accuracy = (TP+TN)/float(TP+FP+TN+FN)

print 'Precision: ', Precision
print 'Recall: ', Recall
print 'F_score :', F_score
print 'Accuracy :', Accuracy



