import math
import re

def extract_word(s):
  regex = r"email_body_term\((.+),\s\'(.+)\',(.+)\)"
  p = re.compile(regex)
  g = p.match(s)
  if g != None and len(g.groups()) > 2:
    return g.group(2)
  else:
    return None

def get_words(file):
  words = []
  with open(file, 'r') as f:
      for line in f:
        w = extract_word(line)
        if w!= None:
          words.append(w)
  return words

def process_files(filelist):
  word_map = {}
  total_word_count = 0
  
  for file in filelist:
    for w in get_words(file):
      if w in word_map:
        word_map[w] = word_map[w] + 1
      else:
        word_map[w] = 1
      total_word_count += 1
  return word_map, total_word_count

def predict(test_files, ham_map, total_ham_words, spam_map, total_spam_words, alpha=1, english_vocab_size=200000, smooth=True):
  predictions = {}
  unseen_ham_score  = float(alpha) / (total_ham_words  + alpha * english_vocab_size)
  unseen_spam_score = float(alpha) / (total_spam_words + alpha * english_vocab_size)
  for file in test_files:
    # Oops, should have written more generalizable code.
    # Instead of 0.5, should be #type / #total
    ham_score = math.log(.5)
    spam_score = math.log(.5)
    for w in get_words(file):
      
      if w in ham_map:
        # if not smooth
        ham_score  += math.log( float(ham_map[w])  / total_ham_words )
        # if smooth, then numerator needs a "+alpha" and denominator needs a "+alpha*english_vocab_size"
      elif smooth:
        ham_score  += math.log(unseen_ham_score)
      
      if w in spam_map:
        spam_score += math.log( float(spam_map[w]) / total_spam_words )
      elif smooth:
        spam_score  += math.log(unseen_spam_score)
      
    if spam_score > ham_score:
      predictions[file] = 'Spam'
    else:
      predictions[file] = 'Ham'
  return predictions

def summary(predictions, truths):
  num_correct = 0
  tp = 0
  tn = 0
  fp = 0
  fn = 0
  num_total = len(predictions)
  for p in predictions:
    if predictions[p] == truths[p]:
      num_correct += 1
      if predictions[p] == 'Spam':
        tp += 1
      else:
        tn += 1
    else:
      if predictions[p] == 'Spam':
        fp += 1
      else:
        fn += 1

  precision = float(tp) / (tp + fp)
  recall = float(tp) / (tp + fn)
  fscore = 2 / float( 1/precision + 1/recall)
  accuracy = float(tp + tn) / num_total

  return tp, tn, fp, fn, precision, recall, fscore, accuracy

def get_truths(truth_file):
  truths = {}
  with open(truth_file, 'r') as f:
    for line in f:
      s = line.index(' ')
      fn = line[:s]
      if 'Spam' in line:
        truths[fn] = 'Spam'
      if 'Ham' in line:
        truths[fn] = 'Ham'
  return truths
