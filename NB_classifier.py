"""
File name:     NB_classifier.py
Authors:       Maximilien Fathi
Date:          December 13, 2020
Description:   Code used for the creation of a Naive Bayes classifier.
"""

from collections import Counter
from math import log10

class NB_classifier():
    """A class that represents the Naive Bayes classifier."""
    
    def __init__(self, train_df):
        """Initialize the attributes of a NB_classifier object."""
        self.df = train_df
        self.vocabulary = None
        self.vocabulary_size = 0
        self.no_df = None
        self.no_counter = None
        self.words_in_no = 0
        self.docs_in_no = 0
        self.yes_df = None
        self.yes_counter = None
        self.words_in_yes = 0
        self.docs_in_yes = 0
        self.probabilities_dict = {}
        self.SMOOTHING = 0.01
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        
    def group_classes(self):
        """Divides original dataframe into yes/no classes."""
        grouped = self.df.groupby("q1_label")
        self.no_df = grouped.get_group("no")
        # print(self.no_df)
        self.docs_in_no = len(self.no_df.index)
        self.yes_df = grouped.get_group("yes")
        # print(self.yes_df)
        self.docs_in_yes = len(self.yes_df.index)
        print(f"number of docs in no is {self.docs_in_no} and in yes is {self.docs_in_yes}")
        
    def count_words(self):
        """Counts number of words in each class."""
        self.no_counter = Counter()
        self.no_df["text"].str.lower().str.split().apply(self.no_counter.update)
        self.words_in_no = sum(self.no_counter.values())
        self.yes_counter = Counter()
        self.yes_df["text"].str.lower().str.split().apply(self.yes_counter.update)
        self.words_in_yes = sum(self.yes_counter.values())
        print(f"number of words in no is {self.words_in_no} and in yes is {self.words_in_yes}")
        
    def create_probabilities_dict(self):
        """Creates a dictionary with vocabulary words as keys and probabilities for yes/no as values."""
        for word in self.vocabulary.keys():
            self.probabilities_dict[word] = [self.find_probability(word, "no")]
            self.probabilities_dict[word].append(self.find_probability(word, "yes"))
        # print(self.probabilities_dict)
            
    def find_probability(self, word, label):
        """Finds probability of a word given a label (i.e. class)."""
        if label == "no":
            return (self.no_counter[word] + self.SMOOTHING) / (self.words_in_no + self.SMOOTHING * self.vocabulary_size)
        else:
            return (self.yes_counter[word] + self.SMOOTHING) / (self.words_in_yes + self.SMOOTHING *  self.vocabulary_size)
            
    def find_no_score(self, tweet):
        """Determines the score for the 'no' class."""
        words_list = tweet.split()
        score = log10(self.docs_in_no / (self.docs_in_no + self.docs_in_yes))
        for word in words_list:
            if word not in self.probabilities_dict.keys():
                continue
            else:
                word_probability = self.probabilities_dict[word][0] 
                score += log10(word_probability)
        return score
                
    def find_yes_score(self, tweet):
        """Determines the score for the 'yes' class."""
        words_list = tweet.split()
        score = log10(self.docs_in_yes / (self.docs_in_no + self.docs_in_yes))
        for word in words_list:
            if word not in self.probabilities_dict.keys():
                continue
            else:
                word_probability = self.probabilities_dict[word][1] 
                score += log10(word_probability)
        return score
                
    def find_best_score(self, tweet, true_class):
        """Determines best score out of yes/no scores for a tweet."""
        no_score = self.find_no_score(tweet)
        yes_score = self.find_yes_score(tweet)
        if no_score >= yes_score:
            return ("no", no_score, self.is_correct("no", true_class))
        else:
            return ("yes", yes_score, self.is_correct("yes", true_class))
        
    def is_correct(self, prediction, true_class):
        """Determines if predicted class is correct or not."""
        if true_class == "yes" and prediction == "yes":
            self.true_positives += 1
            return "correct"
        elif true_class == "no" and prediction == "no":
            self.true_negatives += 1
            return "correct"
        elif true_class == "yes" and prediction == "no":
            self.false_negatives += 1
            return "wrong"
        elif true_class == "no" and prediction == "yes":
            self.false_positives += 1
            return "wrong"
        
    def get_accuracy(self):
        """Returns the accuracy of the model using (TP + TN) / (TP + TN + FP + FN)."""
        number_of_instances = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        print(f"The total numer of test data is {number_of_instances}\n")
        return (self.true_positives + self.true_negatives) / number_of_instances
        
    def get_precision(self, label):
        """Returns the precision of the model using TP / (TP + FP) for the yes class
        and TN / (TN + FN) for the no class."""
        # print("TP", self.true_positives)
        # print("TN", self.true_negatives)
        # print("FP", self.false_positives)
        # print("FN", self.false_negatives)
        if label == "yes":
            return self.true_positives / (self.true_positives + self.false_positives)
        else:
            return  self.true_negatives / (self.true_negatives + self.false_negatives)
    
    def get_recall(self, label):
        """Returns the recall value of the model using TP / (TP + FN) for the yes class
        and TN / (TN + FP) for the no class."""
        if label == "yes":
            return self.true_positives / (self.true_positives + self.false_negatives)
        else:
            return  self.true_negatives / (self.true_negatives + self.false_positives)
        
    def get_f1(self, label):
        """Returns the F1-measure of the model assuming beta is 1 for this project."""
        beta = 1
        precision = self.get_precision(label)
        recall = self.get_recall(label)
        return ((beta**2 + 1) * precision * recall) / ((beta**2) * precision + recall)
