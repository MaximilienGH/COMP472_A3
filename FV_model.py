"""
File name:     FV_model.py
Authors:       Maximilien Fathi
Date:          December 13, 2020
Description:   Code used for the creation of a Naive Bayes classifier using a filtered vocabulary
"""

from NB_classifier import NB_classifier
from collections import Counter

class FV_model(NB_classifier):
    """A class that represents the Naive Bayes classifier using the filtered vocabulary."""
    
    def __init__(self, train_df):
        """Initializes the attributes of an OV_model object."""
        super().__init__(train_df)
            
    def create_vocabulary(self):
        """Creates filtered vocabulary including frequencies."""
        self.vocabulary = Counter()
        self.df["text"].str.lower().str.split().apply(self.vocabulary.update)
        self.vocabulary = {k:v for k, v in self.vocabulary.items() if v >= 2}
        self.vocabulary = Counter(self.vocabulary)
        self.vocabulary_size = len(list(self.vocabulary))
        # print(self.vocabulary)
        print(f"The size of the filtered vocabulary is {self.vocabulary_size}")
        