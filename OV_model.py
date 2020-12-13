"""
File name:     OV_model.py
Authors:       Maximilien Fathi
Date:          December 13, 2020
Description:   Code used for the creation of a Naive Bayes classifier using a non-filtered vocabulary.
"""

from NB_classifier import NB_classifier
from collections import Counter

class OV_model(NB_classifier):
    """A class that represents the Naive Bayes classifier using the original vocabulary."""
    
    def __init__(self, train_df):
        """Initializes the attributes of an OV_model object."""
        super().__init__(train_df)
            
    def create_vocabulary(self):
        """Creates original vocabulary including frequencies."""
        self.vocabulary = Counter()
        self.df["text"].str.lower().str.split().apply(self.vocabulary.update)
        self.vocabulary_size = len(list(self.vocabulary))
        # print(self.vocabulary)
        print(f"The size of the original vocabulary is {self.vocabulary_size}")
        