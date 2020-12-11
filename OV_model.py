"""
File name:     OV_model.py
Authors:       Maximilien Fathi
Date:          December 13, 2020
Description:   
"""

from NB_classifier import *
from collections import Counter

class OV_model(NB_classifier):
    """A class that represents the Naive Bayes classifier using the original vocabulary."""
    
    def __init__(self, train_df):
        """Initialize the attributes of an OV_model object."""
        super().__init__(train_df)
            
    def create_original_vocab(self):
        """Creates original vocabulary including frequencies"""
        self.vocabulary = Counter()
        self.df[1].str.lower().str.split().apply(self.vocabulary.update)
        self.vocabulary_size = len(list(self.vocabulary))
        # print(self.NB_BOW_OV)
        print(f"The size of the vocabulary is {self.vocabulary_size}") # will check again if correct
        