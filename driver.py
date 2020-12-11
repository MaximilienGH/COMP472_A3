"""
File name:     driver.py
Authors:       Maximilien Fathi
Date:          December 13, 2020
Description:   
"""

import data_export
from OV_model import *
from FV_model import *
import pandas as pd

train_df = pd.read_csv("covid_training.tsv", sep='\t', header=None)
test_df = pd.read_csv("covid_test_public.tsv", sep='\t', header=None)

OV_clf = OV_model(train_df)
OV_clf.create_original_vocab()
OV_clf.group_classes()
OV_clf.count_words()
OV_clf.create_probabilities_dict()
data_export.create_trace_file(test_df, OV_clf, "OV")
data_export.create_evaluation_file(test_df, OV_clf, "OV")

FV_clf = FV_model(train_df)
FV_clf.create_filtered_vocab()
FV_clf.group_classes()
FV_clf.count_words()
FV_clf.create_probabilities_dict()
data_export.create_trace_file(test_df, FV_clf, "FV")
data_export.create_evaluation_file(test_df, FV_clf, "FV")
