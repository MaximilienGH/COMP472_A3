"""
File name:     data_export.py
Authors:       Maximilien Fathi
Date:          December 13, 2020
Description:   Code used for the creation of trace files and evaluation files.
"""

def create_trace_file(test_df, clf, vocabulary):
    """Generates a trace file for a model."""
    output_file = f"Output_Files/trace_NB-BOW-{vocabulary}.txt"
    with open(output_file, 'w') as file_object:
        for index, row in test_df.iterrows():
            results = clf.find_best_score(row[1], row[2])
            file_object.write(str(row[0]) + "  " + results[0] + "  " + "{:.3e}".format(results[1])
                              + "  " + row[2] + "  " + results[2] + "\n")
            
def create_evaluation_file(test_df, clf, vocabulary):
    """Generates an evaluation file for a model."""
    output_file = f"Output_Files/eval_NB-BOW-{vocabulary}.txt"
    with open(output_file, 'w') as file_object:
        line1 = "{:.4f}".format(clf.get_accuracy()) + "\n"
        line2 = "{:.4f}".format(clf.get_precision("yes")) + "  " + "{:.4f}".format(clf.get_precision("no")) + "\n"
        line3 = "{:.4f}".format(clf.get_recall("yes")) + "  " + "{:.4f}".format(clf.get_recall("no")) + "\n"
        line4 = "{:.4f}".format(clf.get_f1("yes")) + "  " + "{:.4f}".format(clf.get_f1("no")) + "\n" 
        file_object.write(line1 + line2 + line3 + line4)
