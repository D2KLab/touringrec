import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as f
import base_solution as bs
import matrix_factorization_metadata as mf_metadata
import matrix_factorization as mf
import sys as sys

#Defining working directory
base_dir = "./"
# Defining all the solutions implemented
solutions = {
    "basesolution": bs.get_rec_base,
    "basesolution_nation": bs.get_rec_nation,
    "matrixfactorization_metadata": mf_metadata.get_rec_matrix,
    "matrixfactorization": mf.get_rec_matrix
}

#importing and defining Datasets

#Train
filename = sys.argv[1]
print("Reading train set " + filename)
df_train = pd.read_csv(filename)

#Test
filename = sys.argv[2]
print("Reading test set " + filename)
df_test = pd.read_csv(filename)

gt_csv = sys.argv[3]
print("Groud truth is: " + filename)

metadata_csv = sys.argv[4]
print("The metadata file is: " + filename)

chosen_solution = sys.argv[5]
print("Executing the solution " + chosen_solution)

local_score = True
#df_rec = SOLUTION FUNCTION
#Computing recommendation file
#weights = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 1]
weights = [0.01]
for i in weights:
    df_rec = solutions[chosen_solution](df_train, df_test, file_metadata = metadata_csv, w_nation = 1, w_base = i)
    #Computing score
    subm_csv = "submission_popular.csv"
    if(local_score):
        mrr = f.score_submissions(subm_csv, gt_csv, f.get_reciprocal_ranks)
        print("End execution with score " + str(mrr))
        f.send_telegram_message("End execution with score " + str(mrr))


#print(f'Mean reciprocal rank: {mrr}')
