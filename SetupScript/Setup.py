import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as f
import base_solution as bs
import sys as sys


#Defining working directory
base_dir = "./"
# Defining all the solutions implemented
solutions = {
    "basesolution": bs.get_rec_base
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

chosen_solution = sys.argv[3]
print("Executing the solution " + chosen_solution)
f.send_telegram_message("Starting algorithm " + chosen_solution)

#Ground Truth
df_gt = df_test.copy()

#Exporting Ground Truth for future usage
df_gt.to_csv(base_dir + "gt.csv", index = False)

#Set test reference values to NULL for clickout items as we must not know what will be clicked
df_test.loc[df_test["action_type"] == "clickout item", ["reference"]] = None

#df_rec = SOLUTION FUNCTION
#Computing recommendation file

df_rec = solutions[chosen_solution](base_dir, df_train, df_test)

#Computing score
gt_csv = base_dir + "gt.csv"
subm_csv = base_dir + "submission_popular.csv"
mrr = f.score_submissions(subm_csv, gt_csv, f.get_reciprocal_ranks)
f.send_telegram_message("End execution with score " + str(mrr))

#print(f'Mean reciprocal rank: {mrr}')
