# Ensembles outputs from multiple models for each task (para, sst, sts)
# and writes to a new csv file

import pandas as pd
from sklearn.utils import shuffle

NUM_MODELS = 5
SHUFFLED_DEV = False

# task is a string like 'para', 'sst', 'sts'
# dicts is a list of dictionaries from the datasets for that task
# firstline is a string for the headers for that task
def ensemble(task, set, dicts, firstline):

    new_dict = get_avg(task, dicts)

    print("Ensemble complete, writing to csv file")
    newfilepath = f'sep_model_outputs/ensemble-{set}/{task}-{set}-output.csv'
    print("Writing to ", newfilepath)
    with open(newfilepath, 'w') as f:
        f.write(firstline + '\n')
    
    df = pd.DataFrame(list(new_dict.items()))
    df.to_csv(newfilepath, mode='a', index=False, header=False)

    print("Done writing to csv file for ", task)

def get_avg(task, dicts):
    sbert_dict = dicts[0]
    sbertAnnealData_dict = dicts[1]
    sbertAnneal_dict = dicts[2]
    sbertSqrtData_dict = dicts[3]
    sbertSqrt_dict = dicts[4]
    
    new_dict = {}
    for key in dicts[0]:
        if (NUM_MODELS == 5):
            new_dict[key] = (sbert_dict[key] + sbertAnnealData_dict[key] + sbertAnneal_dict[key] + sbertSqrtData_dict[key] + sbertSqrt_dict[key]) / NUM_MODELS
        elif (NUM_MODELS == 3):
            new_dict[key] = (sbertAnneal_dict[key] + sbertAnnealData_dict[key] + sbertSqrtData_dict[key]) / NUM_MODELS
        
        if task == 'para' or task == 'sst':
            new_dict[key] = round(new_dict[key])

    return new_dict

def read_csv_files(set):
    para_dicts, para_firstline = read_csv('para', set)
    sst_dicts, sst_firstline = read_csv('sst', set)
    sts_dicts, sts_firstline = read_csv('sts', set)

    firstlines = [para_firstline, sst_firstline, sts_firstline]
    return para_dicts, sst_dicts, sts_dicts, firstlines

def read_csv(task, set):
    if not SHUFFLED_DEV:
        sbert_filepath = f'sep_model_outputs/sbert/{task}-{set}-output.csv'
        sbertAnnealData_filepath = f'sep_model_outputs/sbert-additional_data-anneal/{task}-{set}-output.csv'
        sbertAnneal_filepath = f'sep_model_outputs/sbert-anneal/{task}-{set}-output.csv' # best performing
        sbertSqrtData_filepath = f'sep_model_outputs/sbert-additional_data-sqrt/{task}-{set}-output.csv'
        sbertSqrt_filepath = f'sep_model_outputs/sbert-sqrt/{task}-{set}-output.csv'
        print("sbert_filepath: ", sbert_filepath)
        print("sbertAnnealData_filepath: ", sbertAnnealData_filepath)
        print("sbertAnneal_filepath: ", sbertAnneal_filepath)
        print("sbertSqrtData_filepath: ", sbertSqrtData_filepath)
        print("sbertSqrt_filepath: ", sbertSqrt_filepath)
    elif SHUFFLED_DEV:
        sbert_filepath = f'sep_model_outputs/shuffled/sbert/{task}-{set}-output.csv'
        sbertAnnealData_filepath = f'sep_model_outputs/shuffled/sbert-additional_data-anneal/{task}-{set}-output.csv'
        sbertAnneal_filepath = f'sep_model_outputs/shuffled/sbert-anneal/{task}-{set}-output.csv' # best performing
        sbertSqrtData_filepath = f'sep_model_outputs/shuffled/sbert-additional_data-sqrt/{task}-{set}-output.csv'
        sbertSqrt_filepath = f'sep_model_outputs/shuffled/sbert-sqrt/{task}-{set}-output.csv'

    with open(sbertAnneal_filepath, 'r') as f:
        firstline = f.readline().strip()
    print("firstline: ", firstline)

    sbert = pd.read_csv(sbert_filepath, sep='\t', nrows=0).columns
    sbert = pd.read_csv(sbert_filepath, names=sbert, skiprows=1)

    sbertAnnealData = pd.read_csv(sbertAnnealData_filepath, sep='\t', nrows=0).columns
    sbertAnnealData = pd.read_csv(sbertAnnealData_filepath, names=sbertAnnealData, skiprows=1)

    sbertAnneal = pd.read_csv(sbertAnneal_filepath, sep='\t', nrows=0).columns
    sbertAnneal = pd.read_csv(sbertAnneal_filepath, names=sbertAnneal, skiprows=1)

    sbertSqrtData = pd.read_csv(sbertSqrtData_filepath, sep='\t', nrows=0).columns
    sbertSqrtData = pd.read_csv(sbertSqrtData_filepath, names=sbertSqrtData, skiprows=1)
    
    sbertSqrt = pd.read_csv(sbertSqrt_filepath, sep='\t', nrows=0).columns
    sbertSqrt = pd.read_csv(sbertSqrt_filepath, names=sbertSqrt, skiprows=1)
    
    dfs = [sbert, sbertAnnealData, sbertAnneal, sbertSqrtData, sbertSqrt]
    dicts = [df.set_index(df.columns[0]).iloc[:, 0].to_dict() for df in dfs]
    return dicts, firstline


# main to run this file
def main():
    set = 'test'
    para_dicts, sst_dicts, sts_dicts, firstlines = read_csv_files(set)
    ensemble('para', set, para_dicts, firstlines[0])
    ensemble('sst', set, sst_dicts, firstlines[1])
    ensemble('sts', set, sts_dicts, firstlines[2])

if __name__ == "__main__":
    main()

