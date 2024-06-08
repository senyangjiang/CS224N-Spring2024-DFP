# Shuffles data in dev output csv files for testing ensemble script for test outputs, 
# where the data is not in the same order as it is in the dev outputs.

import pandas as pd
from sklearn.utils import shuffle

def scramble_data_task(task):
    set = 'dev'

    sbert_filepath = f'sep_model_outputs/sbert/{task}-{set}-output.csv'
    sbertAnnealData_filepath = f'sep_model_outputs/sbert-additional_data-anneal/{task}-{set}-output.csv'
    sbertAnneal_filepath = f'sep_model_outputs/sbert-anneal/{task}-{set}-output.csv' # best performing
    sbertSqrtData_filepath = f'sep_model_outputs/sbert-additional_data-sqrt/{task}-{set}-output.csv'
    sbertSqrt_filepath = f'sep_model_outputs/sbert-sqrt/{task}-{set}-output.csv'

    with open(sbertAnneal_filepath, 'r') as f:
        firstline = f.readline().strip()

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

    sbert = shuffle(sbert)
    sbertAnnealData = shuffle(sbertAnnealData)
    sbertAnneal = shuffle(sbertAnneal)
    sbertSqrtData = shuffle(sbertSqrtData)
    sbertSqrt = shuffle(sbertSqrt)

    sbert_shuffledfilepath = f'sep_model_outputs/shuffled/sbert/{task}-{set}-output.csv'
    sbertAnnealData_shuffledfilepath = f'sep_model_outputs/shuffled/sbert-additional_data-anneal/{task}-{set}-output.csv'
    sbertAnneal_shuffledfilepath = f'sep_model_outputs/shuffled/sbert-anneal/{task}-{set}-output.csv'
    sbertSqrtData_shuffledfilepath = f'sep_model_outputs/shuffled/sbert-additional_data-sqrt/{task}-{set}-output.csv'
    sbertSqrt_shuffledfilepath = f'sep_model_outputs/shuffled/sbert-sqrt/{task}-{set}-output.csv'

    with open(sbert_shuffledfilepath, 'w') as f:
        f.write(firstline + '\n')
    sbert.to_csv(sbert_shuffledfilepath, mode='a', index=False, header=False)

    with open(sbertAnnealData_shuffledfilepath, 'w') as f:
        f.write(firstline + '\n')
    sbertAnnealData.to_csv(sbertAnnealData_shuffledfilepath, mode='a', index=False, header=False)

    with open(sbertAnneal_shuffledfilepath, 'w') as f:
        f.write(firstline + '\n')
    sbertAnneal.to_csv(sbertAnneal_shuffledfilepath, mode='a', index=False, header=False)

    with open(sbertSqrtData_shuffledfilepath, 'w') as f:
        f.write(firstline + '\n')
    sbertSqrtData.to_csv(sbertSqrtData_shuffledfilepath, mode='a', index=False, header=False)

    with open(sbertSqrt_shuffledfilepath, 'w') as f:
        f.write(firstline + '\n')
    sbertSqrt.to_csv(sbertSqrt_shuffledfilepath, mode='a', index=False, header=False)

    print("Done writing shuffled csv files for ", task)

def main():
    scramble_data_task('para')
    scramble_data_task('sst')
    scramble_data_task('sts')

if __name__ == "__main__":
    main()