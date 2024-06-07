# read in csv files into DataFrames

import pandas as pd

NUM_MODELS = 5

# read in csv files
def read_csv_files():
    # para dev
    sbert_para = pd.read_csv('sep_model_outputs/sbert/para-dev-output.csv')
    sbertAnnealData_para = pd.read_csv('sep_model_outputs/sbert-additional_data-anneal/para-dev-output.csv')
    sbertAnneal_para = pd.read_csv('sep_model_outputs/sbert-anneal/para-dev-output.csv')
    sbertSqrtData_para = pd.read_csv('sep_model_outputs/sbert-additional_data-sqrt/para-dev-output.csv')
    sbertSqrt_para = pd.read_csv('sep_model_outputs/sbert-sqrt/para-dev-output.csv')
    para_dev = [sbert_para, sbertAnnealData_para, sbertAnneal_para, sbertSqrtData_para, sbertSqrt_para]
    para_dev_df = pd.read_csv(para_dev, sep='\t', header=None)

    # sst dev
    sbert_sst = pd.read_csv('sep_model_outputs/sbert/sst-dev-output.csv')
    sbertAnnealData_sst = pd.read_csv('sep_model_outputs/sbert-additional_data-anneal/sst-dev-output.csv')
    sbertAnneal_sst = pd.read_csv('sep_model_outputs/sbert-anneal/sst-dev-output.csv')
    sbertSqrtData_sst = pd.read_csv('sep_model_outputs/sbert-additional_data-sqrt/sst-dev-output.csv')
    sbertSqrt_sst = pd.read_csv('sep_model_outputs/sbert-sqrt/sst-dev-output.csv')
    sst_dev = [sbert_sst, sbertAnnealData_sst, sbertAnneal_sst, sbertSqrtData_sst, sbertSqrt_sst]
    sst_dev_df = pd.read_csv(sst_dev, sep='\t', header=None)

    # sts dev
    sbert_sts = pd.read_csv('sep_model_outputs/sbert/sts-dev-output.csv')
    sbertAnnealData_sts = pd.read_csv('sep_model_outputs/sbert-additional_data-anneal/sts-dev-output.csv')
    sbertAnneal_sts = pd.read_csv('sep_model_outputs/sbert-anneal/sts-dev-output.csv')
    sbertSqrtData_sts = pd.read_csv('sep_model_outputs/sbert-additional_data-sqrt/sts-dev-output.csv')
    sbertSqrt_sts = pd.read_csv('sep_model_outputs/sbert-sqrt/sts-dev-output.csv')
    sts_dev = [sbert_sts, sbertAnnealData_sts, sbertAnneal_sts, sbertSqrtData_sts, sbertSqrt_sts]

    print("Read in csv files")

    return para_dev, sst_dev, sts_dev

def ensemble(para_df, sst_df, sts_df):
    print("Ensembling")

    print("para_df[0] shape: ", para_df[0].shape)
    print("sst_df[0] shape: ", sst_df[0].shape)
    print("sts_df[0] shape: ", sts_df[0].shape)
    
    print("para_df[0] head: ", para_df[0].head())
    print("sst_df[0] head: ", sst_df[0].head())
    print("sts_df[0] head: ", sts_df[0].head())

    sbert_para_pred = para_df[0].iloc[:, 2]
    sbertAnnealData_para_pred = para_df[1].iloc[:, 2]
    sbertAnneal_para_pred = para_df[2].iloc[:, 2]
    sbertSqrtData_para_pred = para_df[3].iloc[:, 2]
    sbertSqrt_para_pred = para_df[4].iloc[:, 2]

    sbert_sst_pred = sst_df[0].iloc[:, 2]
    sbertAnnealData_sst_pred = sst_df[1].iloc[:, 2]
    sbertAnneal_sst_pred = sst_df[2].iloc[:, 2]
    sbertSqrtData_sst_pred = sst_df[3].iloc[:, 2]
    sbertSqrt_sst_pred = sst_df[4].iloc[:, 2]

    sbert_sts_pred = sts_df[0].iloc[:, 2]
    sbertAnnealData_sts_pred = sts_df[1].iloc[:, 2]
    sbertAnneal_sts_pred = sts_df[2].iloc[:, 2]
    sbertSqrtData_sts_pred = sts_df[3].iloc[:, 2]
    sbertSqrt_sts_pred = sts_df[4].iloc[:, 2]

    # ensemble
    para_pred = (sbert_para_pred + sbertAnnealData_para_pred + sbertAnneal_para_pred + sbertSqrtData_para_pred + sbertSqrt_para_pred) / NUM_MODELS
    para_pred = para_pred.round()

    sst_pred = (sbert_sst_pred + sbertAnnealData_sst_pred + sbertAnneal_sst_pred + sbertSqrtData_sst_pred + sbertSqrt_sst_pred) / NUM_MODELS
    sst_pred = sst_pred.round()

    sts_pred = (sbert_sts_pred + sbertAnnealData_sts_pred + sbertAnneal_sts_pred + sbertSqrtData_sts_pred + sbertSqrt_sts_pred) / NUM_MODELS

    # write to new csv, keeping first two columns, third column is ensembled predictions
    new_para = para_df[0].iloc[:, :2]
    new_para.insert(2, '', para_pred)

    new_sst = sst_df[0].iloc[:, :2]
    new_sst.insert(2, '', sst_pred)

    new_sts = sts_df[0].iloc[:, :2]
    new_sts.insert(2, '', sts_pred)

    print("Ensemble complete, writing to csv files")
    new_para.to_csv('sep_model_outputs/ensemble-dev/para-dev-output.csv', index=False)
    new_sst.to_csv('sep_model_outputs/ensemble-dev/sst-dev-output.csv', index=False)
    new_sts.to_csv('sep_model_outputs/ensemble-dev/sts-dev-output.csv', index=False)
    print("Done writing to csv files")




# main to run this file
def main():
    para_dev, sst_dev, sts_dev = read_csv_files()
    ensemble(para_dev, sst_dev, sts_dev)

if __name__ == "__main__":
    main()

