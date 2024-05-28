'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import WeightedRandomSampler


from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask


TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        # MY CODE BEGINS HERE
        self.config = config
        self.num_labels = len(config.num_labels)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        if config.model == 'baseline':
            self.proj_sentiment = torch.nn.Linear(config.hidden_size, self.num_labels) #linear layer for sentiment classification
            self.proj_paraphrase = torch.nn.Linear(config.hidden_size*2, 1) #linear layer for paraphrase classification
            self.proj_similarity = torch.nn.Linear(config.hidden_size*2, 1) #linear layer for similarity classification
        elif config.model == 'sbert':
            self.proj_sentiment = torch.nn.Linear(config.hidden_size, self.num_labels)
            self.proj_paraphrase = torch.nn.Linear(config.hidden_size*3, 2)
        # MY CODE ENDS HERE


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        out = self.bert(input_ids, attention_mask)
        return out # {'last_hidden_state': sequence_output, 'pooler_output': first_tk}


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        out_pooler = self.forward(input_ids, attention_mask)['pooler_output']
        drop_out = self.dropout(out_pooler)
        logits = self.proj_sentiment(drop_out)
        return logits


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        if self.config.model == 'baseline':
            out_pooler_1 = self.forward(input_ids_1, attention_mask_1)['pooler_output']
            out_pooler_2 = self.forward(input_ids_2, attention_mask_2)['pooler_output']
            drop_out = self.dropout(torch.cat((out_pooler_1, out_pooler_2), dim=-1))
            logit = self.proj_paraphrase(drop_out)
        elif self.config.model == 'sbert':
            avg1 = torch.mean(self.forward(input_ids_1, attention_mask_1)['last_hidden_state'], dim=-2)
            avg2 = torch.mean(self.forward(input_ids_2, attention_mask_2)['last_hidden_state'], dim=-2)
            drop_out = self.dropout(torch.cat((avg1, avg2, torch.abs(avg1 - avg2)), dim=-1))
            logit = self.proj_paraphrase(drop_out)
        return logit


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        if self.config.model == 'baseline':
            out_pooler_1 = self.forward(input_ids_1, attention_mask_1)['pooler_output']
            out_pooler_2 = self.forward(input_ids_2, attention_mask_2)['pooler_output']
            drop_out = self.dropout(torch.cat([out_pooler_1, out_pooler_2], dim=-1))
            logit = self.proj_similarity(drop_out)
        elif self.config.model == 'sbert':
            avg1 = torch.mean(self.forward(input_ids_1, attention_mask_1)['last_hidden_state'], dim=-2)
            avg2 = torch.mean(self.forward(input_ids_2, attention_mask_2)['last_hidden_state'], dim=-2)
            logit = F.cosine_similarity(avg1, avg2, dim=-1) # consine similarity is between -1 and 1
            logit = torch.mul(torch.add(logit, 1), 2.5) # scale to between 0 and 5
        return logit




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    #TODO MY EDITS BEGIN HERE
    #For paraphrase task:
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=para_dev_data.collate_fn)

    #For similarity task:
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)
    
    original_dataloaders = [sst_train_dataloader, para_train_dataloader, sts_train_dataloader] #added this for tryna fix things w prop sampling

    #TODO combine into one dataloader
    combined_train_dataloader = CombinedLoader({'sst': sst_train_dataloader, 
                                                'para': para_train_dataloader, 
                                                'sts': sts_train_dataloader}, mode='sequential')
    


    # combined_train_dataloader = CombinedLoader({'sts': sts_train_dataloader,
    #                                             'para': para_train_dataloader,                                                
    #                                             'sst': sst_train_dataloader, 
    #                                             }, mode='sequential')
    # print("First combined train dataloader: ", combined_train_dataloader)
    # TODO might actually not use this if I only need to call the individual dev dataloaders later
    # combined_dev_dataloader = CombinedLoader({'sst': sst_dev_dataloader,
    #                                             'para': para_dev_dataloader,
    #                                             'sts': sts_dev_dataloader}, mode='sequential')
    #TODO MY EDITS END HERE
    


    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode,
              'model': args.model}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0


    # BATCH SAMPLING EXTENSION
    # alpha = 1 - 0.8 * ((epoch - 1) / (args.epochs - 1))
    alpha = 0.5
    # print("Alpha: ", alpha)

    weights = [len(dataloader.dataset)**alpha for dataloader in original_dataloaders]
    # print("Weights computation 1: ", weights)
    weights = [weight / sum(weights) for weight in weights]
    # print("Weights computation 2: ", weights)


    dataloaders = []
    for dataloader, weight in zip(original_dataloaders, weights):
        indices = [i for i in range(len(dataloader.dataset)) for _ in range(int(weight * 1000))]
        # sampler = WeightedRandomSampler(indices, len(indices))
        sampler = WeightedRandomSampler()
        new_dataloader = DataLoader(dataloader.dataset, sampler=sampler)
        dataloaders.append(new_dataloader)

    
    # print("Original dataloaders: ", original_dataloaders)
    # print("Dataloaders: ", dataloaders)

    # print("sst_train_dataloader dataset size: ", len(sst_train_dataloader.dataset))
    # print("para_train_dataloader dataset size: ", len(para_train_dataloader.dataset))
    # print("sts_train_dataloader dataset size: ", len(sts_train_dataloader.dataset))
    # print("dataloaders[0] dataset size: ", len(dataloaders[0].dataset))
    # print("dataloaders[1] dataset size: ", len(dataloaders[1].dataset))
    # print("dataloaders[2] dataset size: ", len(dataloaders[2].dataset))

    # combined_train_dataloader = CombinedLoader(dataloaders)
    # combined_train_dataloader = CombinedLoader({'sst': sst_train_dataloader, 
    #                                         'para': para_train_dataloader, 
    #                                         'sts': sts_train_dataloader}, mode='sequential')
    combined_train_dataloader = CombinedLoader({'sst': dataloaders[0], 
                                            'para': dataloaders[1], 
                                            'sts': dataloaders[2]}, mode='sequential')
    # print("Second combined train dataloader: ", combined_train_dataloader)
    # print("combined_train_dataloader.iterables: ", combined_train_dataloader.iterables)
    


    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
    # for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        num_batches = 0

        # # BATCH SAMPLING EXTENSION
        # # alpha = 1 - 0.8 * ((epoch - 1) / (args.epochs - 1))
        # alpha = 0.5
        # print("Alpha: ", alpha)

        # weights = [len(dataloader.dataset)**alpha for dataloader in original_dataloaders]
        # # print("Weights computation 1: ", weights)
        # weights = [weight / sum(weights) for weight in weights]
        # # print("Weights computation 2: ", weights)
    

        # dataloaders = []
        # for dataloader, weight in zip(original_dataloaders, weights):
        #     indices = [i for i in range(len(dataloader.dataset)) for _ in range(int(weight * 1000))]
        #     sampler = WeightedRandomSampler(indices, len(indices))
        #     new_dataloader = DataLoader(dataloader.dataset, sampler=sampler)
        #     dataloaders.append(new_dataloader)

        # # combined_train_dataloader = CombinedLoader(dataloaders)
        # # combined_train_dataloader = CombinedLoader({'sst': sst_train_dataloader, 
        # #                                         'para': para_train_dataloader, 
        # #                                         'sts': sts_train_dataloader}, mode='sequential')
        # combined_train_dataloader = CombinedLoader({'sst': dataloaders[0], 
        #                                         'para': dataloaders[1], 
        #                                         'sts': dataloaders[2]}, mode='sequential')

        ##    

        _ = iter(combined_train_dataloader)
        for batch, _, dataloader_idx in tqdm(combined_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
    
                if dataloader_idx == 0: # sst
                    
                    b_ids, b_mask, b_labels = (batch['token_ids'],
                                           batch['attention_mask'], batch['labels'])
    
                    b_ids = b_ids.to(device)
                    b_mask = b_mask.to(device)
                    b_labels = b_labels.to(device)
    
                    optimizer.zero_grad()
                    logits = model.predict_sentiment(b_ids, b_mask)
                    
                    loss = F.cross_entropy(logits, b_labels)
    
                elif dataloader_idx == 1: # para
                    b_ids_1, b_mask_1, b_labels, b_ids_2, b_mask_2 = (batch['token_ids_1'],
                                           batch['attention_mask_1'], batch['labels'],
                                           batch['token_ids_2'],
                                           batch['attention_mask_2'])
                    
                    b_ids_1 = b_ids_1.to(device)
                    b_mask_1 = b_mask_1.to(device)
                    b_labels = b_labels.to(device)
                    b_ids_2 = b_ids_2.to(device)
                    b_mask_2 = b_mask_2.to(device)
                    
                    optimizer.zero_grad()
                    logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                    
                    #print("para logits of shape:", logits.shape)   # (8, 1)
                    #print("para labels of shape:", b_labels.shape)    #(8,)
                    #RuntimeError: Expected floating point type for target with class probabilities, got Long
                    if args.model == 'baseline':
                        loss = F.cross_entropy(logits.squeeze().float(), b_labels.float())
                    elif args.model == 'sbert':
                        loss = F.cross_entropy(logits, b_labels)
                
                elif dataloader_idx == 2: # sts
                    b_ids_1, b_mask_1, b_labels, b_ids_2, b_mask_2 = (batch['token_ids_1'],
                                           batch['attention_mask_1'], batch['labels'],
                                           batch['token_ids_2'],
                                           batch['attention_mask_2'])
                    
                    b_ids_1 = b_ids_1.to(device)
                    b_mask_1 = b_mask_1.to(device)
                    b_labels = b_labels.to(device)
                    b_ids_2 = b_ids_2.to(device)
                    b_mask_2 = b_mask_2.to(device)
                    
                    optimizer.zero_grad()
                    logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                    
                    if args.model == 'baseline':
                        loss = F.cross_entropy(logits.squeeze().float(), b_labels.float())
                    elif args.model == 'sbert':
                        loss = F.mse_loss(logits, b_labels.float())
                
                #TODO MY EDITS END HERE
                #Edit: seems we need different lines in each task for calculating loss
                #loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
    
                loss.backward()
                optimizer.step()
    
                train_loss += loss.item()
                num_batches += 1

        train_loss = train_loss / (num_batches)

        #TODO I ADDED THESE LINES:
        # sst_train_acc, sst_train_y_pred, sst_train_sent_ids, para_train_acc, para_train_y_pred, para_train_sent_ids, sts_train_corr, sts_train_y_pred, sts_train_sent_ids = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        sst_dev_acc, sst_dev_y_pred, sst_dev_sent_ids, para_dev_acc, para_dev_y_pred, para_dev_sent_ids, sts_dev_corr, sts_dev_y_pred, sts_dev_sent_ids = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device, args.model)
        
        #Normalize: STS is bw -1 and 1, others bw 0 and 1 --- add 1 to STS and divide by 2 to get bw 0 and 1
        # sts_train_acc = (sts_train_corr + 1) / 2
        sts_dev_acc = (sts_dev_corr + 1) / 2
        #make avg of all dev accuracies across all tasks
        dev_acc = (sst_dev_acc + para_dev_acc + sts_dev_acc) / 3
        # train_acc = (sst_train_acc + para_train_acc + sts_train_acc) / 3
        #TODO END MY EDITS

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath) #for other separated (not avg) option, change filepath - diff model for each task

        # print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")

def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device, args.model)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device, args.model)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    parser.add_argument("--model", type=str,
                        choices=('baseline', 'sbert'),
                        default="baseline")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.model}-{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)
