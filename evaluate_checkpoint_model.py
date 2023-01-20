import os
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
# from writeCheckpointPredictionsInFile import save_data
from pytorch_lightning import LightningModule
from main import argparse_default
from data import Data
import torch
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score

# from nn_models import HybridModel, TransE, complex
from utils.static_funcs import calculate_wilcoxen_score, select_model

class MyLightningModule(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.load_h()


def save_data(dataset, data_dir="", training="test",  scores=[], method="hybrid"):
    # saving the ground truth values
    data = list()
    if training=="train":
        data = dataset.train_set
        folder = "train"
    elif training == "test":
        data = dataset.test_data
        folder = "test"
    elif training == "valid":
        data = dataset.valid_data
        folder = "test"
    else:
        exit(1)

    with open(data_dir +"-"+ "ground_truth_"+training+ "_"+method+".nt", "w") as prediction_file:
        new_line = "\n"
        # <http://swc2019.dice-research.org/task/dataset/s-00001> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> .
        for idx, (head, relation, tail, score) in enumerate(
                (data)):
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\t" + "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>\t<http://rdf.freebase.com/ns/" + \
                        head.split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>\t<http://rdf.freebase.com/ns/" + \
                        relation.split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>\t<http://rdf.freebase.com/ns/" + \
                        tail.split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://swc2017.aksw.org/hasTruthValue>\t\"" + str(
                score) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"
        prediction_file.write(new_line)


    # saving the pridiction values
    with open(data_dir +"-"+ "prediction_"+training+ "_pred_"+method+".nt", "w") as prediction_file:
        new_line = "\n"
        for idx, (tuple, score) in enumerate(
                zip(data,scores)):
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\t" + "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>\t<http://rdf.freebase.com/ns/" + \
                        tuple[0].split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>\t<http://rdf.freebase.com/ns/" + \
                        tuple[1].split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>\t<http://rdf.freebase.com/ns/" + \
                        tuple[2].split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://swc2017.aksw.org/hasTruthValue>\t\"" + str(
                float(score)) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"
        prediction_file.write(new_line)

    # saving the pridiction values
    with open(data_dir +"-"+ "prediction_"+training+ "_pred_"+method+".txt", "w") as prediction_file:
        new_line = "\n"
        for idx, (tuple, score) in enumerate(
                zip(data,scores)):
            new_line += "<http://dbpedia.org/resource/" + \
                        tuple[0].split("/")[-1][:-1] + ">\t"
            new_line += "<http://dbpedia.org/ontology/" + \
                        tuple[1].split("/")[-1][:-1] + ">\t"
            new_line += "<http://dbpedia.org/resource/" + \
                        tuple[2].split("/")[-1][:-1] + ">\t"
            new_line += str(float(score)) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"
        prediction_file.write(new_line)


def restore_checkpoint(self, model: "pl.LightningModule", ckpt_path: Optional[str] = None):
    return  model.load_from_checkpoint(ckpt_path)


datasets_class = ["domain/","domainrange/","mix/","property/","random/","range/"]
# cls2 = datasets_class[2]


properties_split = ["deathPlace/","birthPlace/","author/","award/","foundationPlace/","spouse/","starring/","subsidiary/"]
# make it true or false
prop_split = False
clss = datasets_class
if prop_split:
    clss = properties_split
args = argparse_default()
bpdp_ds = False
args.path_dataset_folder = 'dataset/'
if args.eval_dataset == "BPDP":
    clss = ["bpdp"]
    args.subpath = None
    args.path_dataset_folder='dataset/hybrid_data/bpdp/'
    bpdp_ds = True
elif args.eval_dataset == "Dbpedia5":
    clss = ["Dbpedia5"]
    args.subpath = None
    args.path_dataset_folder = 'dataset/complete_dataset/dbpedia34k/'



for cc in clss:
    # methods = ["full-Hybrid", "KGE-only", "text-only", "path-only", "text-KGE-Hybrid", "text-path-Hybrid", "KGE-path-Hybrid"]
    methods = ["KGE-only","text-KGE-Hybrid"] #["KGE-only","text-KGE-Hybrid"]
    df_test = pd.DataFrame()
    df_train = pd.DataFrame()
    for cls in methods:
        print("processing:" + cls + "--" + cc)
        method = cls #emb-only  #hybrid

        # args.path_dataset_folder = 'dataset/'
        args.model = cls
        # args.subpath = cc
        # hybrid_app = False
        # args.path_dataset_folder = 'dataset/'
        args.subpath = cc
        # if args.model == "full-hybrid":
        #     args.path_dataset_folder += 'hybrid_data/copaal/'
        #     hybrid_app = True


        if (str(args.model).lower().__contains__("full-hybrid")) or (str(args.model).lower().__contains__("path")):# == "kge-path-hybrid":
            args.path_dataset_folder += 'hybrid_data/copaal/'
            # hybrid_app = True
        # elif str(args.model).lower().__contains__("path"):
        #     args.path_dataset_folder += 'hybrid_data/copaal/'
        #     hybrid_app = True


        args.dataset = Data(args)
            # data_dir=args.path_dataset_folder,
            #                 subpath=args.subpath,
            #                 prop=args.prop,
            #                 complete_data= args.cmp_dataset,emb_typ= args.emb_type ,
        #                 emb_file="",bpdp_dataset=bpdp_ds, full_hybrid=hybrid_app)
        args.num_entities, args.num_relations = args.dataset.num_entities, args.dataset.num_relations

        model, frm = select_model(args)

        dirs = os.listdir(os.path.dirname(os.path.abspath("dataset")) + "/dataset/HYBRID_Storage/")

        for flder in dirs:
        # flder = "2022-02-11 10:53:39.443146"
            chkpnts = os.listdir(os.path.dirname(os.path.abspath("dataset"))+ "/dataset/HYBRID_Storage/" + flder)
            for chk in chkpnts:
                cc_change = cc
                if cc.__contains__("/"):
                    cc_change= cc[:-1].lower()
                if chk.startswith("sample") and chk.lower().__contains__("-"+cc.replace("/","").lower()+"=") \
                        and (chk.lower()).__contains__(cls.lower()) and (chk.lower()).__contains__("--"+str(args.emb_type).lower()+"")\
                        and (chk.lower()).__contains__(cc_change.lower()):
                    print(chk)
                    file_name = chk #"sample-"+cls.replace("/","")+"=0--"+cls2.replace("/","")+"=0-epoch=09-val_loss=0.00.ckpt"
                    pth = os.path.dirname(os.path.abspath(file_name)).replace("comparison","")+"/dataset/HYBRID_Storage/"+flder+"/"+file_name
                    print("Resuls for "+cc)
                    model = model.load_from_checkpoint(pth,args=args)

                    model.eval()

                    # Train F1 train dataset
                    X_train = np.array(args.dataset.idx_train_data)[:, :5]
                    y_train = np.array(args.dataset.idx_train_data)[:, -2]

                    X_train_tensor = torch.Tensor(X_train).long()

                    jj = np.arange(0, len(X_train))
                    x_data = torch.tensor(jj)

                        # X_sen_train_tensor = torch.Tensor(X_sen_train).long()
                    idx_s, idx_p, idx_o = X_train_tensor[:, 0], X_train_tensor[:, 1], X_train_tensor[:, 2]
                    prob = model.forward_triples(idx_s, idx_p, idx_o,x_data).flatten()
                    np.savetxt(os.path.dirname(os.path.abspath("dataset")) + "/dataset/HYBRID_Storage/" + flder + "/"+'predictions_train.txt', prob.detach().numpy())
                    pred = (prob > 0.50).float()
                    pred = pred.data.detach().numpy()
                    print('Acc score on train data', accuracy_score(y_train, pred))
                    print('report:', classification_report(y_train,pred))
                    df_train[cls] = pred

                    save_data(args.dataset,
                              os.path.dirname(os.path.abspath("dataset")) + "/dataset/HYBRID_Storage/" + flder + "/"+chk.split("epoch=")[1].split("-val_loss")[0],
                              "train", pred, method)

                    # Train F1 test dataset
                    X_test = np.array(args.dataset.idx_test_data)[:, :5]
                    y_test = np.array(args.dataset.idx_test_data)[:, -2]
                    X_test_tensor = torch.Tensor(X_test).long()
                    idx_s, idx_p, idx_o = X_test_tensor[:, 0], X_test_tensor[:, 1], X_test_tensor[:, 2]

                    jj = np.arange(0, len(X_test))
                    x_data = torch.tensor(jj)

                    prob = model.forward_triples(idx_s, idx_p, idx_o,x_data,"testing").flatten()
                    np.savetxt(os.path.dirname(os.path.abspath(
                        "dataset")) + "/dataset/HYBRID_Storage/" + flder + "/" + 'predictions_test.txt', prob.detach().numpy())
                    pred = (prob > 0.50).float()
                    pred = pred.data.detach().numpy()
                    print('Acc score on test data', accuracy_score(y_test, pred))
                    print('report:', classification_report(y_test,pred))
                    df_test[cls] = pred

                    save_data(args.dataset,
                              os.path.dirname(os.path.abspath("dataset")) + "/dataset/HYBRID_Storage/" + flder + "/"+chk.split("epoch=")[1].split("-val_loss")[0],
                              "test", pred, method)
                    os.remove(os.path.dirname(os.path.abspath("dataset"))+ "/dataset/HYBRID_Storage/" + flder +"/"+chk)
                    os.remove(
                        os.path.dirname(os.path.abspath("dataset")) + "/dataset/HYBRID_Storage/" + flder + "/model.pt")
                    # exit(1)

# # stats test
# calculate_wilcoxen_score(df_train,"train")
# calculate_wilcoxen_score(df_test,"test")

