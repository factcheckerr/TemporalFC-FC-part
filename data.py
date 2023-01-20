from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import pandas as pd
from copy import deepcopy
from urllib.parse import quote, unquote
class Data:
    def __init__(self, args=None):
        # , data_dir=None,
        #          sub_dataset_path=None, prop=None,
        #          complete_data= False, emb_typ = "TransE", emb_file = "",
        #          bpdp_dataset=False, full_hybrid = False, emb_folder = ""):
        complete_dataset  = args.cmp_dataset
        full_hybrid = (args.model == "full-Hybrid")
        data_dir = args.path_dataset_folder
        sub_dataset_path = "" if (args.sub_dataset_path==None) else args.sub_dataset_path
        prop = args.prop
        emb_typ = args.emb_type
        bpdp_dataset = (args.eval_dataset == "BPDP")
        emb_folder = str(args.eval_dataset).lower()+"/"
        # Quick workaround as we happen to have duplicate triples.
        # None if load complete data, otherwise load parts of dataset with folders in wrong directory.
        # emb_folder = ""
        if args.eval_dataset == "BPDP": #bpdp_dataset == True:
            emb_folder = "bpdp/"
            if full_hybrid:
                data_dir = "dataset/hybrid_data/bpdp/hybrid_data/copaal/"
                self.train_set = list((self.load_data(data_dir + "train/", data_type="train")))
                self.test_data = list((self.load_data(data_dir + "test/", data_type="test")))
            else:
                data_dir = "dataset/hybrid_data/bpdp/"
                self.train_set = list((self.load_data(data_dir+"train/", data_type="train")))
                self.test_data = list((self.load_data(data_dir+"test/", data_type="test")))
        elif args.eval_dataset == "FactBench" and complete_dataset==True: # for the entire dataset
            self.train_set = list((self.load_data(data_dir+"complete_dataset/", data_type="train")))
            self.test_data = list((self.load_data(data_dir+"complete_dataset/", data_type="test")))
        elif args.eval_dataset == "FactBench" and args.prop != None: # for properties split based datasets
            self.train_set = list((self.load_data(data_dir + "properties_split/train/" + prop, data_type="train")))
            self.test_data = list((self.load_data(data_dir + "properties_split/test/" + prop, data_type="test")))
        # elif sub_dataset_path == None:
        #     self.train_set = list((self.load_data(data_dir, data_type="train")))
        #     self.test_data = list((self.load_data(data_dir, data_type="test")))
        elif args.eval_dataset == "FactBench" and full_hybrid == True:
            self.train_set = list((self.load_data(data_dir + "train/" + sub_dataset_path, data_type="train")))
            self.test_data = list((self.load_data(data_dir + "test/" + sub_dataset_path, data_type="test")))
        elif args.eval_dataset == "Dbpedia5":
            self.train_set = list((self.load_data(data_dir + "hybrid_data/train/", data_type="train")))
            self.test_data = list((self.load_data(data_dir + "hybrid_data/test/", data_type="test")))
            # self.train_set_time = list((self.load_data_with_time(args.path_dataset_folder+  "train/", data_type="train_original")))
            # self.test_set_time = list((self.load_data_with_time(args.path_dataset_folder+ "test/", data_type="test_original")))
            # self.train_set_time_final = self.update_and_match_triples_start(self, args.path_dataset_folder, "train", "train_with_time_final.txt",
            #                                                                 self.train_set, self.train_set_time)
            # self.test_set_time_final = self.update_and_match_triples_start(self, args.path_dataset_folder, "test", "test_with_time_final.txt",
            #                                                                self.test_set, self.test_set_time)
        else:
            self.train_set = list((self.load_data(data_dir+"hybrid_data/train/"+sub_dataset_path, data_type="train")))
            self.test_data = list((self.load_data(data_dir+"hybrid_data/test/"+sub_dataset_path, data_type="test")))

        # random split
        # test_size = len(self.test_data) - int(len(self.test_data) / 3)
        # valid_size = len(self.test_data) - (len(self.test_data) - int(len(self.test_data) / 3))
        # # adding validation set in the sets
        # self.test_data, self.valid_set = random_split(self.test_data, [test_size, valid_size])
        # adding validation set in the sets
        # self.test_data, self.valid_set  = self.test_data[0:test_size], self.test_data[test_size:len(self.test_data)+1]
        #generate test and validation sets
        self.test_data, self.valid_data = self.generate_test_valid_set(self, self.test_data)



        # factcheck predictions on train and test data
        if bpdp_dataset == True:
            self.train_set_pred = list(
                (self.load_data(data_dir+"train/", data_type="train_pred", pred=True)))
            self.test_data_pred = list(
                (self.load_data(data_dir+"test/", data_type="test_pred", pred=True)))

        elif args.eval_dataset == "FactBench" and complete_dataset==True:
            self.train_set_pred = list((self.load_data(data_dir+"complete_dataset/", data_type="train_pred", pred=True)))
            self.test_data_pred = list((self.load_data(data_dir+"complete_dataset/", data_type="test_pred", pred=True)))
            sub_dataset_path = "complete_data"
        elif args.eval_dataset == "FactBench" and prop != None:
            self.train_set_pred = list((self.load_data(data_dir + "properties_split/train/" + prop, data_type="train_pred", pred=True)))
            self.test_data_pred = list((self.load_data(data_dir + "properties_split/test/" + prop, data_type="test_pred", pred=True)))
        elif sub_dataset_path==None:
            self.train_set_pred = list((self.load_data(data_dir , data_type="train_pred",pred=True)))
            self.test_data_pred = list((self.load_data(data_dir , data_type="test_pred",pred=True)))
        elif full_hybrid == True:
            self.train_set_pred = list(
                (self.load_data(data_dir + "train/" + sub_dataset_path, data_type="train_pred", pred=True)))
            self.test_data_pred = list(
                (self.load_data(data_dir + "test/" + sub_dataset_path, data_type="test_pred", pred=True)))
        elif args.eval_dataset == "Dbpedia5":
            self.train_set_pred = list((self.load_data(data_dir + "train/" , data_type="train_pred", pred=True)))
            self.test_data_pred = list((self.load_data(data_dir + "test/" , data_type="test_pred", pred=True)))
        else:
            self.train_set_pred = list((self.load_data(data_dir+"hybrid_data/train/"+sub_dataset_path, data_type="train_pred",pred=True)))
            self.test_data_pred = list((self.load_data(data_dir+"hybrid_data/test/"+sub_dataset_path, data_type="test_pred",pred=True)))
        self.test_data_pred, self.valid_data_pred = self.generate_test_valid_set(self, self.test_data_pred)

        self.data = self.train_set + list(self.test_data) + list(self.valid_data)
        self.entities = self.get_entities(self.data)
        # uncomment it later when needed
        # if bpdp_dataset:
        #     if full_hybrid:
        #         if data_dir == 'dataset/hybrid_data/copaal/':
        #             data_dir = 'dataset/hybrid_data/bpdp/hybrid_data/copaal/'
        #         self.save_all_resources(self.entities, data_dir, "combined/",
        #                                 True)
        #     else:
        #         self.save_all_resources(self.entities, data_dir, "/combined/", True)
        # elif args.eval_dataset == "FactBench" and prop != None:
        #     self.save_all_resources(self.entities, data_dir, "hybrid_data/combined/properties_split/" + prop.replace("/","_"), True)
        # elif full_hybrid == True:
        #     self.save_all_resources(self.entities, data_dir.replace("hybrid_data/copaal", ""), "hybrid_data/combined/" + sub_dataset_path,
        #                             True)
        # else:
        #     self.save_all_resources(self.entities, data_dir, "hybrid_data/combined/" + sub_dataset_path, True)



        # self.relations = list(set(self.get_relations(self.train_set) + self.get_relations(self.test_data)))
        self.relations = self.get_relations(self.data)
        # uncomment it later  when needed
        # if bpdp_dataset:
        #     self.save_all_resources(self.relations, data_dir, "/combined/", False)
        #     # exit(1)
        # elif args.eval_dataset == "FactBench" and prop != None:
        #     self.save_all_resources(self.relations, data_dir, "hybrid_data/combined/properties_split/" + prop.replace("/","_"), False)
        # elif full_hybrid == True:
        #     self.save_all_resources(self.relations, data_dir.replace("hybrid_data/copaal",""), "hybrid_data/combined/" + sub_dataset_path, False)
        # else:
        #     self.save_all_resources(self.relations, data_dir, "hybrid_data/combined/" + sub_dataset_path, False)

        self.num_entities = len(self.entities)
        self.num_relations = len(self.relations)

        self.idx_entities = dict()
        self.idx_relations = dict()

        # Generate integer mapping
        for i in self.entities:
            self.idx_entities[i] = len(self.idx_entities)
        for i in self.relations:
            self.idx_relations[i] = len(self.idx_relations)


        self.emb_entities = self.get_embeddings(self.idx_entities,'Embeddings/'+emb_typ+'/'+emb_folder,'all_entities_embeddings')
        self.emb_relation = self.get_embeddings(self.idx_relations,'Embeddings/'+emb_typ+'/'+emb_folder,'all_relations_embeddings')
        if bpdp_dataset == True:
            self.emb_sentences_train1,self.train_set = self.get_sent_embeddings(self, data_dir + "combined/", 'trainSE.csv',
                                                                 self.train_set)

            self.emb_sentences_train = self.update_sent_train_embeddings(self, self.emb_sentences_train1)
            self.emb_sentences_test, self.emb_sentences_valid , self.test_data, self.valid_data = self.get_sent_test_valid_embeddings(
                data_dir + "combined/", 'testSE.csv', self.test_data, self.valid_data)
            self.emb_sentences_test = self.update_sent_train_embeddings(self, self.emb_sentences_test)
            self.emb_sentences_valid = self.update_sent_train_embeddings(self, self.emb_sentences_valid)
            if full_hybrid:
                self.copaal_veracity_score1 = self.get_copaal_veracity(data_dir + "combined/" , 'trainSE.csv',
                                                                       self.train_set)
                self.copaal_veracity_train = self.update_veracity_train_data(self, self.copaal_veracity_score1)
                self.copaal_veracity_test1, self.copaal_veracity_valid1 = self.get_veracity_test_valid_data(
                    data_dir + "combined/", 'testSE.csv', self.test_data, self.valid_data)
                self.copaal_veracity_test = self.update_veracity_train_data(self, self.copaal_veracity_test1)
                self.copaal_veracity_valid = self.update_veracity_train_data(self, self.copaal_veracity_valid1)

        elif args.eval_dataset == "FactBench" and complete_dataset==True:
            self.emb_sentences_train1,self.train_set = self.get_sent_embeddings(self, data_dir + "complete_dataset/",'trainSE.csv', self.train_set)
            self.emb_sentences_train = self.update_sent_train_embeddings(self, self.emb_sentences_train1)
            self.emb_sentences_test, self.emb_sentences_valid , self.test_data, self.valid_data = self.get_sent_test_valid_embeddings(
                data_dir + "complete_dataset/" , 'testSE.csv', self.test_data, self.valid_data)
            self.emb_sentences_test = self.update_sent_train_embeddings(self, self.emb_sentences_test)
            self.emb_sentences_valid = self.update_sent_train_embeddings(self, self.emb_sentences_valid)

        elif full_hybrid == True:
            print("to be updated")
            self.emb_sentences_train1,self.train_set = self.get_sent_embeddings(self, data_dir + "train/" + sub_dataset_path, 'trainSE.csv',
                                                                 self.train_set)
            self.copaal_veracity_score1 = self.get_copaal_veracity(data_dir + "train/" + sub_dataset_path, 'trainSE.csv',
                                                                 self.train_set)
            self.emb_sentences_train = self.update_sent_train_embeddings(self, self.emb_sentences_train1)
            self.copaal_veracity_train = self.update_veracity_train_data(self,self.copaal_veracity_score1)

            self.emb_sentences_test1, self.emb_sentences_valid1, self.test_data, self.valid_data = self.get_sent_test_valid_embeddings(
                self, path=data_dir + "test/" + sub_dataset_path, name='testSE.csv', test_data=self.test_data, valid_data=self.valid_data)

            self.copaal_veracity_test1, self.copaal_veracity_valid1 = self.get_veracity_test_valid_data(
                data_dir + "test/" + sub_dataset_path, 'testSE.csv', self.test_data, self.valid_data)
            self.emb_sentences_test = self.update_sent_train_embeddings(self, self.emb_sentences_test1)
            self.emb_sentences_valid = self.update_sent_train_embeddings(self, self.emb_sentences_valid1)

            self.copaal_veracity_test = self.update_veracity_train_data(self, self.copaal_veracity_test1)
            self.copaal_veracity_valid = self.update_veracity_train_data(self, self.copaal_veracity_valid1)

        elif args.eval_dataset == "Dbpedia5":
            print("sentence embeddings parsing started....it may take a while...please wait.....")
            if (os.path.exists(data_dir + "hybrid_data/train/trainSEUpdated.csv")):
                self.emb_sentences_train1 = pd.read_csv(data_dir + "hybrid_data/train/trainSEUpdated.csv", sep="\t")
                self.train_set = self.load_data(data_dir + "hybrid_data/train/", data_type="trainUpdated")
                # self.emb_sentences_train1 = self.emb_sentences_train1.to_dict()
                self.emb_sentences_train1 = self.dataframe_to_dict(self.emb_sentences_train1)
            else:
                self.emb_sentences_train1, self.train_set = self.get_sent_embeddings(self, data_dir + "hybrid_data/train/" + sub_dataset_path, 'trainSE.csv',
                                                                                     self.train_set)
                # self.save_tuples_to_file(self.emb_sentences_train1,data_dir+"hybrid_data/train/trainSEUpdated.csv")
                df = pd.DataFrame(self.emb_sentences_train1)
                df.to_csv(data_dir + "hybrid_data/train/trainSEUpdated.csv", index=False, sep="\t")
                self.save_tuples_to_file(self.train_set, data_dir + "hybrid_data/train/trainUpdated.txt")

            if (os.path.exists(data_dir + "hybrid_data/test/testSEUpdated.csv")):
                self.emb_sentences_test1 = pd.read_csv(data_dir + "hybrid_data/test/testSEUpdated.csv", sep="\t")
                self.test_data = self.load_data(data_dir + "hybrid_data/test/", data_type="testUpdated")
                self.emb_sentences_valid1 = pd.read_csv(data_dir + "hybrid_data/test/validSEUpdated.csv", sep="\t")
                self.valid_data = self.load_data(data_dir + "hybrid_data/test/", data_type="validUpdated")
                self.emb_sentences_test1 = self.dataframe_to_dict(self.emb_sentences_test1)
                self.emb_sentences_valid1 = self.dataframe_to_dict(self.emb_sentences_valid1)
                # self.emb_sentences_test1 = self.emb_sentences_test1.to_dict()
                # self.emb_sentences_valid1 = self.emb_sentences_valid1.to_dict()
            else:
                self.emb_sentences_test1, self.emb_sentences_valid1, self.test_data, self.valid_data = self.get_sent_test_valid_embeddings(
                    self, path=data_dir + "hybrid_data/test/" + sub_dataset_path, name='testSE.csv', test_data=self.test_data, valid_data=self.valid_data)
                # self.save_tuples_to_file(self.emb_sentences_test1, data_dir + "hybrid_data/test/testSEUpdated.csv")
                self.save_tuples_to_file(self.test_data, data_dir + "hybrid_data/test/testUpdated.txt")
                df = pd.DataFrame(self.emb_sentences_test1)
                df.to_csv(data_dir + "hybrid_data/test/testSEUpdated.csv", index=False, sep="\t")
                df = pd.DataFrame(self.emb_sentences_valid1)
                df.to_csv(data_dir + "hybrid_data/test/validSEUpdated.csv", index=False, sep="\t")
                # self.save_tuples_to_file(self.emb_sentences_valid1, data_dir + "hybrid_data/test/validSEUpdated.csv")
                self.save_tuples_to_file(self.valid_data, data_dir + "hybrid_data/test/validUpdated.txt")


            self.emb_sentences_train = self.update_sent_train_embeddings(self, self.emb_sentences_train1)
            self.emb_sentences_test = self.update_sent_train_embeddings(self, self.emb_sentences_test1)
            self.emb_sentences_valid = self.update_sent_train_embeddings(self, self.emb_sentences_valid1)

        elif args.eval_dataset == "FactBench" and prop!=None:
            self.emb_sentences_train1,self.train_set = self.get_sent_embeddings(self, data_dir + "properties_split/train/" + prop, 'trainSE.csv', self.train_set)
            self.emb_sentences_train = self.update_sent_train_embeddings(self, self.emb_sentences_train1)
            self.emb_sentences_test, self.emb_sentences_valid, self.test_data, self.valid_data = self.get_sent_test_valid_embeddings(
            data_dir + "properties_split/test/" + prop, 'testSE.csv', self.test_data, self.valid_data)
            self.emb_sentences_test = self.update_sent_train_embeddings(self, self.emb_sentences_test)
            self.emb_sentences_valid = self.update_sent_train_embeddings(self, self.emb_sentences_valid)
        else:
            self.emb_sentences_test, self.emb_sentences_valid, self.test_data, self.valid_data = self.get_sent_test_valid_embeddings(
                self, data_dir + "hybrid_data/test/" + sub_dataset_path, 'testSE.csv', self.test_data, self.valid_data)
            self.emb_sentences_train1,self.train_set = self.get_sent_embeddings(self, data_dir+"hybrid_data/train/"+sub_dataset_path,'trainSE.csv', self.train_set)
            self.emb_sentences_train = self.update_sent_train_embeddings(self, self.emb_sentences_train1)
            self.emb_sentences_test = self.update_sent_train_embeddings(self, self.emb_sentences_test)
            self.emb_sentences_valid = self.update_sent_train_embeddings(self, self.emb_sentences_valid)

        self.idx_train_data = []
        i = 0
        for (s, p, o, label) in self.train_set:
            idx_s, idx_p, idx_o, label = self.idx_entities[s], self.idx_relations[p], self.idx_entities[o], label
            self.idx_train_data.append([idx_s, idx_p, idx_o, label , i])
            i = i + 1

        self.idx_valid_data = []
        j = 0
        for (s, p, o, label) in self.valid_data:
            idx_s, idx_p, idx_o, label = self.idx_entities[s], self.idx_relations[p], self.idx_entities[o], label
            self.idx_valid_data.append([idx_s, idx_p, idx_o, label,j])
            j = j + 1

        self.idx_test_data = []
        k = 0
        for (s, p, o, label) in self.test_data:
            idx_s, idx_p, idx_o, label = self.idx_entities[s], self.idx_relations[p], self.idx_entities[o], label
            self.idx_test_data.append([idx_s, idx_p, idx_o, label,k])
            k = k + 1
    def dataframe_to_dict(self, df):
        data_dict = {}
        for i, row in df.iterrows():
            # Extract the keys (first 3 columns) and values (remaining columns)
            keys = i
            values = row[0:].tolist()
            # Add the key-value pair to the dictionary
            data_dict[keys] = values

        return data_dict.values()
    def save_tuples_to_file(self, tuples, file_name):
        with open(file_name, 'w') as file:
            for t in tuples:
                file.write(str(t[0] +"\t" + t[1] +"\t" +t[2] +"\t"+ str(True if (t[3]==1) else False)) + '\n')

    def is_valid_test_available(self):
        if len(self.valid_data) > 0 and len(self.test_data) > 0:
            return True
        return False
    @staticmethod
    def save_all_resources(list_all_entities, data_dir, sub_path, entities):
        if entities:
            with open(data_dir+sub_path+'all_entities.txt',"w") as f:
                for item in list_all_entities:
                    f.write("%s\n" % item)
        else:
            with open(data_dir + sub_path + 'all_relations.txt', "w") as f:
                for item in list_all_entities:
                    f.write("%s\n" % item)

    @staticmethod
    def update_and_match_triples_start(self, selected_dataset_data_dir, type, file_name, data_set1, data_set2, properties_split=None, veracity=False):
        if veracity == False:
            if (os.path.exists(selected_dataset_data_dir + type + "/" + file_name)):
                self.set_time_final = list(self.load_data(selected_dataset_data_dir + type + "/", data_type=str(file_name).replace(".txt", ""), pred=True))
            else:
                if len(data_set1) != len(data_set2):
                    self.set_time_final = self.update_match_triples(data_set1, data_set2)
                else:
                    self.set_time_final = data_set2
                self.save_triples(selected_dataset_data_dir, type + "/" + file_name, self.set_time_final)
        else:
            tt = "properties/train/" if (file_name.__contains__("train")) else "properties/test/"
            split = "" if (properties_split == 'None') else tt + "correct/" + properties_split + "_"
            if (os.path.exists(selected_dataset_data_dir + type + "/" + split + file_name)):
                self.set_time_final = list(self.load_data(selected_dataset_data_dir + type + "/" + split, data_type=str(file_name).replace(".txt", ""), pred=True))
            else:
                self.set_time_final = self.update_match_triples(data_set1, data_set2, veracity=veracity)
                self.save_triples(selected_dataset_data_dir, type + "/" + split + file_name, self.set_time_final, veracity=veracity)
        return self.set_time_final

    @staticmethod
    def save_triples(data_dir, type, triples, veracity=False):
        if veracity == False:
            with open(data_dir + type, "w") as f:
                for item in triples:
                    f.write("" + (item[0]) + "\t" + (item[1]) + "\t" + (item[2]) + "\t" + str(item[3]) + "\t" + str(item[4]) + "\n")
        else:
            with open(data_dir + type, "w") as f:
                for item in triples:
                    f.write("" + str(item[0]) + "\t" + str(item[1]) + "\t" + str(item[2]) + "\t" + str(item[3]) + "\n")

    @staticmethod
    def update_match_triples(data_set1, data_set2, veracity=False, final=False):
        data = []
        data_set21 = deepcopy(data_set2)
        # subs = [tp2[0] for tp2 in data_set2]
        # preds = [tp2[1].replace("Of","") for tp2 in data_set2]
        # objs = [tp2[2] for tp2 in data_set2]
        for tp in data_set1:
            found = False
            if veracity == False:
                for tpt in data_set21:
                    # if tpt[0].__contains__('Amadou_Toumani') and (tp[2].__contains__('Amadou_Toumani_')):
                    #     print("test")
                    if (tp[0] == tpt[0] and tp[1] == tpt[1].replace("Of", "") and tp[2] == tpt[2]):
                        data.append([tp[0], tp[1], tp[2], tpt[3], tpt[4]])
                        found = True
                        break
                    elif (tp[2] == tpt[0] and tp[0] == tpt[2]):  # to cover negative triples we are doing like this, swaping the sub and obj and not checking the predicate
                        data.append([tpt[0], tpt[1], tpt[2], tpt[3], 'False'])
                        found = True
                        break

                if found == False:
                    print("not found:" + str(tp))
            elif veracity == True and final == True:  # final is for second check
                for tpt in data_set21:
                    if (tp[0] == tpt[0] and tp[1] == tpt[1].replace("Of", "") and tp[2] == tpt[2]):
                        data.append([tp[0], tp[1], tp[2], tp[3], tp[4]])
                        found = True
                        break
                    elif (tp[2] == tpt[0] and tp[1] == tpt[1].replace("Of", "") and tp[0] == tpt[
                        2]):  # to cover negative triples we are doing like this, swaping the sub and obj and not checking the predicate
                        data.append([tp[0], tp[1], tp[2], tp[3], tp[4]])
                        found = True
                        break
                if found == False:
                    print("not found:" + str(tp))
                else:
                    data_set21.remove(tpt)

            else:
                for tpt in data_set21:
                    if (tp[0] == tpt[0] and tp[1] == tpt[1].replace("Of", "") and tp[2] == tpt[2]):
                        data.append([tp[0], tp[1], tp[2], tp[3]])
                        found = True
                        break
                    elif (tp[2] == tpt[0] and tp[0] == tpt[2]):  # to cover negative triples we are doing like this, swaping the sub and obj and not checking the predicate
                        data.append([tp[0], tp[1], tp[2], tp[3]])
                        found = True
                        break
                if found == False:
                    print("not found:" + str(tp))
                # break

                # else:
                #     print("problematic triple:"+ str(tp))

        return data
    @staticmethod
    def generate_test_valid_set(self, test_data):
        test_set = []
        valid_set = []
        i = 0
        sent_i = 0
        for data in test_data:
            if i % 20 == 0:
                valid_set.append(data)
            else:
                test_set.append(data)

            i += 1
        return  test_set, valid_set

    @staticmethod
    def load_data_with_time(data_dir, data_type, mapped_entities=None, prop=None):
        try:
            data = []
            with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                for datapoint in f:
                    datapoint = datapoint.split("\t")
                    if len(datapoint) >= 5:
                        if len(datapoint) > 5:
                            datapoint[5] = '_'.join(datapoint[4:])
                        s, p, o, time, loc = datapoint[0:5]
                        if prop != None:
                            if not str(p).__eq__(prop + "Of"):
                                continue
                        s = "http://dbpedia.org/resource/" + s
                        if (mapped_entities != None and s in mapped_entities.keys()):
                            s = mapped_entities[s]
                        p = "http://dbpedia.org/ontology/" + p
                        o = "http://dbpedia.org/resource/" + o
                        if (mapped_entities != None and o in mapped_entities.keys()):
                            o = mapped_entities[o]
                        data.append(("<" + s + ">", "<" + p + ">", "<" + o + ">", time, "True"))
                    elif len(datapoint) == 3:
                        s, p, label = datapoint
                        assert label == 'True' or label == 'False'
                        if label == 'True':
                            label = 1
                        else:
                            label = 0
                        data.append((s, p, 'DUMMY', label))
                    else:
                        raise ValueError
        except FileNotFoundError as e:
            print(e)
            print('Add empty.')
            data = []
        return data
    @staticmethod
    def load_data(data_dir, data_type, pred=False):
        try:
            data = []
            if pred == False:
                with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                    for datapoint in f:
                        datapoint = datapoint.split()
                        if len(datapoint) == 4:
                            s, p, o, label = datapoint
                            if label == 'True':
                                label = 1
                            else:
                                label = 0
                            data.append((s, p, o, label))
                        elif len(datapoint) == 3:
                            s, p, label = datapoint
                            assert label == 'True' or label == 'False'
                            if label == 'True':
                                label = 1
                            else:
                                label = 0
                            data.append((s, p, 'DUMMY', label))
                        else:
                            raise ValueError
            else:
                with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                    for datapoint in f:
                        datapoint = datapoint.split()
                        if len(datapoint) == 5:
                            s, p, o, label, dot = datapoint
                            data.append((s, p, o, label.replace("\"^^<http://www.w3.org/2001/XMLSchema#double>","")))
                        elif len(datapoint) == 4:
                            s, p, o, label = datapoint
                            data.append((s, p, o, label))
                        elif len(datapoint) == 3:
                            s, p, label = datapoint
                            data.append((s, p, 'DUMMY', label))
                        else:
                            raise ValueError
        except FileNotFoundError as e:
            print(e)
            print('Add empty.')
            data = []
        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities
    # update the embeddings manually with tabs instead of commas if commas are there
    # / home / umair / Documents / pythonProjects / HybridFactChecking / Embeddings / ConEx_dbpedia
    @staticmethod
    def get_embeddings(idxs,path,name):
        embeddings = dict()
        # print("%s%s.txt" % (path,name))
        with open("%s%s.txt" % (path,name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0	 1	 2	 3	 4	 5	 6	 7	 8"):
                    continue
                data = datapoint.split('>\t')
                if len(data)==1:
                    print("stoped: getting embeddings function")
                    exit(1)
                if len(data) > 1:
                    data2 = data[0]+">",data[1].split('\t')
                    # test = data2[0].replace("\"","").replace("_com",".com").replace("Will-i-am","Will.i.am").replace("Will_i_am","Will.i.am")
                    # test = data2[0].replace("\"","")
                    if data2[0] in idxs:
                        embeddings[data2[0]] = data2[1]
        for emb in idxs:
            if emb not in embeddings.keys():
                print("this is missing in embeddings file:"+ emb)
                exit(1)

        if len(idxs) > len(embeddings):
            print("embeddings missing")
            exit(1)
        embeddings_final = dict()
        for emb in idxs.keys():
            if emb in embeddings.keys():
                embeddings_final[emb] = embeddings[emb]
            else:
                print('no embedding', emb)
                exit(1)

        return embeddings_final.values()

    @staticmethod
    def get_comma_seperated_embeddings(idxs, path, name):
        embeddings = dict()
        # print("%s%s.txt" % (path,name))
        with open("%s%s.txt" % (path, name), "r") as f:
            for datapoint in f:
                data = datapoint.split('> ,')
                if datapoint.startswith("<http://dbpedia.org/resource/Abu_Jihad_("):
                    print(datapoint)
                if len(data) == 1:
                    data = datapoint.split('>\",')
                if len(data) > 1:
                    data2 = data[0] + ">", data[1].split(',')
                    # test = data2[0].replace("\"","").replace("_com",".com").replace("Will-i-am","Will.i.am").replace("Will_i_am","Will.i.am")
                    test = data2[0].replace("\"", "")
                    if test in idxs:
                        embeddings[test] = data2[1]
                    # else:
                    #     print('Not in embeddings:',datapoint)
                    # exit(1)
                # else:
                #     print('Not in embeddings:',datapoint)
                #     exit(1)
        for emb in idxs:
            if emb not in embeddings.keys():
                print("this is missing in embeddings file:" + emb)
                exit(1)

        if len(idxs) > len(embeddings):
            print("embeddings missing")
            exit(1)
        embeddings_final = dict()
        for emb in idxs.keys():
            if emb in embeddings.keys():
                embeddings_final[emb] = embeddings[emb]
            else:
                print('no embedding', emb)
                exit(1)

        return embeddings_final.values()
    @staticmethod
    def get_copaal_veracity(path, name, train_data):
        emb = dict()

        embeddings_train = dict()
        # print("%s%s" % (path,name))

        i = 0
        train_i = 0
        found = False
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0\t1\t2"):
                    continue
                else:
                    emb[i] = datapoint.split('\t')
                    try:
                        for dd in train_data:
                            # figure out some way to handle this first argument well
                            if (emb[i][0] == dd[0].replace(',', '')) and (emb[i][1] == dd[1].replace(',', '')) and (
                                    emb[i][2] == dd[2].replace(',', '')):
                                # print('train data found')
                                embeddings_train[train_i] =np.append(emb[i][:3],emb[i][-1].replace("\n",""))
                                train_i += 1
                                found = True
                                break

                            # else:
                            #     print('error')
                            # exit(1)
                    except:
                        print('ecception')
                        exit(1)
                    if found == False:
                        if (train_i >= len(train_data)):
                            break
                        else:
                            print("some training data missing....not found:" + str(emb[i]))
                            exit(1)
                    i = i + 1
                    found = False

                    # i = i+1
            embeddings_train_final = dict()
            jj = 0
            # print("sorting")
            for embb in train_data:
                ff = False
                for embb2 in embeddings_train.values():
                    if ((embb[0].replace(',', '') == embb2[0].replace(',', '')) and (
                            embb[1].replace(',', '') == embb2[1].replace(',', '')) and (
                            embb[2].replace(',', '') == embb2[2].replace(',', ''))):
                        embeddings_train_final[jj] = embb2
                        jj = jj + 1
                        ff = True
                        break
                if ff == False:
                    print("problem: not found")
                    exit(1)

        if len(train_data) != len(embeddings_train_final):
            print("problem")
            exit(1)
        return embeddings_train_final.values()
    @staticmethod
    def update_entity(self, ent):
        ent = ent.replace("+", "")
        if (ent.__contains__("&") or ent.__contains__("%")) and (
                (not ent.__contains__("%3F")) and (not ent.__contains__("%22"))):
            sub2 = ""

            for chr in ent:
                if chr == "&" or chr == "%":
                    break
                else:
                    sub2 += chr
            if ent[0]=="<":
                ent = sub2 + ">"
            else:
                ent = sub2

        if ent.__contains__("?"):
            ent = ent.replace("?", "%3F")

        if ent.__contains__("\"\""):
            ent= ent.replace("\"\"", "%22")
        if ent[0] == "\"" and ent[-1] == "\"":
            ent = ent[1:-1]
        if ent[0] == "\'" and ent[-1] == "\'":
            ent = ent[1:-1]
        return ent

    def without(self,d, key):
        new_d = d.copy()
        new_d2 = dict()
        new_d.pop(key)
        count = 0
        for dd in new_d.values():
            new_d2[count]=dd
            count+=1
        return new_d2
    @staticmethod
    def get_sent_embeddings(self, path, name, train_data):
        emb = dict()
        embeddings_train = dict()
        # print("%s%s" % (path,name))
        train_data_copy = deepcopy(train_data)
        train_data1 = np.array(train_data_copy)
        i = 0
        train_i = 0
        found = False
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0\t1\t2"):
                    continue
                else:
                    if datapoint.startswith("http://dbpedia.org/resource/Vlado_Brankovic"):
                        print("test")
                    emb[i] = datapoint.split('\t')
                    try:
                        if emb[i][0] != "0":
                            emb[i][0] = self.update_entity(self, emb[i][0])    # todo same thing for other array..somehting like thisself.update_entity(self, train_data1)
                            emb[i][1] = self.update_entity(self, emb[i][1])
                            emb[i][2] = self.update_entity(self, emb[i][2])
                            if ((list(train_data1[:, 0]).__contains__("<" + emb[i][0] + ">")) and
                                    (list(train_data1[:, 1]).__contains__("<" + emb[i][1] + ">")) and
                                    (list(train_data1[:, 2]).__contains__("<" + emb[i][2] + ">"))):
                                emb[i][-1] = emb[i][-1].replace("'", "").replace("\n", "")
                                if (len(emb[i])) == ((768 * 3) + 3 + 1):
                                    # because defacto scores are also appended at the end
                                    embeddings_train[train_i] = emb[i][:-1]
                                elif (len(emb[i])) == ((768 * 3) + 3):
                                    # emb[i][-1] = emb[i]
                                    embeddings_train[train_i] = emb[i]
                                else:
                                    print("there is something fishy:" + str(emb[i]))
                                    exit(1)

                                train_i += 1
                                found = True
                            else:
                                found = False
                                # else:
                                #     print('error')
                                # exit(1)
                    except:
                        print('ecception')
                        exit(1)
                    # if found==True:
                        # train_data1.remove("<" + emb[i][0] + ">")
                    # else:
                    #     train_data.remove(dd)
                    # if (train_i >= 5):  # len(train_data)
                    #     break
                    if (train_i >= 8000):  # len(train_data)
                        break
                    if found == False:
                        if (train_i >= len(train_data)): #len(train_data)
                            break
                        # else:
                            # print("some training data missing....not found:" + str(emb[i]))
                            # print(i)
                            # print("train")
                            # exit(1)
                    i = i + 1
                    found = False

                    # i = i+1

        if len(train_data) != len(embeddings_train):
            print("problem: length of train and sentence embeddings arrays are different:train:"+str(len(train_data))+",emb:"+str(len(embeddings_train)))
            # exit(1)
        # following code is just for ordering the data in sentence vectors
        train_i = 0
        train_data_copy = deepcopy(train_data)
        embeddings_train_final = dict()
        for dd in train_data:
            found_data = False
            jj = 0
            for sd in embeddings_train.values():
                sub = self.update_entity(self, dd[0])  # to be updated please
                pred = self.update_entity(self, dd[1])
                obj = self.update_entity(self, dd[2])
                if sd[0][0]=='<' and sd[0][-1] == '>':
                    sub1 =  self.update_entity(self, sd[0])
                    pred1 = self.update_entity(self, sd[1])
                    obj1 =  self.update_entity(self, sd[2])
                else:
                    sub1 = '<'+self.update_entity(self, sd[0])+'>'
                    pred1 = '<'+self.update_entity(self, sd[1])+'>'
                    obj1 = '<'+self.update_entity(self, sd[2])+'>'
                if (((sub.replace(",", "") == sub1.replace(",", "")) and (pred == pred1) and (obj.replace(",", "") == obj1.replace(",", "")))
                    or
                    (( sub1.lower()== sub.lower()) and ( pred1.lower() == pred.lower()) and
                     ( obj1.lower() == obj.lower()))):
                    embeddings_train_final[train_i] = sd
                    train_i+=1
                    found_data = True
                    break
                jj += 1
            if found_data== False:
                train_data_copy.remove(dd)
                print("missing train data from sentence embeddings file:"+str(dd))
            else:
                # print("to delete from list: "+str(sd))
                embeddings_train =  self.without(embeddings_train,jj)
                # embeddings_train = embeddings_train.dropna().reset_index(drop=True)
                # del embeddings_train[sd]

        train_data = deepcopy(train_data_copy)
        return embeddings_train_final.values(), train_data

    # @staticmethod
    # def get_sent_embeddings(path, name, train_data):
    #     emb = dict()
    #
    #     embeddings_train = dict()
    #     # print("%s%s" % (path,name))
    #
    #     i = 0
    #     train_i = 0
    #     found = False
    #     with open("%s%s" % (path, name), "r") as f:
    #         for datapoint in f:
    #             if datapoint.startswith("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"):
    #                 continue
    #             else:
    #                 if datapoint.startswith("http://dbpedia.org/resource/Vlado_Brankovic"):
    #                     print("test")
    #                 emb[i] = datapoint.split('\t')
    #                 try:
    #                     if emb[i][0] != "\"0\"":
    #                         for dd in train_data:
    #                             # replaced because factcheck results does not contained punctuations
    #                             sub = dd[0].replace("+", "")
    #                             if (sub.__contains__("&") or sub.__contains__("%")) and (
    #                                     (not sub.__contains__("%3F")) and (not sub.__contains__("%22"))):
    #                                 sub2 = ""
    #                                 for chr in sub:
    #                                     if chr == "&" or chr == "%":
    #                                         break
    #                                     else:
    #                                         sub2 += chr
    #
    #                                 sub = sub2 + ">"
    #                             pred = dd[1]
    #                             obj = dd[2].replace("+", "")
    #                             if (obj.__contains__("&") or obj.__contains__("%")) and (
    #                                     (not obj.__contains__("%3F")) and (not obj.__contains__("%22"))):
    #                                 sub2 = ""
    #
    #                                 for chr in obj:
    #                                     if chr == "&" or chr == "%":
    #                                         break
    #                                     else:
    #                                         sub2 += chr
    #                                 obj = sub2 + ">"
    #                             # emb[i][0][0] == "\"" and
    #                             if emb[i][0][0] == "\"":
    #                                 emb[i][0] = emb[i][0][1:]
    #                             if emb[i][0][-1] == "\"":
    #                                 emb[i][0] = emb[i][0][0:-1]
    #                             if emb[i][1][0] == "\"" and emb[i][1][-1] == "\"":
    #                                 emb[i][1] = emb[i][1][1:-1]
    #                             if emb[i][2][0] == "\"" and emb[i][2][-1] == "\"":
    #                                 emb[i][2] = emb[i][2][1:-1]
    #
    #                             if emb[i][0].__contains__("?"):
    #                                 emb[i][0] = emb[i][0].replace("?", "%3F")
    #                             if emb[i][2].__contains__("?"):
    #                                 emb[i][2] = emb[i][2].replace("?", "%3F")
    #
    #                             if str(dd).__contains__("http://dbpedia.org/resource/Ismael_") and emb[i][
    #                                 0].__contains__("http://dbpedia.org/resource/Ismael_%22El_Mayo%22_Zambada"):
    #                                 print("test")
    #
    #                             if emb[i][0].__contains__("\"\""):
    #                                 emb[i][0] = emb[i][0].replace("\"\"", "%22")
    #
    #                             if emb[i][2].__contains__("\"\""):
    #                                 emb[i][2] = emb[i][2].replace("\"\"", "%22")
    #
    #                             # ('<http://dbpedia.org/resource/Ismael_%22El_Mayo%22_Zambada>',
    #                             #  '<http://dbpedia.org/ontology/commander>',
    #                             #  '<http://dbpedia.org/resource/Battle_of_Culiacán>', 1)
    #                             # figure out some way to handle this first argument well
    #                             if ((emb[i][0] == sub) and
    #                                     (emb[i][1] == pred) and
    #                                     (emb[i][2] == obj)
    #                                     or
    #                                     ('<' + emb[i][0].lower() + '>' == sub.lower()) and
    #                                     ('<' + emb[i][1].lower() + '>' == pred.lower()) and
    #                                     ('<' + emb[i][2].lower() + '>' == obj.lower())):
    #                                 # print('train data found')
    #                                 if (len(emb[i])) == ((768 * 3) + 1):
    #                                     embeddings_train[train_i] = emb[i][:-1]
    #                                 else:
    #                                     embeddings_train[train_i] = emb[i]
    #                                 train_i += 1
    #                                 found = True
    #                                 break
    #
    #                             # else:
    #                             #     print('error')
    #                             # exit(1)
    #                 except:
    #                     print('ecception')
    #                     exit(1)
    #                 if found == False:
    #                     if (train_i >= len(train_data)):
    #                         break
    #                     else:
    #                         print("some training data missing....not found:" + str(emb[i]))
    #                         print(i)
    #                         print("test")
    #                         # exit(1)
    #                 i = i + 1
    #                 found = False
    #
    #                 # i = i+1
    #         embeddings_train_final = dict()
    #         jj = 0
    #         # print("sorting")
    #         # for embb2 in embeddings_train.values():
    #         #     ff = False
    #         #     for embb in train_data:
    #         #         # ////////////////////////////////////////////////////
    #         #         sub = embb[0].replace("+", "")
    #         #         pred = embb[1].replace("+", "")
    #         #         obj = embb[2].replace("+", "")
    #         #         if (embb[0].__contains__("&") or embb[0].__contains__("%")) and (
    #         #                 (not embb[0].__contains__("%3F")) and (not embb[0].__contains__("%22"))):
    #         #             sub2 = ""
    #         #             for chr in embb[0]:
    #         #                 if chr == "&" or chr == "%":
    #         #                     break
    #         #                 else:
    #         #                     sub2 += chr
    #         #             sub = sub2 + ">"
    #         #         if (embb[2].__contains__("&") or embb[2].__contains__("%")) and (
    #         #                 (not embb[2].__contains__("%3F")) and (not embb[2].__contains__("%22"))):
    #         #             sub2 = ""
    #         #             for chr in embb[2]:
    #         #                 if chr == "&" or chr == "%":
    #         #                     break
    #         #                 else:
    #         #                     sub2 += chr
    #         #             obj = sub2 + ">"
    #         #         # //////////////////////////////////////////////
    #         #         if (((sub.replace(',', '') == embb2[0].replace(',', '')) and (
    #         #                 pred.replace(',', '') == embb2[1].replace(',', '')) and (
    #         #                 obj.replace(',', '') == embb2[2].replace(',', ''))) or
    #         #             ((sub.replace(',', '') == "<"+embb2[0].replace(',', '')+">") and (
    #         #                     pred.replace(',', '') == "<"+embb2[1].replace(',', '')+">") and (
    #         #                      obj.replace(',', '') == "<"+embb2[2].replace(',', '')+">"))):
    #         #             embeddings_train_final[jj] = embb2
    #         #             jj = jj + 1
    #         #             ff = True
    #         #             break
    #         #     if ff == False:
    #         #         print("problem: not found")
    #         # exit(1)
    #
    #     if len(train_data) != len(embeddings_train):
    #         print("problem")
    #         # exit(1)
    #     return embeddings_train.values()
    @staticmethod
    def update_copaal_veracity_score(self, train_emb):
        embeddings_train = dict()
        i = 0
        for train in train_emb:
            embeddings_train[i] = train[3:]
            i += 1

        return embeddings_train.values()

    @staticmethod
    def update_veracity_train_data(self, train_emb):
        embeddings_train = dict()
        i = 0
        for train in train_emb:
            embeddings_train[i] = train[3:]
            i += 1

        return embeddings_train.values()
    @staticmethod
    def update_sent_train_embeddings(self, train_emb):
        embeddings_train = dict()
        i=0
        for train in train_emb:
            embeddings_train[i] = train[3:]
            i+=1

        return embeddings_train.values()

    @staticmethod
    def get_veracity_test_valid_data(path, name, test_data, valid_data):
        embeddings_test, embeddings_valid = dict(), dict()
        emb = dict()
        # print("%s%s" % (path, name))
        found = False
        i = 0
        test_i = 0
        valid_i = 0
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0\t1\t2"):
                    continue
                else:
                    emb[i] = datapoint.split('\t')
                    try:
                        for dd in test_data:
                            # figure out some way to handle this first argument well
                            if (emb[i][0].replace(',', '') == dd[0].replace(',', '')) and (
                                    emb[i][1].replace(',', '') == dd[1].replace(',', '')) and (
                                    emb[i][2].replace(',', '') == dd[2].replace(',', '')):
                                # print('test data found')
                                embeddings_test[test_i] = np.append(emb[i][:3],emb[i][-1].replace("\n",""))
                                test_i += 1
                                found = True
                                break
                        for vd in valid_data:
                            # figure out some way to handle this first argument well
                            if (emb[i][0].replace(',', '') == vd[0].replace(',', '')) and (
                                    emb[i][1].replace(',', '') == vd[1].replace(',', '')) and (
                                    emb[i][2].replace(',', '') == vd[2].replace(',', '')):
                                # print('valid data found')
                                embeddings_valid[valid_i] = np.append(emb[i][:3],emb[i][-1].replace("\n",""))
                                valid_i += 1
                                found = True
                                break
                        if found == False:
                            print("some data missing from test and validation sets..error" + str(emb[i]))
                            exit(1)
                        else:
                            found = False

                    except:
                        print('ecception')
                        exit(1)
                    i = i + 1

        embeddings_test_final, embeddings_valid_final = dict(), dict()
        i = 0
        for dd in test_data:
            for et in embeddings_test.values():
                if (et[0].replace(',', '') == dd[0].replace(',', '')) and (
                        et[1].replace(',', '') == dd[1].replace(',', '')) and (
                        et[2].replace(',', '') == dd[2].replace(',', '')):
                    embeddings_test_final[i] = et
                    i = i + 1
                    break
        i = 0
        for dd in valid_data:
            # print(dd)
            for et in embeddings_valid.values():
                if (et[0].replace(',', '') == dd[0].replace(',', '')) and (
                        et[1].replace(',', '') == dd[1].replace(',', '')) and (
                        et[2].replace(',', '') == dd[2].replace(',', '')):
                    embeddings_valid_final[i] = et
                    i = i + 1
                    break
        if (len(embeddings_valid_final) != len(valid_data)) and (len(embeddings_test_final) != len(test_data)):
            exit(1)
        return embeddings_test_final.values(), embeddings_valid_final.values()


    @staticmethod
    def get_sent_test_valid_embeddings(self, path, name, test_data, valid_data):
        embeddings_test, embeddings_valid = dict(),dict()
        emb = dict()
        # print("%s%s" % (path, name))
        found = False
        i = 0
        test_i = 0
        valid_i = 0
        test_data1 = np.array(test_data)
        valid_data1 = np.array(valid_data)
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0\t1\t2"):
                    continue
                else:
                    emb[i] = datapoint.split('\t')
                    try:
                        if emb[i][0] != "\"0\"":
                            emb[i][0] = self.update_entity(self, emb[i][0])
                            emb[i][1] = self.update_entity(self, emb[i][1])
                            emb[i][2] = self.update_entity(self, emb[i][2])
                            if ((list(test_data1[:, 0]).__contains__("<" + emb[i][0] + ">")) and
                                (list(test_data1[:, 1]).__contains__("<" + emb[i][1] + ">")) and
                                (list(test_data1[:, 2]).__contains__("<" + emb[i][2] + ">"))):

                                emb[i][-1] = emb[i][-1].replace("'", "").replace("\n", "")
                                if (len(emb[i])) == ((768 * 3) + 3 + 1):
                                    # because defacto scores are also appended at the end
                                    embeddings_test[test_i] = emb[i][:-1]
                                elif (len(emb[i])) == ((768 * 3) + 3):
                                    # emb[i][-1] = emb[i]
                                    embeddings_test[test_i] = emb[i]
                                else:
                                    print("there is something fishy:" + str(emb[i]))
                                    exit(1)
                                # embeddings_test[test_i] = emb[i]
                                test_i += 1
                                found = True
                            else:
                                found = False


                            # for dd in test_data:
                            #     # figure out some way to handle this first argument well
                            #     sub = self.update_entity(self, dd[0])
                            #     pred = self.update_entity(self, dd[1])
                            #     obj = self.update_entity(self, dd[2])
                            #
                            #
                            #
                            #
                            #     if  (((emb[i][0].replace(',', '') == sub.replace(',','')) and
                            #             (emb[i][1].replace(',', '') == pred.replace(',','')) and (
                            #             emb[i][2].replace(',', '') == obj.replace(',','')))
                            #             or
                            #             (('<'+emb[i][0].lower()+'>' == sub.lower()) and
                            #             ('<'+emb[i][1].lower()+'>' == pred.lower()) and
                            #             ('<'+emb[i][2].lower()+'>' == obj.lower()))):
                            #         # print('test data found')
                            #         emb[i][-1] = emb[i][-1].replace("'", "").replace("\n", "")
                            #         if (len(emb[i])) == ((768 * 3) + 3 + 1):
                            #             # because defacto scores are also appended at the end
                            #             embeddings_test[test_i] = emb[i][:-1]
                            #         elif (len(emb[i])) == ((768 * 3) + 3):
                            #             # emb[i][-1] = emb[i]
                            #             embeddings_test[test_i] = emb[i]
                            #         else:
                            #             print("there is something fishy:" + str(emb[i]))
                            #             exit(1)
                            #         # embeddings_test[test_i] = emb[i]
                            #         test_i += 1
                            #         found = True
                            #         break
                            if found == False:
                                if ((list(valid_data1[:, 0]).__contains__("<" + emb[i][0] + ">")) and
                                        (list(valid_data1[:, 1]).__contains__("<" + emb[i][1] + ">")) and
                                        (list(valid_data1[:, 2]).__contains__("<" + emb[i][2] + ">"))):

                                    emb[i][-1] = emb[i][-1].replace("'", "").replace("\n", "")
                                    if (len(emb[i])) == ((768 * 3) + 3 + 1):
                                        # because defacto scores are also appended at the end
                                        embeddings_valid[valid_i] = emb[i][:-1]
                                    elif (len(emb[i])) == ((768 * 3) + 3):
                                        # emb[i][-1] = emb[i]
                                        embeddings_valid[valid_i] = emb[i]
                                    else:
                                        print("there is something fishy:" + str(emb[i]))
                                        exit(1)
                                    # embeddings_test[valid_i] = emb[i]
                                    valid_i += 1
                                    found = True
                                else:
                                    found = False

                                # for vd in valid_data:
                                #     sub = self.update_entity(self, vd[0])
                                #     pred = self.update_entity(self, vd[1])
                                #     obj = self.update_entity(self, vd[2])
                                #
                                #     emb[i][0] = self.update_entity(self, emb[i][0])
                                #     emb[i][1] = self.update_entity(self, emb[i][1])
                                #     emb[i][2] = self.update_entity(self, emb[i][2])
                                #
                                #     # figure out some way to handle this first argument well
                                #     if (((emb[i][0].replace(',', '') == sub.replace(',', '')) and (
                                #             emb[i][1].replace(',', '') == pred.replace(',', '')) and (
                                #             emb[i][2].replace(',', '') == obj.replace(',', '')))
                                #             or
                                #             (('<' + emb[i][0].lower() + '>' == sub.lower()) and
                                #              ('<' + emb[i][1].lower() + '>' == pred.lower()) and
                                #              ('<' + emb[i][2].lower() + '>' == obj.lower()))):
                                #         # print('valid data found')
                                #         emb[i][-1] = emb[i][-1].replace("'", "").replace("\n", "")
                                #         if (len(emb[i])) == ((768 * 3) + 3 + 1):
                                #             # because defacto scores are also appended at the end
                                #             embeddings_valid[valid_i] = emb[i][:-1]
                                #         elif (len(emb[i])) == ((768 * 3) + 3):
                                #             # emb[i][-1] = emb[i]
                                #             embeddings_valid[valid_i] = emb[i]
                                #         else:
                                #             print("there is something fishy:" + str(emb[i]))
                                #             exit(1)
                                #         # embeddings_valid[valid_i] = emb[i]
                                #         valid_i += 1
                                #         found = True
                                #         break
                            # if found == False:
                                # if valid_i > 5:
                                #     break
                                # print("some data missing from test and validation sets..error"+ str(emb[i]))
                                    # exit(1)
                            # else:
                            if test_i >= 3500:
                                break
                            found = False

                    except:
                        print('ecception')
                        exit(1)
                    i = i + 1

        # embeddings_test_final, embeddings_valid_final = dict(), dict()
        # i = 0
        # for dd in test_data:
        #     for et in embeddings_test.values():
        #         if ((et[0].replace(',', '') == dd[0].replace(',', '')) and \
        #                 (et[1].replace(',', '') == dd[1].replace(',', '')) and \
        #                 (et[2].replace(',', '') == dd[2].replace(',', '')) \
        #                 or
        #                 (('<' + et[0].lower() + '>' == dd[0].lower()) and
        #                  ('<' + et[1].lower() + '>' == dd[1].lower()) and
        #                  ('<' + et[2].lower() + '>' == dd[2].lower()))):
        #             embeddings_test_final[i] = et
        #             i = i + 1
        #             break
        # i = 0
        # for dd in valid_data:
        #     # print(dd)
        #     for et in embeddings_valid.values():
        #         if ((et[0].replace(',', '') == dd[0].replace(',', '')) and\
        #                 (et[1].replace(',', '') == dd[1].replace(',', '')) and\
        #                 (et[2].replace(',', '') == dd[2].replace(',', ''))
        #                 or
        #                 (('<' + et[0].lower() + '>' == dd[0].lower()) and
        #                  ('<' + et[1].lower() + '>' == dd[1].lower()) and
        #                  ('<' + et[2].lower() + '>' == dd[2].lower()))):
        #             embeddings_valid_final[i] = et
        #             i = i + 1
        #             break
        if (len(embeddings_valid)!= len(valid_data)) and (len(embeddings_test)!= len(test_data)):
            print("check lengths of valid and test data:valid_emb:"+str(len(embeddings_valid))+
                  " valid_data"+str(len(valid_data))+
                  "test_data:"+str(len(test_data))+"test_emb:"+str(len(embeddings_test)))
            # exit(1)
        train_i = 0
        test_data_copy = deepcopy(test_data)
        valid_data_copy = deepcopy(valid_data)
        embeddings_test_final = dict()
        embeddings_valid_final = dict()
        for dd in test_data:
            found_data = False
            jj = 0
            for sd in embeddings_test.values():
                sub = self.update_entity(self, dd[0])
                pred = self.update_entity(self, dd[1])
                obj = self.update_entity(self, dd[2])
                if sd[0][0]=='<' and sd[0][-1] == '>':
                    sub1 =  self.update_entity(self, sd[0])
                    pred1 = self.update_entity(self, sd[1])
                    obj1 =  self.update_entity(self, sd[2])
                else:
                    sub1 = '<' + self.update_entity(self, sd[0]) + '>'
                    pred1 = '<' + self.update_entity(self, sd[1]) + '>'
                    obj1 = '<' + self.update_entity(self, sd[2]) + '>'
                if (((sub.replace(",", "") == sub1.replace(",", "")) and (pred == pred1) and (obj.replace(",", "") == obj1.replace(",", "")))
                        or
                        ((sub1.lower().replace(",", "") == sub.lower().replace(",", "")) and (pred1.lower() == pred.lower()) and
                         (obj1.lower().replace(",", "") == obj.lower().replace(",", "")))):
                    embeddings_test_final[train_i] = sd
                    train_i += 1
                    found_data = True
                    break
                jj += 1
            if found_data == False:
                test_data_copy.remove(dd)
                print("missing test data from sentence embeddings file:" + str(dd))
            else:
                # embeddings_test.pop(jj)
                embeddings_test = self.without(embeddings_test, jj)
                # print("to delete from list: " + str(sd))
                # del embeddings_test[sd]

        test_data = deepcopy(test_data_copy)

        train_i = 0
        for dd in valid_data:
            found_data = False
            jj = 0
            for sd in embeddings_valid.values():
                sub = self.update_entity(self, dd[0])
                pred = self.update_entity(self, dd[1])
                obj = self.update_entity(self, dd[2])
                if sd[0][0]=='<' and sd[0][-1] == '>':
                    sub1 =  self.update_entity(self, sd[0])
                    pred1 = self.update_entity(self, sd[1])
                    obj1 =  self.update_entity(self, sd[2])
                else:
                    sub1 = '<' + self.update_entity(self, sd[0]) + '>'
                    pred1 = '<' + self.update_entity(self, sd[1]) + '>'
                    obj1 = '<' + self.update_entity(self, sd[2]) + '>'
                if (((sub.replace(",", "") == sub1.replace(",", "")) and (pred == pred1) and (obj.replace(",", "") == obj1.replace(",", "")))
                        or
                        ((sub1.lower().replace(",", "") == sub.lower().replace(",", "")) and (pred1.lower() == pred.lower()) and
                         (obj1.lower().replace(",", "") == obj.lower().replace(",", "")))):
                    embeddings_valid_final[train_i] = sd
                    train_i += 1
                    found_data = True
                    break
                jj += 1
            if found_data == False:
                valid_data_copy.remove(dd)
                print("missing valid data from sentence embeddings file:" + str(dd))
            else:
                # print("to delete from list: " + str(sd))
                # embeddings_valid.pop(jj)
                embeddings_valid = self.without(embeddings_valid, jj)


        valid_data = deepcopy(valid_data_copy)

        return embeddings_test_final.values(), embeddings_valid_final.values(), test_data, valid_data


        # return embeddings.values()



# # Test data class
# bpdp = True
# if not bpdp:
#     properties_split = ["deathPlace/","birthPlace/","author/","award/","foundationPlace/","spouse/","starring/","subsidiary/"]
#     datasets_class = ["range/","domain/","mix/","property/","domainrange/","random/"]
#     # make it true or false
#     prop_split = True
#     clss = datasets_class
#     if prop_split:
#         clss = properties_split
#
#     for cls in clss:
#         method = "emb-only" #emb-only  hybrid
#         path_dataset_folder = 'dataset/'
#         if prop_split:
#             dataset = Data(data_dir=path_dataset_folder, sub_dataset_path= None, prop = cls)
#         else:
#             dataset = Data(data_dir=path_dataset_folder, sub_dataset_path= cls)
# else:
#     path_dataset_folder = 'dataset/hybrid_data/bpdp/'
#     dataset = Data(data_dir=path_dataset_folder, bpdp_dataset=True)
#     print("success")
