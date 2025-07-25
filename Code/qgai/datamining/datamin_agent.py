import json
import os

<<<<<<< Updated upstream
=======
from datamining.deal_flow_api import deal_flow

>>>>>>> Stashed changes
table_mark={
    "<IMG>":0,
    "<SIG>":1,
    "<SEL>":2,
    "<PAD>":3,
    "<DAT>":4,
}

class DataMiningAgent:
    def __init__(self, tables_path='datamining/tables.json',
                 idx_path='datamining/name_to_idx.json',
                 flow_path='datamining/flows.json', ):
        assert os.path.exists(tables_path), f"{tables_path} does not exist"
        assert os.path.exists(idx_path), f"{idx_path} does not exist"
        assert os.path.exists(flow_path), f"{flow_path} does not exist"

        with open(idx_path, 'r', encoding='utf-8') as f:
            self.name_to_idx = json.load(f)
        with open(tables_path, 'r', encoding='utf-8') as f:
            self.tables = json.load(f)
        with open(flow_path, 'r', encoding='utf-8') as f:
            self.flows = json.load(f)

    def label_to_idx(self, label: str)->int:
        """
        Transform label to index
        :param label: key word of business like '身份证'
        :return: index / error:-1
        """
        try :
            return self.name_to_idx[label]
        except KeyError:
            return -1

    def idx_to_label(self, idx: int)->str:
        """
        Transform idx to label
        :param idx: index of business
        :return: label / error:-1
        """
        reverse_dict = {v: k for k, v in self.name_to_idx.items()}
        try:
            return reverse_dict[idx]
        except KeyError:
            return '-1'

    def get_tables(self, idx: int)->list:
        """
        Get table by index
        :param idx: table index(view in 'name_to_idx.json')
        :return: tables list (list of dicts from 'tables.json') / error: ['-1']
        """
        try:
            idx = str(idx)
            return list(self.tables[idx].values())
        except KeyError:
            return ['-1']

    def get_flow(self, idx)->str:
        """Get flow by index
        :param idx: flow index(view in 'name_to_idx.json')
        :return: flow / error:-1
        """
        try:
            idx = str(idx)
            return self.flows[idx]
        except KeyError:
            return '-1'

    def __getitem__(self, idx: int)->(str, list):
        """
        Get basic data by idx
        :param idx: index of business
        :return: flow: str, tables: list of dicts
        """
        return self.get_flow(idx), self.get_tables(idx)


# ## 调试
# data = DataMiningAgent()
# print(data[666])