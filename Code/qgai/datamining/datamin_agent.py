import json
import os

from deal_flow_api import deal_flow

table_mark={
    "<IMG>":0,
    "<SIG>":1,
    "<SEL>":2,
    "<PAD>":3,
    "<DAT>":4,
}

class DataMiningAgent:
    def __init__(self, tables_path='tables.json',
                 idx_path='name_to_idx.json',
                 flow_path='flows.json', ):
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

    def get_org_flow(self, idx:int)->str:
        """
        Get flow by index
        :param idx: flow index(view in 'name_to_idx.json')
        :return: flow / error:-1
        """
        try:
            idx = str(idx)
            return self.flows[idx]
        except KeyError:
            return '-1'

    def get_flow(self, idx:int, user_info:dict)->str:
        try:
            flow = deal_flow(user_info, self.get_org_flow(idx))
            return flow
        except KeyError:
            return '-1'

# # 调试
# data = DataMiningAgent()
# print(data.get_tables(0))
# print(data.get_flow(0))