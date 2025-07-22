import json

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
        :return: index
        """
        return self.name_to_idx[label]

    def idx_to_label(self, idx: int)->str:
        """
        Transform idx to label
        :param idx: index of business
        :return: label
        """
        reverse_dict = {v: k for k, v in self.name_to_idx.items()}
        return reverse_dict[idx]

    def get_tables(self, idx: int)->list:
        """
        Get table by index
        :param idx: table index(view in 'name_to_idx.json')
        :return: tables list (list of dicts from 'tables.json')
        """
        idx = str(idx)
        return list(self.tables[idx].values())

    def get_flow(self, idx)->str:
        idx = str(idx)
        return self.flows[idx]

    def __getitem__(self, idx: int)->(str,list):
        return self.get_flow(idx), self.get_tables(idx)

# 调试
# data = DataMiningAgent()
# print(data[0])