import json
import os

from datamining.deal_flow_api import deal_flow_a
from datamining.deal_flow_local import deal_flow_l
from server.console import log
# from deal_flow_api import deal_flow_a
# from deal_flow_local import deal_flow_l


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

    def get_flow(self, idx:int, user_info:dict, mod='remote', cat_mode=False)->str:
        """
        Get final flow
        :param idx: index(view in 'name_to_idx.json')
        :param user_info: user info
        :param mod: 'local' or 'remote' (default remote)
        :param cat_mode: cat gril mode (default False) (just local)
        :return: flow / error:-1 / mod error
        """
        user_info['业务类型'] = self.idx_to_label(idx)
        user_info = self.anonymize_user_data(user_info)
        if mod == 'remote':
            try:
                flow = deal_flow_a(user_info, self.get_org_flow(idx))
                return flow
            except KeyError:
                return '-1'
        elif mod == 'local':
            try:
                flow = deal_flow_l(user_info, self.get_org_flow(idx), cat_gril=cat_mode)
                return flow
            except KeyError:
                return '-1'
        return 'mod error'

    def anonymize_user_data(self, user_info):
        """脱敏用户敏感信息"""
        anonymized = user_info.copy()

        # 脱敏姓名（保留姓氏）
        if '姓名' in anonymized:
            name = anonymized['姓名']
            if len(name) > 1:
                anonymized['姓名'] = name[0] + '*' * (len(name) - 1)

        # 脱敏年龄范围
        if '年龄' in anonymized:
            age = anonymized['年龄']
            if age < 20:
                anonymized['年龄范围'] = str(age)
            elif age < 40:
                anonymized['年龄范围'] = "20-39岁"
            elif age < 60:
                anonymized['年龄范围'] = "40-59岁"
            else:
                anonymized['年龄范围'] = str(age)
            del anonymized['年龄']

        # 移除其他敏感字段
        sensitive_fields = ['身份证号', '联系电话', '详细地址']
        for field in sensitive_fields:
            if field in anonymized:
                del anonymized[field]

        return anonymized



# # 调试
#data = DataMiningAgent()
# print(data.get_tables(0))
#print(data.get_flow(0))