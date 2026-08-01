import json
from collections.abc import AsyncGenerator
from pathlib import Path
# from server.console import log


table_mark={
    "<IMG>":0,
    "<SIG>":1,
    "<SEL>":2,
    "<PAD>":3,
    "<DAT>":4,
}

DATA_DIR = Path(__file__).resolve().parent


class DataMiningAgent:
    def __init__(
        self,
        tables_path: str | Path = DATA_DIR / "tables.json",
        idx_path: str | Path = DATA_DIR / "name_to_idx.json",
        flow_path: str | Path = DATA_DIR / "flows.json",
    ):
        tables_path = Path(tables_path)
        idx_path = Path(idx_path)
        flow_path = Path(flow_path)
        for path in (tables_path, idx_path, flow_path):
            if not path.is_file():
                raise FileNotFoundError(path)

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

    # def get_flow(self, idx:int, user_info:dict, mod='remote', cat_mode=False)->str:
    #     """
    #     Get final flow
    #     :param idx: index(view in 'name_to_idx.json')
    #     :param user_info: user info
    #     :param mod: 'local' or 'remote' (default remote)
    #     :param cat_mode: cat gril mode (default False)
    #     :return: flow / error:-1 / mod error
    #     """
    #     user_info['业务类型'] = self.idx_to_label(idx)
    #     user_info = self.anonymize_user_data(user_info)
    #     if mod == 'remote':
    #         try:
    #             flow = deal_flow_a(user_info, self.get_org_flow(idx), cat_gril=cat_mode)
    #             return flow
    #         except KeyError:
    #             return '-1'
    #     elif mod == 'local':
    #         try:
    #             flow = deal_flow_l(user_info, self.get_org_flow(idx), cat_gril=cat_mode)
    #             return flow
    #         except KeyError:
    #             return '-1'
    #     return 'mod error'


    async def get_flow(
        self, idx: int, user_info: dict
    ) -> AsyncGenerator[str, None] | None:
        """
        返回流程
        :param idx: table index(view in 'name_to_idx.json')
        :param user_info: user info
        :return:
        """
        enriched_info = {**user_info, "业务类型": self.idx_to_label(idx)}
        anonymized_info = self.anonymize_user_data(enriched_info)
        try:
            from .deal_flow_plus import generate_streaming_response

            flow = generate_streaming_response(
                anonymized_info,
                raw_text=self.get_org_flow(idx),
            )
            return flow
        except ValueError:
            return None


    @staticmethod
    def anonymize_user_data(user_info: dict) -> dict:
        """脱敏用户敏感信息"""
        anonymized = user_info.copy()

        # 脱敏姓名（保留姓氏）
        if '姓名' in anonymized:
            name = anonymized['姓名']
            if len(name) > 1:
                anonymized['姓名'] = name[0] + '*' * (len(name) - 1)

        # 脱敏年龄范围
        if "年龄" in anonymized:
            age = anonymized["年龄"]
            if not isinstance(age, (int, float)):
                anonymized.pop("年龄")
            elif age < 20:
                anonymized["年龄范围"] = "0-19岁"
                del anonymized["年龄"]
            elif age < 40:
                anonymized['年龄范围'] = "20-39岁"
                del anonymized["年龄"]
            elif age < 60:
                anonymized['年龄范围'] = "40-59岁"
                del anonymized["年龄"]
            else:
                anonymized['年龄范围'] = "60岁以上"
                del anonymized["年龄"]

        # 移除其他敏感字段
        sensitive_fields = ['身份证号', '联系电话', '详细地址', "电子邮箱"]
        for field in sensitive_fields:
            if field in anonymized:
                del anonymized[field]

        return anonymized



# # 调试
#data = DataMiningAgent()
# print(data.get_tables(0))
#print(data.get_flow(0))
