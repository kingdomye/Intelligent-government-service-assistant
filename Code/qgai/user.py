import threading
import json

__all__=['User']


class User:
    __default_info=json.loads(open('user_info.json','r').read())

    __default_nes_info=json.loads(open('user_info.json','r').read())['necessary']

    def __init__(self, info_dic: dict):
        """
        初始化一个用户
        :param info_dic: 数据库中保存的所有的信息，包含所有必要信息和已有的基本信息
        """
        self.__thread_block = threading.Event()

        #直接录入必要+基本信息
        self.__info_dic=info_dic

        #一次性信息
        self.__once_info_dic={}

        #检查必要信息
        for key in User.__default_nes_info:
            if key not in info_dic:
                raise Exception("Missing necessary information")

        #检查多于信息
        return

    @property
    def info(self):
        return self.__info_dic


    @property
    def waiter(self):
        """
        此用户流程的进程锁
        :return:
        """
        return self.__thread_block


    def fill_in_table(self,table: dict):

        for key in table:
            if table[key] is not None:
                continue
            #遍历特殊标记
            for mark in table_mark:
                if mark is None:
                    break

            #如果有对应的键值，填充表格
            if key in self.info:
                table[key]=self.info[key]
            elif key in self.__default_nes_info:
                table[key]=self.__default_nes_info[key]

        return table

    #对应询问环节
    def inquire(self):
        question = inquiry.inquire()
        return question         #key,sentence



    # 一个流程的主程序
    def activate(self, input):
        """
        label = classify(input)


        # 提取空表+流程
        data = Data()
        table = data[label]

        table = self.fill_in_table(table)

        self.waiter.wait()
        """
        return 1

