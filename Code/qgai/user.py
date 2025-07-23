import threading
import json


import inquiry
import datamining
import classify

__all__=['User','TableFiller']


class TableFiller:
    def __init__(self, tables):
        #表组
        self.__tables = tables

        #表格遍历迭代器的序号-1
        self.__index = -1

        #当前填写表格的指针序号
        self.__pointer = 0

    @property
    def table(self):
        """
        返回当前所填表
        :return:
        """
        return self.__tables[self.__index]
    @table.setter
    def table(self, table):
        """
        修改当前所填表
        :param table: 修改内容
        :return:
        """
        self.__tables[self.__index] = table

    @property
    def tables(self):
        return self.__tables

    @property
    def pointer(self):
        return self.__pointer

    @property
    def is_finish(self):
        return self.__pointer==len(self.__tables)

    def __iter__(self):
        self._index=-1
        return self
    def __next__(self):
        self._index+=1

        if self._index<len(self.__tables):
            return self.__tables[self._index]
        else:
            raise StopIteration
    def __getitem__(self, index):
        return self.__tables[index]



    def reset_pointer(self):
        self.__pointer=0

    def export_table(self):
        """
        导出当前所填表，并将pointer换到下一张表
        :return:
        """
        table = self.__tables[self.__index]
        self.__pointer+=1

        return table

default_nes_info=json.loads(open('user_info.json','r').read())['necessary']
default_adt_info=json.loads(open('user_info.json','r').read())['addition']


default_main_info=default_nes_info.copy()
default_main_info.update(default_adt_info)

class User:

    def __init__(self, info_dic: dict):
        """
        初始化一个用户
        :param info_dic: 数据库中保存的所有的信息，包含所有必要信息和已有的基本信息
        """
        self.__thread_block = threading.Event()

        #直接录入必要+基本信息
        self.__main_info=info_dic

        #一次性信息
        self.__once_info_dic={}

        #当前业务流程标签
        #self.__label=-1

        #业务流程
        #self.__flow=None

        #表组迭代器
        #self.__tables_filler = None

        #初始化是否完成
        self.__init = False

        #检查必要信息
        for key in default_nes_info:
            if key not in info_dic:
                raise Exception("Missing necessary information")

        #检查多于信息
        return


    @property
    def init_finish(self)->bool:
        return self.__init

    def _check_init(self)->bool:
        if self.init_finish:
            return True
        else:
            raise Exception("initialization of this user is not finished,please execute fuction activate if you didn't do it")

    @property
    def info(self):
        """
        用户的所有信息
        :return:
        """
        once_tb = self.once_info
        once_tb.update(self.main_info)
        return once_tb

    @property
    def main_info(self):
        """
        用户的主要信息，即必要信息+基本信息
        :return:
        """
        return self.__main_info

    @property
    def once_info(self):
        """
        用户的一次性信息，不会录入数据库，下次需要重新询问
        :return:
        """
        return self.__once_info_dic

    @property
    def waiter(self)->threading.Event:
        """
        此用户流程的进程锁
        :return:
        """
        return self.__thread_block


    @property
    def bus_type(self):
        self._check_init()
        return self.__label

    @property
    def tables_filler(self)->TableFiller:
        self._check_init()

        return self.__tables_filler

    @property
    def tables(self):
        return self.tables_filler.tables

    @property
    def flow(self)->str:
        """
        此用户的业务流程
        :return:
        """
        self._check_init()

        if self.__flow=="":
            return ""
        return self.__flow

    def __setitem__(self, key, value):
        """
        录入用户信息
        :param key: 用户信息键
        :param value: 用户信息值
        :return:
        """
        if key in default_main_info:
            self.__main_info[key]=value
        else:
            self.__once_info_dic[key]=value



    def _fill_in_table(self,table: dict):

        for key in table:
            if table[key] is not None:
                continue
            #遍历特殊标记
            for mark in datamining.table_mark:
                if mark is None:
                    break

            #如果有对应的键值，填充表格
            if key in self.info:
                table[key]=self.info[key]
            elif key in default_nes_info:
                table[key]=default_nes_info[key]

        return table

    #对应询问环节
    def inquire(self):
        self._check_init()

        question = inquiry.Inquiring(self.tables_filler.table)
        if question is None:
            return None
        return question.keys()[0],question.values()[0]         #key,sentence



    # 一个流程的主程序
    def activate(self, input):
        #分类
        self.__label = classify.classify(input,classify.type_dic)


        # 提取空表+流程
        data = datamining.DataMiningAgent()
        self.__flow,lables = data[self.__label]

        #迭代填表
        self.__tables_filler = TableFiller(lables)
        self.__tables_filler.reset_pointer()


        #到此处为初始化过程，以上信息是必要的
        self.__init=True

        while not self.__tables_filler.is_finish:
            self.__tables_filler.table = self._fill_in_table(self.__tables_filler.table)

            #填完表等待服务器录入信息
            self.waiter.wait()




        return 1

