import threading
import json
import asyncio


import inquiry
import datamining
import classify

__all__=['User','TableFiller','UserAsyncModel']


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

    @property
    def bussiness_type(self):
        return classify.type_dic

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

        #业务数据库
        self._data = datamining.DataMiningAgent()

        #进程锁
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

        #异步模块
        self.__get_answer_res=UserAsyncModel(self.__get_answer)
        self.__inquire_res = UserAsyncModel(self.__inquire)
        self.__classify_res=UserAsyncModel(self.__classify)
        self.__get_flow_res=UserAsyncModel(self.__get_flow)

        self.__activate_res=UserAsyncModel(self.__activate)



        #检查必要信息
        for key in default_nes_info:
            if key not in info_dic:
                raise Exception("Missing necessary information")

        #检查多于信息
        return

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



    def _check_init(self)->bool:
        if self.init_finish:
            return True
        else:
            raise Exception("initialization of this user is not finished,please execute fuction activate if you didn't do it")

    @property
    def init_finish(self)->bool:
        return self.__init

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
    def label(self):
        self._check_init()
        return self.__label

    @property
    def bus_type(self):
        """
        用户要办理的业务类型，将会把label对应到相应的文本
        :return:
        """
        invert_dic = dict(zip(classify.type_dic.values(), classify.type_dic.keys()))
        return invert_dic[self.label]

    @property
    def tables(self):
        """
        返回所有的表格
        :return:
        """
        return self.tables_filler.tables

    @property
    def flow(self)->str:
        """
        此用户的业务流程
        :return: 用户流程的纯文本
        """
        self._check_init()

        if self.__flow=="":
            return ""
        return self.__flow

    @property
    def tables_filler(self)->TableFiller:
        """
        一个TableFiller类型的变量，用于填充表格
        :return: 填表迭代器
        """
        self._check_init()
        return self.__tables_filler


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


    async def __inquire(self):
        """
        对当前表进行询问，inquire的异步封装
        :return:
        """
        self._check_init()

        question = inquiry.Inquiring(self.tables_filler.table)
        if question is None:
            return None
        return question.keys()[0],question.values()[0]         #key,sentence
    def inquire(self):
        return self.__inquire_res.activate()


    async def __get_answer(self,text,key):
        """
        对用户自然语言回答进行关键字提取，get_answer的异步封装
        :param text: 用户自然语言
        :param key: 问题键
        :return: 提取的关键字
        """
        return inquiry.Get_Answer(text, key)
    def get_answer(self,text,key):
        """
        对用户自然语言回答进行关键字提取
        :param text: 用户自然语言
        :param key: 问题键
        :return: 结束时返回用户回答的关键字，过程中详见UserAsyncModel
        """
        return self.__get_answer_res.activate(text,key)


    async def __classify(self,require):
        """
        用户业务分类的异步封装
        :param require: 用户需求的自然语言
        :return: 返回用户业务类别对应的label值
        """
        return classify.classify(require, classify.type_dic)
    def classify(self,require):
        """
        用户业务分类，返回异步模块对象
        :param require: 用户需求的自然语言
        :return: 结束时返回用户业务类别对应的label值，过程中详见UserAsyncModel
        """
        return self.__classify_res.activate(require, classify.type_dic)


    async def __get_flow(self):
        self._data.get_flow(self.__label, self.info)



    # 一个流程的主程序
    def activate(self,require):
        return self.__activate_res.activate(require)
    async def __activate(self, require):
        #分类
        self.__label = await asyncio.create_task(self.__classify(require))

        # 提取空表+流程

        #提取空表
        self.__tables_filler = TableFiller(self._data.get_tables(self.__label))
        self.__tables_filler.reset_pointer()

        #提取流程
        self.__flow = await asyncio.create_task(self.__get_flow())



        #到此处为初始化过程，以上信息是必要的
        self.__init = True

        while not self.__tables_filler.is_finish:
            self.__tables_filler.table = self._fill_in_table(self.__tables_filler.table)

            #填完表等待服务器录入信息
            self.waiter.wait()

        return True



class UserAsyncModel:
    processing = -2147483648


    def __init__(self, task):
        """
        创建一个异步操作模块
        :param task: 一个函数，必须为async的异步函数
        """
        self.result = -2147483648

        self.__task = task

    def __call__(self, *args):
        return asyncio.run(self.activate(*args))


    async def __activate(self,*args):
        """
        调用异步函数，在异步结束前会返回processing=-2147483648
        :param args: 原函数的参数
        :return: 原函数的返回值或者processing=-2147483648
        """

        #无异步操作
        if self.result is -2147483648:
            self.result = asyncio.create_task(self.__task(*args))
        #异步操作已完成
        elif self.result.done():
            value = await self.result
            # 重置异步器
            self.result = -2147483648
            return value
        #异步进行中
        else:
            return -2147483648

    def activate(self,*args):
        return asyncio.run(self.__activate(*args))

    async def wait_done(self,*args):
        return await self.result