import threading
import json
import asyncio
import time

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

async def testtt():
    print("testtt")

class UserAsyncModel:
    """
    在其他线程的异步操作上分支出一个新的异步的操作，并在执行时加入循环
    """
    processing = -2147483648


    def __init__(self, task,loop,wait_time=0.5):
        """
        创建一个异步操作模块
        :param task: 一个函数，必须为async的异步函数
        """
        self.__result = None

        self.__task = task
        self.__loop = loop

        self.wait_time = wait_time

        self.index=0

    def __call__(self,*args):
        return self.activate(*args)

    @property
    def __get_task(self):
        return self.__task

    async def test(self):
        print("test")
        return 1

    def activate(self,*args):
        """
        调用异步函数，在异步结束前会返回processing=-2147483648
        :param args: 原函数的参数
        :return: 原函数的返回值或者processing=-2147483648
        """
        #无异步操作，则创建
        if self.__result is None:
            print("c async")
            if self.index >= 0:
                self.__result = asyncio.run_coroutine_threadsafe(self.__task(*args),self.__loop)
                self.index+=1
            else:
                self.__result = asyncio.run_coroutine_threadsafe(testtt(), self.__loop)


        time.sleep(self.wait_time)

        print(self.__result.running())
        #异步操作已完成
        if self.__result.done():
            print("d async")
            value = self.__result.result()
            # 重置异步器
            self.__result = None
            return value
        #异步进行中
        else:
            print("i async")
            return -2147483648

    @property
    def done(self):
        if self.__result is None:
            return True
        return self.__result.done()

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

        self.__loop = asyncio.new_event_loop()
        #异步模块
        self.__get_answer_res=UserAsyncModel(self.__get_answer,self.__loop)
        self.__inquire_res = UserAsyncModel(self.__inquire,self.__loop)
        self.__classify_res=UserAsyncModel(self.__classify,self.__loop)
        self.__get_flow_res=UserAsyncModel(self.__get_flow,self.__loop)
        self.__activate_res=UserAsyncModel(self.__activate,self.__loop)



        #进程锁
        self.__process_block = None

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
        """
        检测activate初始化的进程，如果未初始化完毕将会报错
        :return:
        """
        if self.init_finish:
            return True
        else:
            raise Exception("initialization of this user is not finished,please execute fuction activate if you didn't do it")

    @property
    def init_finish(self)->bool:
        """
        获取一个bool值，表示是否初始化完毕
        :return:
        """
        return self.__init

    @property
    def info(self)->dict:
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
        """
        此方法将会把现在用户已有的所有，表格需要的信息填入表中，将会对传入的参数做出改变
        :param table: 所填表格，需要填写的值为None
        :return: 返回一个填完的表格
        """
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


    @property
    def _waiter(self)->asyncio.Event:
        """
        此用户流程的进程锁
        :return:
        """
        return self.__process_block
    async def __set_waiter(self):
        self._waiter.set()
    def set_waiter(self):
        """
        使进程锁解除阻塞状态，程序继续运行
        :return:
        """
        asyncio.run_coroutine_threadsafe(self.__set_waiter(),self.__loop)


    async def __inquire(self):
        """
        对当前表进行询问，inquire的异步封装
        :return:
        """
        print("inquire")
        self._check_init()

        question = inquiry.Inquiring(self.tables_filler.table)
        if question is None:
            return None
        return list(question.keys())[0],list(question.values())[0]         #key,sentence
    @property
    def inquire(self)->UserAsyncModel:
        """
        询问模块的异步模块封装
        :return:
        """
        return self.__inquire_res
    def inquire_func(self):
        """
        对当前table_filler指向的表进行询问，本质为异步操作
        :return:
        """
        return self.__inquire_res.activate()


    async def __get_answer(self,text,key):
        """
        对用户自然语言回答进行关键字提取，get_answer的异步封装
        :param text: 用户自然语言
        :param key: 问题键
        :return: 提取的关键字
        """
        print("get_answer")
        return inquiry.Get_Answer(text, key)
    @property
    def get_answer(self)->UserAsyncModel:
        """
        提取回答模块的异步模块封装
        :return:
        """
        return self.__get_answer_res
    def get_answer_func(self,text,key):
        """
        对用户自然语言回答进行关键字提取，本质为异步操作
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
        print("classify")
        return classify.classify(require, classify.type_dic)
    @property
    def classify(self)->UserAsyncModel:
        """
        用户业务分类的异步模块封装
        :return:
        """
        return self.__classify_res
    def classify_func(self,require):
        """
        用户业务分类，本质为异步操作
        :param require: 用户需求的自然语言
        :return: 结束时返回用户业务类别对应的label值，过程中详见UserAsyncModel
        """
        return self.__classify_res.activate(require)


    async def __get_flow(self):
        """
        获取流程的异步封装
        :return:
        """
        return self._data.get_flow(self.__label, self.info)
    @property
    def get_flow(self)->UserAsyncModel:
        """
        获取流程模块的异步模块封装
        :return:
        """
        return self.__get_flow_res
    def get_flow_func(self):
        """
        获取用户当前分类的流程，本质为异步操作
        :return:
        """
        return self.__get_flow_res.activate(self.__label, self.info)


    # 一个流程的主程序
    def activate(self,require):
        asyncio.set_event_loop(self.__loop)
        self.__process_block = asyncio.Event()
        self.__loop.run_until_complete(self.__activate(require))
        self.__loop.close()

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
            await self._waiter.wait()
            self._waiter.clear()

        return True

    def run_on_new_thread(self,require):
        threading.Thread(target=self.activate,args=(require,)).start()



