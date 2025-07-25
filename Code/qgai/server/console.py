import asyncio
import http
import json
import threading
import hashlib
import requests
import http.server as httpserver
import datetime

import inspect


from user import User,TableFiller,UserAsyncModel

__all__=['log']

host="192.168.1.229"
port=10925
ip=host+":"+str(port)



flow_dic = {

}

hash_obj = hashlib.sha256()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        return
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        # 请求内容长度
        content_length = int(self.headers.get("Content-Length", 0))

        # 请求内容
        request = json.loads(self.rfile.read(content_length).decode("utf8"))
        self.wfile.write(json.dumps(self.respond(request)).encode("utf8"))
        self.wfile.flush()


    def processing_response(self,user_id):
        log("\"%s\" is request but processing"%(user_id))
        response = {
            "user_id": user_id,
            "type": "processing",
            "hash": "a1dadafgdas3asd4s2sfad",
        }

        return response

    def respond(self,request)->dict:

        # 用户id
        user_id = request["user_id"]
        # 请求类型
        req_type = request["type"]
        # 响应必要信息
        response = {
            "user_id": user_id,
            "type": "busy",
            "hash": "a1dadafgdas3asd4s2sfad",
        }

        # 握手时创建新用户
        if user_id not in flow_dic and req_type == "handshake":
            log("(%s)a new user handshake" % ip)

            user = User(request["info"])
            flow_dic[user_id] = user

            user = flow_dic[user_id]
            user.run_on_new_thread(request["input"]["text"])


        # 获取用户
        user = flow_dic[user_id]


        # 在握手前使用，报错
        if user_id not in flow_dic:
            response["type"] = "error"
            response["message"]="use this user_in before handshake"
            return response
        #用户正在初始化（处理信息）
        if not user.init_finish:
            return self.processing_response(user_id)




        # 初始化握手响应
        if req_type == "handshake":
            log("handshake----user_id:%s" % user_id)

            response["type"] = "handshake"
            response["classify"] = user.bus_type
            response["flow"] = user.flow

        #询问
        elif req_type == "question":
            log("question----user_id:%s" % user_id)

            # 获取问题
            question = user.inquire()
            # 异步是否完成
            if not user.inquire.done:
                return self.processing_response(user_id)



            # 判断这张表的信息
            if question is None:
                #此表填完了，换下一张表
                user.tables_filler.export_table()
                user.waiter.set()

            response["type"] = "question"
            if user.tables_filler.is_finish:
                response["output"] = {"type": None}
            else:
                # 响应问题-文本
                key, sentence = question
                response["output"] = {
                    "type": "text",
                    "key": key,
                    "text": sentence
                }





        #解析回答
        elif req_type == "answer":
            log("answer----user_id:%s" % user_id)
            #解析请求体
            key, text, value = "", "", ""
            if request["input"]["type"] == "text":
                key = request["input"]["key"]
                text = request["input"]["text"]
            elif request["input"]["type"] == "audio":
                1 == 1
                #key = request["input"]["key"]
                #text = voice.voice2text(request["input"]["media"])


            #ai分析用户语言，提取关键字
            value = user.get_answer(text,key)
            #异步是否完成
            if not user.get_answer.done:
                return self.processing_response(user_id)
            print(value)

            response["type"] = "answer"
            response["output"] = {
                "type": "text",
                "key": key,
                "text": text,
                "value": value
            }

            #信息已更新，录入
            user[key] = value
            user.fill_in_table()


        #数据库储存请求
        elif req_type == "storage":
            response["type"] = "storage"
            response["output"] = user.main_info


        elif req_type == "summary":
            flow = user.get_flow(user.info)

            response["type"] = "summary"
            response["output"] = {
                "tables":user.tables_filler.tables,
                "classify":user.classify,
                "flow":user.flow
            }
        return response

def log(message:str,model:str="Undefined"):
    model = model if not model == "Undefined" else inspect.getmodule(inspect.currentframe()).__name__

    if model == "__main__":
        model = "main"
    elif model == "server.console":
        model = "Server"
    elif model.split(".")[0] == "datamining":
        model = "Datamining"
    elif model.split(".")[0] == "inquiry":
        model = "Inquiry"
    elif model.split(".")[0] == "classify":
        model = "Classify"

    print("[%s]<%s>:%s"%(datetime.datetime.now().strftime('%H:%M:%S'),model,message))


def run():
    web = httpserver.ThreadingHTTPServer((host, port), Handler)
    log("a new http listener open on %s"%ip)
    web.serve_forever()
    while True:
        command = input("console@%s:~$"%ip)