import http
import json
import threading
import hashlib
import requests
import http.server as httpserver
import datetime


from user import User,TableFiller,UserAsyncModel

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
            threading.Thread(target=user.activate, args=(request["input"]["text"],)).start()


        # 获取用户
        user = flow_dic[user_id]


        # 在握手前使用，报错
        if user_id not in flow_dic:
            response["type"] = "error"
            response["message"]="use this user_in before handshake"
            return response
        #用户正在初始化（处理信息）
        if not user.init_finish:
            response["type"] = "processing"
            return response




        # 初始化握手响应
        if req_type == "handshake":
            log("build a new flow,user id:%s" % user_id)

            response["type"] = "handshake"
            response["classify"] = user.bus_type
            response["flow"] = user.flow

        #询问
        elif req_type == "question":
            log("user")

            # 获取问题
            question = user.inquire()
            # 异步是否完成
            if question is int and question ==UserAsyncModel.processing:
                response["type"] = "processing"
                return response


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
            #解析请求体
            key, text, value = "", "", ""
            if request["input"]["type"] == "text":
                key = request["input"]["key"]
                text = request["input"]["text"]
            elif request["input"]["type"] == "audio":
                #key = request["input"]["key"]
                #text = voice.voice2text(request["input"]["media"])

            #ai分析用户语言，提取关键字
            value = user.get_answer(text,key)
            #异步是否完成
            if value is UserAsyncModel.processing:
                response["type"] = "processing"
                return response

            response["type"] = "answer"
            response["output"] = {
                "type": "text",
                "key": key,
                "text": text,
                "value": value
            }
            user[key] = value
            #信息已更新，录入
            user.waiter.set()


        #数据库储存请求
        elif req_type == "storage":
            response["type"] = "storage"
            response["output"] = user.main_info
        elif req_type == "summary":
            response["type"] = "summary"
            response["output"] = {
                "tables":user.tables_filler.tables,
                "classify":user.classify,
                "flow":user.flow
            }
        return response

def log(message:str,model:str="Undefined"):
    model = model if not model is None else __name__

    if model == "__main__":
        model = "main"
    elif model == "server.console":
        model = "Server"

    print("<%s>:[%s]%s"%(model,datetime.datetime.now().strftime('%H:%M:%S'),message))


def run():
    web = httpserver.ThreadingHTTPServer((host, port), Handler)
    log("a new http listener open on %s"%ip)
    web.serve_forever()
    while True:
        command = input("console@%s:~$"%ip)