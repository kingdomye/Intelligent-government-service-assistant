import http
import json
import threading
import hashlib
import requests
import http.server as httpserver
import datetime

import inquiry
import voice
from user import User,TableFiller

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

        #初始化握手响应
        if req_type == "handshake":
            log("(%s)a new user handshake" % ip)
            response["type"] = "handshake"

            info_dic = request["info"]
            input_dic = request["input"]

            user = User(info_dic)
            flow_dic[user_id] = user

            log("(%s) build a new flow,user id:%s" % (ip, user_id))

            threading.Thread(target=user.activate, args=(input_dic["text"],)).start()
            return response


        # 获取用户
        user = flow_dic[user_id]

        #用户正在初始化（处理信息）
        if not user.init_finish:
            response["type"] = "processing"
            return response



        if req_type == "question":
            log("(%s)user" % ip)

            # 获取问题
            question = user.inquire()

            # 判断问完了没
            if question is None:
                response["output"] = {"type": None}
                user.tables_filler.export_table()

            else:
                # 响应问题-文本
                key, sentence = question
                response["output"] = {
                    "type": "text",
                    "key": key,
                    "text": sentence
                }

        elif req_type == "answer":
            #解析请求体
            key, text, value = "", "", ""
            if request["input"]["type"] == "text":
                key = request["input"]["key"]
                text = request["input"]["text"]
            elif request["input"]["type"] == "audio":
                key = request["input"]["key"]
                text = voice.voice2text(request["input"]["media"])

            #ai分析用户语言，提取关键字
            value = inquiry.Get_Answer(text,key)


            response["output"] = {
                "type": "text",
                "key": key,
                "text": text,
                "value": value
            }

            user[key] = value
            user.waiter.set()

        elif req_type == "storage":
            user = flow_dic[request["user_id"]]
            user.waiter.set()

        return response

def log(message:str,model:str="Undefined"):
    model = model if not model is None else __name__

    if model == "__main__":
        model = "main"
    elif model == "server.console":
        model = "Server"

    print("<%s>[%s]:%s"%(model,datetime.datetime.now().strftime('%H:%M:%S'),message))


def run():
    web = httpserver.ThreadingHTTPServer((host, port), Handler)
    log("(%s)a new http listener open on %s"%(ip,port))
    print("<http_server>" + ip + ":a new http listener open on " + str(port))
    web.serve_forever()