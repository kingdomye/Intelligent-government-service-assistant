import http
import json
import threading
import hashlib
import requests
import http.server as httpserver
import datetime

from user import User

host="127.0.0.1"
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


        content_length = int(self.headers.get("Content-Length", 0))
        content_dic = json.loads(self.rfile.read(content_length).decode("utf8"))


        user_id = content_dic["user_id"]
        req_type = content_dic["type"]

        respond = {
            "user_id": user_id,
            "type": "question",
            "hash": "a1dadafgdas3asd4s2sfad",
        }


        if req_type == "handshake":
            log("(%s)a new user handshake" % ip)

            info_dic = content_dic["info"]
            input_dic = content_dic["input"]

            user = User(info_dic)
            flow_dic[user_id] = user


            log("(%s) build a new flow,user id:%s"%(ip,user_id))


            threading.Thread(target=user.activate, args=(input_dic["text"],)).start()

        elif req_type == "question":
            log("(%s)user" % ip)

            user = flow_dic[user_id]

            # 获取问题
            question = user.inquire()

            # 判断问完了没
            if question is None:
                respond["output"] = {"type": None}
            else:
                # 响应问题-文本
                key,sentence = question
                respond["output"] = {
                        "type":"text",
                        "key":key,
                        "text":sentence
                    }

        elif req_type == "answer":
            user = flow_dic[content_dic["user_id"]]
            user.waiter.set()

        elif req_type == "storage":
            user = flow_dic[content_dic["user_id"]]
            user.waiter.set()

        self.wfile.write(json.dumps(respond).encode("utf8"))
        self.wfile.flush()



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