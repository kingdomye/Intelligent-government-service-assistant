import datetime
import inspect


__all__=['log']





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
    elif model.split(".")[0] == "face":
        model = "Face_rec"

    print("[%s]<%s>:%s"%(datetime.datetime.now().strftime('%H:%M:%S'),model,message))


