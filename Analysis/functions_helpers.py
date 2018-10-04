import os 

def safemkdir(path):
    try:
        os.mkdir(path)
    except:
        pass
