def safemkdir(path):
    try:
        os.mkdir(path)
    except:
        pass
