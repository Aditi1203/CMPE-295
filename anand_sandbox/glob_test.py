from glob import glob

def get_paths():
    paths = glob("ecg-id-database/Person_**/*.atr")
    for path in paths:
        print(path)

get_paths()