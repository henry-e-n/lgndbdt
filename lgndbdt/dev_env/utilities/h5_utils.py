import h5py
import os

def openGroup(group, kList):
    for key in group.keys():
        # print(f"{group.name}/{key}")
        kList.append(f"{group.name}/{key}")
        if type(group[key])==h5py._hl.group.Group:
            if len(group[key].keys())>0:
                kList = openGroup(group[key], kList)
    return kList

def searchFile(name, path):
    try: 
        for root, dirs, files in os.walk(path):
                if name in files:
                    return os.path.join(root, name)
    except FileNotFoundError:
        return