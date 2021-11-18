import pickle

path_base = f'../{name}_{GPU_TYPE}'
with open(path_base + '.pickle', 'rb') as f:
    d = pickle.load(f)
    print(d[-1])