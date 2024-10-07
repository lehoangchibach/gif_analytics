import os
import pickle as pk

files = os.listdir("/home/lehoangchibach/Documents/Code/gif_analytics")
files.extend(os.listdir("/d/gifs"))

gen = (file.split(".")[0] for file in files)
existed = set(gen)
print("existed gifs:", len(existed))
with open("data/existed.pickle", "wb") as f:
    pk.dump(existed, f)
