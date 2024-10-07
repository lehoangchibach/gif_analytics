import json

with open("./metadata.json", "r") as f:
    metadata = json.load(f)


print("metadata:", len(metadata))

s = set()
for data in metadata:
    s.add(data["href"])
print("unique:", len(s))
