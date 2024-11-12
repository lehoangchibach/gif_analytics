import json

with open("./metadata_w_examples.json", "r") as f:
    metadata = json.load(f)


print("metadata:", len(metadata))

href = set()
template_id = set()
for data in metadata:
    href.add(data["href"])
    template_id.add(data["html_file"])
print("unique href:", len(href))
print("unique html_file:", len(template_id))
print("data[0]: ", metadata[0])
