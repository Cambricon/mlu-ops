#!/usr/bin/python

import json

json_file = 'build.property'
output = 'dependency.txt'
modules = ["cntoolkit"]


with open(json_file) as json_data:
    data = json.load(json_data)

text_file= open(output, "w")

for key in modules:
    value=data["build_requires"][key]
    text_file.write("%s:%s:%s\n"%(key, value[0], value[1]))

text_file.close()
