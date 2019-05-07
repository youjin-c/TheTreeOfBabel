import json
import sys,os

from flask import Flask, render_template

app = Flask(__name__)

def datalist(path):
    cat = []
    data = []
    for i, folder in enumerate(sorted(os.scandir(path), key=lambda x: (x.is_dir(), x.name))):
        cat.append(folder.name)
        data.append([])
        for j,files in enumerate(sorted(os.scandir(folder.path), key=lambda x: (x.is_dir(), x.name))):
            with open(files.path,'rt') as onefile:
                jsons =json.load(onefile)
                data[i].append(jsons)
    # print(data[0][0]['x'])
    # print(data[0][0]['edge_index'])
    return data



@app.route("/")
def index():
    data = datalist("static/dataset")
    return render_template("index.html", data=data)


if __name__ == "__main__":
    app.run(debug=True)
