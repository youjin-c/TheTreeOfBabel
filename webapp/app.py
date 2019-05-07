import json
import sys,os
from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

def datalist(path):
    for entry in sorted(os.scandir(path), key=lambda x: (x.is_dir(), x.name)):
        print(entry)
    #     if entry.name.split('.')[0] .isdigit():
    #         with open(entry,'rt') as jsonfile:
    #             jsons = json.load(jsonfile)
    #             x = torch.tensor(jsons['x'], dtype=torch.float)
    #             edge_index = torch.tensor(jsons['edge_index'],dtype=torch.long)
    #             data = Data(x=x, edge_index=edge_index)# print(entry.name.split('.')[0],data)
    #             data_list.append(data)
    #             filename_list.append(entry.name)
    # return data_list,filename_list

path = "dataset"
datalist(path)
@app.route("/")
def index():
    df = pd.read_csv('data.csv').drop('Open', axis=1)
    chart_data = df.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return render_template("index.html", data=data)
    # return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
