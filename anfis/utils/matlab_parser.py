import os
import sys


def read_mathlab_anfis_structure(file_name="anfis/data/iris/4x3_gaussmf_linear.fis"):
    data_dict = dict()
    real_path = os.path.realpath(file_name)

    parts = []

    with open(real_path, "r") as f:
        data = f.read()
        data = data.split("\n")

    part = []

    for d in data:
        if not d == "":
            part.append(d)
        else:
            parts.append(part)
            part = []

    for p in parts:
        key = p[0][1:-1]
        data_dict[key] = p[1:]

    for k, v in data_dict.items():
        if k == "System":
            data_dict[k] = parse_system(v)
        elif k.startswith("Input"):
            data_dict[k] = parse_input_output(v)
        elif k.startswith("Output"):
            data_dict[k] = parse_input_output(v)
        elif k.startswith("Rules"):
            data_dict[k] = parse_rules(v)

    result = {
        "inputs": [data_dict[k] for k, v in data_dict.items() if k.startswith("Input")],
        "outputs": [data_dict[k] for k, v in data_dict.items() if k.startswith("Output")],
        "rules": [data_dict[k] for k, v in data_dict.items() if k.startswith("Rules")],
        "agg": data_dict["System"]["AggMethod"][1:-1],
        "and": data_dict["System"]["AndMethod"][1:-1],
        "defuzz": data_dict["System"]["DefuzzMethod"][1:-1],
        "implication": data_dict["System"]["ImpMethod"][1:-1],
        "or": data_dict["System"]["OrMethod"][1:-1],
        "type": data_dict["System"]["Type"][1:-1],
        "version": data_dict["System"]["Version"]
    }

    return result


def parse_system(data):
    result = dict()
    for d in data:
        dd = d.split("=")
        result[dd[0]] = dd[1]
    return result


def parse_input_output(data):
    result = {"range": [], "mfs": []}
    for d in data:
        dd = d.split("=")
        if dd[0] == "Range":
            result["range"] = [float(r) for r in dd[1][1:-1].split(" ")]
        if dd[0].startswith("MF"):
            result["mfs"].append(parse_mf(dd[1]))
    return result


def parse_rules(data):
    result = []
    for d in data:
        dd = d.split(", ")
        result.append({
            "connections": [int(i) for i in dd[0].split(" ")],
            "number": dd[1].split(" : ")[0],
            "value": dd[1].split(" : ")[1]
        })
    return result


def parse_mf(data):
    d = data.split(",")
    func = d[0].split(":")[1][1:-1]
    params = [float(i) for i in d[1][1:-1].split(" ")]
    return [func, params]


# if __name__ == "main":
res = read_mathlab_anfis_structure("../data/iris/4x3_gaussmf_linear.fis")

print(res)
