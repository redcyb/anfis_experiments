import json
import os


def read_mathlab_anfis_structure(file_name, to_file=False):
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

    if to_file:
        with open(f"{real_path}.json", "w") as ff:
            ff.write(json.dumps(result))

        # === FOR GAUSSIAN ===

        with open(f"{real_path}_old_ver.json", "w") as ff:
            inputs = result["inputs"]
            inputs_new = [
                [{"mean": j[1][0], "sigma": j[1][1]} for j in i["mfs"]]
                for i in inputs]
            ff.write(json.dumps(inputs_new))

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
            "connections": [(int(i) - 1) for i in dd[0].split(" ")],
            "number": dd[1].split(" : ")[0],
            "value": dd[1].split(" : ")[1]
        })
    return result


def parse_mf(data):
    d = data.split(",")
    func = d[0].split(":")[1][1:-1]
    params = [float(i) for i in d[1][1:-1].split(" ")]
    params.reverse()
    return [func, params]


if __name__ == "__main__":
    read_mathlab_anfis_structure("../data/iris/4x3_gaussmf_linear___before_training.fis", True)
