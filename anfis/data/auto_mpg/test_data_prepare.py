from random import shuffle

with open("auto-mpg.data-original", "r") as f:
    data = f.read()
    data = data.split("\n")

prepared_data = []

for d in data:
    try:
        dd = d.split("\t")[0].split(" ")
        dddd = d.split("\t")[0]
        print("", dddd)
        ddd = [
            # str(float(dd[3])),          # cylinders
            # str(float(dd[6])),          # displacement
            # str(float(dd[12])),         # horsepower
            str(float(dd[18])),         # weight
            # str(float(dd[24])),         # acceleration
            str(float(dd[27])),         # year
            # str(float(dd[29])),         # origin
            str(float(dd[0])),            # mpg
        ]
        print(ddd)
        prepared_data.append(ddd)
    except Exception as e:
        print(e)

# prepared_data.sort(key=lambda x: x[5])

with open("carTrain.dat", "w") as f:
    for d in prepared_data:
        f.write("\t".join(d) + "\n")
