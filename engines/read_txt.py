import re
import matplotlib.pyplot as plt
losses = []
iters = []
j = 0
with open(r'E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transs_effiunet.txt',encoding="utf8") as f:
    lines = f.readlines()
    for i,line in enumerate(lines):
        if line[:9] == "iteration":
            j+=1
            splitted_line = re.split(" ",line)
            loss = float(splitted_line[4][:-1])
            losses.append(loss)
            iters.append(j)
print(len(iters))
print(len(losses))

plt.figure()
plt.xlabel('Iters')
plt.ylabel('Total Loss')
plt.plot(iters, losses)
plt.savefig(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\transs_effiunet125_txt.jpg")