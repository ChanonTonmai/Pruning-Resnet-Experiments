#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 18:27:00 2025

@author: khongpra
"""



import matplotlib.pyplot as plt
import numpy as np

# ic_list_p = [1269988, 1371332, 352322, 1075802, 352322, 1318836, 1391834, 352322, 1378636, 
#              352322, 1336516, 1384632, 1259384, 352322, 1368236, 1333318, 1303988, 1221112, 
#              1023426, 1346888, 1304600, 1229178, 1325906, 1364808, 1320636, 1333894, 1397446, 
#              1348580, 1355440, 1146674, 1259198, 1374244, 1268074, 1361808, 1197778, 1302692, 
#              352322, 1048908, 352322, 1342074, 1341636, 1282232, 1262108, 1341882, 1312546, 
#              1175500, 1218622, 1260782, 352322, 1290202, 1321632, 1282796, 1312048, 1312570, 
#              1270296, 1224952, 1348196, 1338798, 1163202, 1260074, 1254240, 1264678, 1130794, 1284926]



# plt.figure(figsize=(15, 7))
# plt.bar(range(len(ic_list_p)), ic_list_p)
# plt.axhline(y=1556546, color='r', linestyle='--', label='x=150000')
# plt.xlabel("#PE")
# plt.ylim(0, 1.6e6)
# plt.ylabel("Cycles")
# plt.title("Executed Cycle in each PE (Schedule 1 filter per 1 PE) for 50% prunned version")
# plt.tight_layout()
# plt.savefig("bar_chart.png")

# 64 PE
# mem_conflict = [71689, 74991, 27792, 66980, 29029, 76794, 85368, 29428, 86227, 30209, 
#                 83714, 89537, 83733, 31357, 86716, 83397, 104916, 102626, 89936, 
#                 107737, 110317, 107523, 111980, 128861, 121705, 124526, 117587, 
#                 133673, 136369, 110196, 125071, 126268, 88949, 97113, 87770, 94059, 
#                 34802, 83948, 35752, 95270, 106365, 103546, 99287, 108812, 103702, 
#                 100116, 103597, 108428, 35698, 99749, 113422, 101154, 106611, 116064, 
#                 100060, 100506, 106055, 108700, 100197, 108250, 119508, 108654, 105281, 
#                 123182]
# ic_real = [1341729, 1446377, 380168, 1142834, 381406, 1395683, 1477255, 381805, 
    # 1464916, 382586, 1420285, 1474222, 1343170, 383734, 1455005, 1416768, 
    # 1408957, 1323793, 1113417, 1454678, 1414970, 1336756, 1437941, 1493722, 
    # 1442394, 1458473, 1515086, 1482306, 1491862, 1256923, 1384322, 1500565, 
    # 1357076, 1458976, 1285601, 1396806, 387179, 1132909, 388129, 1437399, 1448054, 
    # 1385833, 1361450, 1450749, 1416301, 1275669, 1322274, 1369263, 388075, 1390004, 
    # 1435107, 1384003, 1418712, 1428687, 1370409, 1325513, 1454304, 1447551, 1263454, 
    # 1368377, 1373803, 1373387, 1236128, 1408161]

# 16 PE
# ic_real = [4405273, 3711015, 4803728, 4671596, 5287054, 5620373, 5792070, 5534072, 5530677,
#              3378484, 5628383, 5371356, 4634982, 5555343, 5535491, 5370052]

# mem_conflict = [335777, 295646, 351569, 358283, 391587, 395828, 391461, 398463, 
#                 400270, 282803, 400470, 403853, 387977, 435422, 425168, 435361]

# ic_real_no_conflict = np.array(ic_real) - np.array(mem_conflict)



# plt.figure(figsize=(15, 7))

# plt.bar(range(len(ic_real)), ic_real_no_conflict)
# plt.bar(range(len(ic_real)), mem_conflict, bottom=ic_real_no_conflict)
# plt.xlabel("PE")
# plt.ylabel("Cycles")
# plt.legend(["Execution", "Memory Conflict"], loc='lower left')
# plt.axhline(y=6803451, color='r', linestyle='--', label='x=150000')
# plt.title("Executed Cycle in each PE (Schedule 4 filter per 1 PE) for 50% prunned version with XVI-V ZD ISA")
# plt.show()

# max_ic_real = max(ic_real)
# speedup = 1+((6803451-max_ic_real)/6803451)
# print(f"speedup = {speedup}")



ic_balanced = [5386923,5384490,5459995,5348740,4664966,4614872,5493623,
                5522508,4671513,4691844,5522545,4643781,4622119,5482893,
                4608001,4624638]

plt.figure(figsize=(15, 7))

plt.bar(range(len(ic_balanced)), ic_balanced)
# plt.bar(range(len(ic_real)), mem_conflict, bottom=ic_real_no_conflict)
plt.xlabel("PE")
plt.ylabel("Cycles")
# plt.legend(["Execution", "Memory Conflict"], loc='lower left')
plt.axhline(y=6803451, color='r', linestyle='--', label='x=150000')
plt.title("Simulated Cycle in each PE (Schedule 4 filter per 1 PE) for 50% prunned version with XVI-V ZD ISA with balance load")
plt.show()

max_ic_real = max(ic_balanced)
speedup = 1+((6803451-max_ic_real)/6803451)
print(f"speedup = {speedup}")