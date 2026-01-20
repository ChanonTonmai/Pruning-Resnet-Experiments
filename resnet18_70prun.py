manual_conv_1 = [
    1062125568, # conv1
    4161798144, # layer1.0.conv1
    4161798144, # layer1.0.conv2
    4161798144, # layer1.1.conv1
    4161798144, # layer1.1.conv2
    2080899072, # layer2.0.conv1
    4161798144, # layer2.0.conv2
    57802752,   # layer2.0.downsample.0
    1040449536, # layer2.1.conv1
    1040449536, # layer2.1.conv2
    520224768,  # layer3.0.conv1
    1040449536, # layer3.0.conv2
    14450688,   # layer3.0.downsample.0
    260112384,  # layer3.1.conv1
    260112384,  # layer3.1.conv2
    169869312,  # layer4.0.conv1
    339738624,  # layer4.0.conv2
    4718592,    # layer4.0.downsample.0
    84934656,   # layer4.1.conv1
    84934656    # layer4.1.conv2
]

prunned_conv = [
    734324696,  # conv1
    2337806538, # layer1.0.conv1
    2567806462, # layer1.0.conv2
    2643505786, # layer1.1.conv1
    2592965390, # layer1.1.conv2
    1233059688, # layer2.0.conv1
    2187213378, # layer2.0.conv2
    39770752,   # layer2.0.downsample.0
    546345404,  # layer2.1.conv1
    529345692,  # layer2.1.conv2
    250139190,  # layer3.0.conv1
    443429382,  # layer3.0.conv2
    7777672,    # layer3.0.downsample.0
    97193370,   # layer3.1.conv1
    91582478,   # layer3.1.conv2
    55223104,   # layer4.0.conv1
    98668036,   # layer4.0.conv2
    2572224,    # layer4.0.downsample.0
    22292160,   # layer4.1.conv1
    15130198    # layer4.1.conv2
]

speedups = [
    0.30862723003387865, # conv1
    0.4382700801166007,  # layer1.0.conv1
    0.38300552473887595, # layer1.0.conv2
    0.3648164340187663,  # layer1.1.conv1
    0.37696031852524176, # layer1.1.conv2
    0.40743897453187006, # layer2.0.conv1
    0.47445471829207486, # layer2.0.conv2
    0.3119574652777778,  # layer2.0.downsample.0
    0.47489485544832805, # layer2.1.conv1
    0.49123366998166457, # layer2.1.conv2
    0.5191709326688575,  # layer3.0.conv1
    0.5738098132997774,  # layer3.0.conv2
    0.4617784288194444,  # layer3.0.downsample.0
    0.6263408588804446,  # layer3.1.conv1
    0.6479118887319106,  # layer3.1.conv2
    0.6749082965615355,  # layer4.0.conv1
    0.7095766302980022,  # layer4.0.conv2
    0.4548746744791667,  # layer4.0.downsample.0
    0.7375375253182871,  # layer4.1.conv1
    0.821860725497022    # layer4.1.conv2
]



import numpy as np

print(np.mean(speedups))