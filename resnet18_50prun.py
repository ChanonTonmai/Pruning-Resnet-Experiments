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

prunned_conv_2 = [
    803833652,  # conv1
    2792214508, # layer1.0.conv1
    3107127106, # layer1.0.conv2
    3154375718, # layer1.1.conv1
    3122509916, # layer1.1.conv2
    1512825618, # layer2.0.conv1
    2823031594, # layer2.0.conv2
    46412800,   # layer2.0.downsample.0
    698763808,  # layer2.1.conv1
    688926746,  # layer2.1.conv2
    331809748,  # layer3.0.conv1
    611247592,  # layer3.0.conv2
    9978360,    # layer3.0.downsample.0
    137498090,  # layer3.1.conv1
    132466312,  # layer3.1.conv2
    79477106,   # layer4.0.conv1
    148645336,  # layer4.0.conv2
    3288032,    # layer4.0.downsample.0
    31725588,   # layer4.1.conv1
    23939752    # layer4.1.conv2
]

speedups = [
    0.24318397351677348, # conv1
    0.3290845900285932,  # layer1.0.conv1
    0.25341715323711766, # layer1.0.conv2
    0.2420642210753026,  # layer1.1.conv1
    0.24972096003702768, # layer1.1.conv2
    0.2729942367911249,  # layer2.0.conv1
    0.3216798373390788,  # layer2.0.conv2
    0.1970486111111111,  # layer2.0.downsample.0
    0.3284020187212617,  # layer2.1.conv1
    0.33785664545675764, # layer2.1.conv2
    0.3621800259998386,  # layer3.0.conv1
    0.41251586852550626, # layer3.0.conv2
    0.3094889322916667,  # layer3.0.downsample.0
    0.47138968208449467, # layer3.1.conv1
    0.49073431274998425, # layer3.1.conv2
    0.5321279337376724,  # layer4.0.conv1
    0.5624714839605638,  # layer4.0.conv2
    0.3031751844618056,  # layer4.0.downsample.0
    0.6264706364384404,  # layer4.1.conv1
    0.7181391774872203   # layer4.1.conv2
]

import numpy as np

print(np.mean(speedups))