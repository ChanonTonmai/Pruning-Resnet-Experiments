import torch
import torchvision.models as models
from torchsummary import summary
import numpy as np
import torch.nn.utils.prune as prune
import os
import copy
import time



resnet_filter_size = {
    1:  [64, 3, 7, 7],    # conv1.weight 
    2:  [64, 64, 3, 3],   # layer1.0.conv1.weight  
    3:  [64, 64, 3, 3],   # layer1.0.conv2.weight 
    4:  [64, 64, 3, 3],   # layer1.1.conv1.weight   
    5:  [64, 64, 3, 3],   # layer1.1.conv2.weight
    6:  [128, 64, 3, 3],  # layer2.0.conv1.weight
    7:  [128, 128, 3, 3], # layer2.0.conv2.weight
    8:  [128, 64, 1, 1],  # layer2.0.downsample.0.weight   
    9:  [128, 128, 3, 3], # layer2.1.conv1.weight  
    10: [128, 128, 3, 3], # layer2.1.conv2.weight 
    11: [256, 128, 3, 3], # layer3.0.conv1.weight
    12: [256, 256, 3, 3], # layer3.0.conv2.weight 
    13: [256, 128, 1, 1], # layer3.0.downsample.0.weight 
    14: [256, 256, 3, 3], # layer3.1.conv1.weight 
    15: [256, 256, 3, 3], # layer3.1.conv2.weight 
    16: [512, 256, 3, 3], # layer4.0.conv1.weight   
    17: [512, 512, 3, 3], # layer4.0.conv2.weight 
    18: [512, 256, 1, 1], # layer4.0.downsample.0.weight
    19: [512, 512, 3, 3], # layer4.1.conv1.weight
    20: [512, 512, 3, 3]  # layer4.1.conv2.weight 
}

def convolve_2d_manual(image, kernels, stride, padding, conv_sel):
    """
    Performs 2D convolution on an image using only for loops.

    Args:
        image (np.array): Input image of shape (in_channels, height, width).
        kernels (np.array): Filters of shape (out_channels, in_channels, kernel_h, kernel_w).
        stride (int): The stride of the convolution.
        padding (int): The zero-padding to apply.
    """
    # Get dimensions
    num_kernels, in_channels, kernel_h, kernel_w = kernels.shape
    _, in_h, in_w = image.shape

    # Apply padding to the input image
    padded_image = pad_with_zeros(image, padding, padding)

    # Calculate output dimensions
    out_h = (in_h - kernel_h + 2 * padding) // stride + 1
    out_w = (in_w - kernel_w + 2 * padding) // stride + 1

    # Initialize the output feature map with zeros
    output = np.zeros((num_kernels, out_h, out_w))

    
    ic_list = list() # instruction cycle counts
    ic = 0 
    
    sparsity = list()
    
    # print(f"num_kernels = {num_kernels}")
    # print(f"out_h = {out_h}")
    # print(f"out_w = {out_w}")
    # print(f'in_channels = {in_channels}')
    # print(f"kernel_size = {kernel_h}x{kernel_w}")
    # print(f"stride = {stride}" )
    # --- THE CORE CONVOLUTION LOGIC ---
    # Loop over each filter/kernel
    for f in range(num_kernels):
        ic = 2
        
        # count = 0 
        # for ii in range(len(kernels[f, :, :, :])):
        #     for jj in range(len(kernels[f, ii, :, :])):
        #         for kk in range(len(kernels[f, ii, jj, :])):
        #             if kernels[f, ii, jj, kk] == 0:
        #                 count += 1
        # sparsity.append(count)
        
        # Loop over the vertical dimension of the output feature map
        for y in range(out_h):
            ic = ic + 2
            # Loop over the horizontal dimension of the output feature map
            for x in range(out_w):
                ic = ic + 2
                # Calculate the top-left corner of the current receptive field on the padded image
                h_start = y * stride
                w_start = x * stride

                convolution_sum = 0.0  # Initialize accumulator for the dot product

                # Now, loop through the channels, height, and width of the kernel
                # to perform element-wise multiplication and summation manually.
                for c in range(in_channels):
                    ic = ic + 2
                    for kh in range(kernel_h):
                        ic = ic + 2
                        # if (x == 0 and y == 0):
                        #     print(f"--------------------")
                        for kw in range(kernel_w):
                            # Get the value from the image patch
                            image_val = padded_image[c, h_start + kh, w_start + kw]
                            # Get the value from the kernel
                            kernel_val = kernels[f, c, kh, kw]
                            # Multiply and accumulate
                            convolution_sum += image_val * kernel_val

                            # counting instructions
                            if (conv_sel == 0): 
                                ic = ic + 10
                            elif (conv_sel == 1): 
                                # if (x == 0 and y == 0):
                                #     print(f"{image_val} x {kernel_val}={convolution_sum}")
                                    
                                if (image_val != 0 and kernel_val != 0):
                                    ic = ic + 10
                                    # if (x == 0 and y == 0):
                                    #     print(f"meet image non zero {kh}, {kw}")
                                else:
                                    if (kernel_val == 0): 
                                        ic = ic + 2
                                        # if (x == 0 and y == 0):
                                        #     print(f"meet ic zero {kh}, {kw}")
                                    elif (image_val == 0): 
                                        ic = ic + 1
                                        # if (x == 0 and y == 0):
                                        #     print(f"meet image val zero {kh}, {kw}")
                                    else: 
                                        ic = ic + 10
                                    # print(f"instruction count = {ic}")
                        # ic=ic+3
            

                # Store the final sum in the corresponding location of the output feature map
                output[f, y, x] = convolution_sum
                
        ic_list.append(ic)
                
                
                
            
    return output, ic, ic_list, sparsity

def read_weights(filepath, shape):
    """
    Reads flattened weights from a text file and reshapes them
    to the specified 4D kernel shape.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    str_weights = content.split()
    float_weights = [float(w) for w in str_weights]
    kernels = np.array(float_weights).reshape(shape)
    return kernels

def print_model():
    resnet18 = models.resnet18(weights='IMAGENET1K_V1')
    print("--- ResNet-18 Architecture ---")
    print(resnet18)
    
    # --- Calculate and Print the Size ---
    print("\n--- ResNet-18 Size ---")
    
    # Use torchsummary for a detailed breakdown (optional but recommended)
    # The input size (3, 224, 224) is a standard ImageNet image size (3 channels, 224x224 pixels)
    print("Using torchsummary:")
    summary(resnet18, (3, 224, 224))
    
    # --- Detailed Parameter Size Breakdown ---
    print("\n--- Detailed Parameter Size per Layer ---")
    print(f"{'Layer Name':<45} {'Shape':<25} {'Param #':<15} {'Size (KB)':<10}")
    print("-" * 100)
    total_size_kb = 0
    for name, param in resnet18.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            # Each parameter is float32 (4 bytes)
            param_size_kb = num_params * 4 / 1024
            total_size_kb += param_size_kb
            print(f"{name:<45} {str(list(param.shape)):<25} {num_params:<15,} {param_size_kb:<10.2f}")
    print("-" * 100)
    total_params = sum(p.numel() for p in resnet18.parameters())
    print(f"{'Total':<45} {'':<25} {total_params:<15,} {total_size_kb:<10.2f}\n")
    
    # Manual calculation of parameters
    print(f"Total parameters: {total_params:,}")
    trainable_params = sum(p.numel() for p in resnet18.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Calculate the model size in megabytes (MB)
    model_size_mb = total_size_kb / 1024
    print(f"Estimated model size: {model_size_mb:.2f} MB")

def pad_with_zeros(image, padding_h, padding_w):
    """
    Adds zero-padding to a 3D image (channels, height, width).
    """
    padded_height = image.shape[1] + 2 * padding_h
    padded_width = image.shape[2] + 2 * padding_w
    padded_image = np.zeros((image.shape[0], padded_height, padded_width))
    # Copy the original image into the center of the padded image
    padded_image[:, padding_h:padding_h + image.shape[1], padding_w:padding_w + image.shape[2]] = image
    return padded_image


if __name__ == '__main__':
    # print_model()
    resnet_layers = {
        'conv1':                         {'shape': [64, 3, 7, 7], 'stride': 1, 'padding': 3},
        'layer1.0.conv1':                {'shape': [64, 64, 3, 3], 'stride': 1, 'padding': 1},
        'layer1.0.conv2':                {'shape': [64, 64, 3, 3], 'stride': 1, 'padding': 1},
        'layer1.1.conv1':                {'shape': [64, 64, 3, 3], 'stride': 1, 'padding': 1},
        'layer1.1.conv2':                {'shape': [64, 64, 3, 3], 'stride': 1, 'padding': 1},
        'layer2.0.conv1':                {'shape': [128, 64, 3, 3], 'stride': 2, 'padding': 1},
        'layer2.0.conv2':                {'shape': [128, 128, 3, 3], 'stride': 1, 'padding': 1},
        'layer2.0.downsample.0':         {'shape': [128, 64, 1, 1], 'stride': 2, 'padding': 0},
        'layer2.1.conv1':                {'shape': [128, 128, 3, 3], 'stride': 1, 'padding': 1},
        'layer2.1.conv2':                {'shape': [128, 128, 3, 3], 'stride': 1, 'padding': 1},
        'layer3.0.conv1':                {'shape': [256, 128, 3, 3], 'stride': 2, 'padding': 1},
        'layer3.0.conv2':                {'shape': [256, 256, 3, 3], 'stride': 1, 'padding': 1},
        'layer3.0.downsample.0':         {'shape': [256, 128, 1, 1], 'stride': 2, 'padding': 0},
        'layer3.1.conv1':                {'shape': [256, 256, 3, 3], 'stride': 1, 'padding': 1},
        'layer3.1.conv2':                {'shape': [256, 256, 3, 3], 'stride': 1, 'padding': 1},
        'layer4.0.conv1':                {'shape': [512, 256, 3, 3], 'stride': 2, 'padding': 1},
        'layer4.0.conv2':                {'shape': [512, 512, 3, 3], 'stride': 1, 'padding': 1},
        'layer4.0.downsample.0':         {'shape': [512, 256, 1, 1], 'stride': 2, 'padding': 0},
        'layer4.1.conv1':                {'shape': [512, 512, 3, 3], 'stride': 1, 'padding': 1},
        'layer4.1.conv2':                {'shape': [512, 512, 3, 3], 'stride': 1, 'padding': 1}
    }
    
    input_image_numpy = np.random.rand(3, 32, 32).astype(np.float32)
    # print(f"Created a random input image of shape: {input_image_numpy.shape}\n")
    
    # current_feature_map = np.random.rand(3, 32, 32).astype(np.float32)
    # print(f"Created a random input image of shape: {current_feature_map.shape}\n")
    # current_feature_map_prunned = current_feature_map
    # kernels_numpy = read_weights("resnet18_weights/conv1.weight.txt")
    # kernels_prunned_numpy = read_weights("resnet18_prunned_weights/conv1.weight.txt")
    

    # input_image_numpy = read_weights(f"/home/khongpra/PhD_project/RISC-V-CGRA-FPGA/software/kernel/image_pad/input.txt", [3, 32, 32])
    current_feature_map = input_image_numpy
    current_feature_map_prunned = input_image_numpy
    i = 0;
    for name, config in resnet_layers.items():
        # print(f"--- Processing Layer: {name} ---")
        
        # Load the kernels for the current layer
        shape = config['shape']
        kernels_numpy = read_weights(f"resnet18_weights/{name}.weight.txt", shape)
        # prunned_kernels_numpy = read_weights(f"/home/khongpra/PhD_project/RISC-V-CGRA-FPGA/software/kernel/data/resnet18_prunned_weights50/{name}.weight_raw_fxp.txt", shape)
    
        prunned_kernels_numpy = read_weights(f"resnet18_prunned_weights50/{name}.weight.txt", shape)
        
        # Get stride and padding
        stride = config['stride']
        padding = config['padding']
        # --- Run our manual convolution ---
        # print(f"Input shape: {current_feature_map.shape}")
        manual_output, ic, ic_list, spar = convolve_2d_manual(current_feature_map, kernels_numpy, stride, padding, conv_sel=0)
        print(f"Manual convolution take: {ic} cycles")
        

        manual_output_p, ic_p, ic_list_p, spar_p = convolve_2d_manual(current_feature_map_prunned, 
                                                   prunned_kernels_numpy, stride, padding, conv_sel=1)
        print(f"Prunned convolution take: {ic_p} cycles")
        
        current_feature_map = manual_output
        current_feature_map_prunned = manual_output_p
        
        speedup = (np.abs(ic-ic_p)/ic)
        print(f"Speedup for layer {name}: {speedup}")
        # print("-----------------------------------------")
        i = i + 1
    
