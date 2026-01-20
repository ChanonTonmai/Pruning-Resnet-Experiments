import torch
import torchvision.models as models
from torchsummary import summary
import numpy as np
import torch.nn.utils.prune as prune
import os
import copy

def extract_resnet18_weights():
    """
    Loads a pre-trained ResNet-18 model, extracts all its parameters (weights and biases),
    and saves each parameter tensor into a separate .txt file.
    """
    # 1. Define the output directory
    output_dir = "resnet18_weights"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Weights will be saved in the '{output_dir}/' directory.")

    # 2. Load the pre-trained ResNet-18 model
    # Using the recommended 'weights' argument for modern torchvision
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval() # Set the model to evaluation mode

    # 3. Iterate through all named parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Processing layer: {name} with shape: {list(param.shape)}")

            # 4. Prepare the tensor for saving
            # Detach the tensor from the computation graph and move it to the CPU
            param_data = param.detach().cpu().numpy()

            # 5. Reshape the tensor to 2D for saving with np.savetxt
            # np.savetxt can only handle 1D or 2D arrays.
            # We reshape multi-dimensional arrays (like conv kernels) into a 2D matrix.
            if param_data.ndim > 2:
                # Flatten all dimensions except the first one
                num_features = np.prod(param_data.shape[1:])
                param_data_2d = param_data.reshape(param_data.shape[0], num_features)
            elif param_data.ndim == 1:
                # Reshape 1D bias vectors into a 2D array with one row
                param_data_2d = param_data.reshape(1, -1)
            else:
                param_data_2d = param_data

            # 6. Define the filename and save the array
            filename = os.path.join(output_dir, f"{name}.txt")
            np.savetxt(filename, param_data_2d, fmt='%.18e', delimiter=' ')

    print("\nExtraction complete! All weight and bias files have been saved.")


def extract_resnet18_prunned_weights():
    """
    Loads a pre-trained ResNet-18 model, extracts all its parameters (weights and biases),
    and saves each parameter tensor into a separate .txt file.
    """
    # 1. Define the output directory
    output_dir = "resnet18_prunned_weights70"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Weights will be saved in the '{output_dir}/' directory.")

    # 2. Load the pre-trained ResNet-18 model
    # Using the recommended 'weights' argument for modern torchvision
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval() # Set the model to evaluation mode
    
    model_to_prune = copy.deepcopy(model)
    
    parameters_to_prune = []
    for module in model_to_prune.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply global unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.7,  # Prune 50% of the weights
    )

    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    print("Pruning complete. The model is now sparse but still FP32.")
    model_pruned_fp32 = model_to_prune

    # 3. Iterate through all named parameters
    for name, param in model_pruned_fp32.named_parameters():
        if param.requires_grad:
            print(f"Processing layer: {name} with shape: {list(param.shape)}")

            # 4. Prepare the tensor for saving
            # Detach the tensor from the computation graph and move it to the CPU
            param_data = param.detach().cpu().numpy()

            # 5. Reshape the tensor to 2D for saving with np.savetxt
            # np.savetxt can only handle 1D or 2D arrays.
            # We reshape multi-dimensional arrays (like conv kernels) into a 2D matrix.
            if param_data.ndim > 2:
                # Flatten all dimensions except the first one
                num_features = np.prod(param_data.shape[1:])
                param_data_2d = param_data.reshape(param_data.shape[0], num_features)
            elif param_data.ndim == 1:
                # Reshape 1D bias vectors into a 2D array with one row
                param_data_2d = param_data.reshape(1, -1)
            else:
                param_data_2d = param_data

            # 6. Define the filename and save the array
            filename = os.path.join(output_dir, f"{name}.txt")
            np.savetxt(filename, param_data_2d, fmt='%.18e', delimiter=' ')

    print("\nExtraction complete! All weight and bias files have been saved.")


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
    
    
    # Manual calculation of parameters
    total_params = sum(p.numel() for p in resnet18.parameters())
    trainable_params = sum(p.numel() for p in resnet18.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Calculate the model size in megabytes (MB)
    # Each parameter is a 32-bit float (4 bytes)
    model_size_bytes = total_params * 4
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    print(f"Estimated model size: {model_size_mb:.2f} MB")


def read_weights(filepath):
    """
    Reads the flattened weights from the text file and reshapes them
    back to the original 4D kernel shape (64, 3, 7, 7).
    """
    print(f"Reading weights from '{filepath}'...")
    with open(filepath, 'r') as f:
        # Read the entire file content
        content = f.read()

    # Split the content by any whitespace (handles spaces, newlines, tabs)
    # This is more robust than splitting by just spaces.
    str_weights = content.split()

    # Convert the string numbers to floats
    float_weights = [float(w) for w in str_weights]

    # Convert the list to a NumPy array and reshape it
    # ResNet-18's first layer has 64 filters, each processing 3 input channels, with a 7x7 kernel
    kernels = np.array(float_weights).reshape(64, 3, 7, 7)
    print(f"Weights loaded and reshaped to: {kernels.shape}\n")
    return kernels

if __name__ == '__main__':
    # extract_resnet18_weights()
    extract_resnet18_prunned_weights()
    print_model()