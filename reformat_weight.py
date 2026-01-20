import os
from pathlib import Path

def reformat_all_txt_in_folder(folder_path: str):
    """
    Finds all .txt files in a given folder and reformats them so that
    each space-separated value is on its own line in a new output file.

    Args:
        folder_path (str): The path to the folder containing the .txt files.
    """
    # Use pathlib for robust path handling
    target_folder = Path(folder_path)

    # Check if the provided path is a valid directory
    if not target_folder.is_dir():
        print(f"❌ Error: Folder not found at '{folder_path}'")
        return

    # Find all files ending with .txt in the target folder
    txt_files = list(target_folder.glob('*.txt'))

    if not txt_files:
        print(f"ℹ️ No .txt files were found in '{folder_path}'.")
        return

    print(f"📂 Found {len(txt_files)} text file(s). Starting the process...\n")

    # Loop through each found text file
    for input_path in txt_files:
        # Avoid reprocessing files that have already been reformatted by this script
        if '_v' in input_path.stem:
            print(f"Skipping already processed file: {input_path.name}")
            continue

        # Create a new name for the output file
        # e.g., 'conv1.weight.txt' -> 'conv1.weight_reformatted.txt'
        output_path = input_path.with_stem(f"{input_path.stem}_v")

        try:
            print(f"Processing '{input_path.name}'...")
            with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
                for line in infile:
                    # Split the line by spaces to get individual values
                    values = line.strip().split()
                    
                    # Write each value on a new line in the output file
                    for value in values:
                        outfile.write(f"{value}\n")
            
            print(f"✅ Success! Saved to '{output_path.name}'\n")

        except Exception as e:
            print(f"❌ Failed to process {input_path.name}: {e}\n")

    print("✨ All done!")

# --- HOW TO USE ---
if __name__ == '__main__':
    # 1. IMPORTANT: Replace this placeholder with the actual path to your folder.
    #    Examples:
    #    - Windows: 'C:/Users/YourUser/Desktop/MyWeights'
    #    - macOS/Linux: '/home/youruser/documents/my_weights'
    #    - To process the current folder where the script is located, use '.'
    path_to_your_folder = './resnet18_prunned_weights50' 

    # 2. Run the script. It will handle the rest.
    reformat_all_txt_in_folder(path_to_your_folder)