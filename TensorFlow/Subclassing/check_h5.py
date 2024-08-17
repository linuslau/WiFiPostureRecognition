import h5py
import sys
import os

def print_structure(name, obj):
    """Recursively print the hierarchy of the HDF5 file."""
    indent_level = name.count('/')
    indent = '    ' * indent_level
    obj_type = "Group" if isinstance(obj, h5py.Group) else "Dataset"
    print(f"{indent}{name} ({obj_type})")
    
    # Print attributes
    for key, val in obj.attrs.items():
        print(f"{indent}    Attribute: {key} = {val}")

def explore_hdf5(file_path):
    print(f"\nExploring file: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        print(f"File structure for {file_path}:")
        f.visititems(print_structure)

def main():
    # Check if a file argument was passed
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.isfile(file_path) and file_path.endswith('.h5'):
            explore_hdf5(file_path)
        else:
            print(f"File '{file_path}' does not exist or is not an HDF5 file.")
    else:
        # If no argument is provided, parse all .h5 files in the current directory
        current_dir = os.getcwd()
        h5_files = [f for f in os.listdir(current_dir) if f.endswith('.h5')]

        if not h5_files:
            print("No .h5 files found in the current directory.")
        else:
            for h5_file in h5_files:
                explore_hdf5(h5_file)

if __name__ == "__main__":
    main()
