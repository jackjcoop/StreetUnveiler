import numpy as np
import os
from pathlib import Path
import sys

def analyze_npy_file(filepath):
    """
    Analyze a single .npy file and return metadata and sample data.
    """
    try:
        # Load the array
        data = np.load(filepath, allow_pickle=True)
        
        metadata = {}
        
        # Basic information
        metadata['File'] = os.path.basename(filepath)
        metadata['Full Path'] = str(filepath)
        metadata['File Size'] = f"{os.path.getsize(filepath) / 1024:.2f} KB"
        
        # Array information
        metadata['Data Type'] = str(data.dtype)
        metadata['Shape'] = str(data.shape)
        metadata['Number of Dimensions'] = data.ndim
        metadata['Total Elements'] = data.size
        metadata['Memory Usage'] = f"{data.nbytes / 1024:.2f} KB"
        
        # Check if it's a scalar, structured array, or object array
        if data.dtype == object:
            metadata['Array Type'] = 'Object Array'
            metadata['Content Type'] = str(type(data.item()) if data.size == 1 else 'Multiple Objects')
        elif data.dtype.names is not None:
            metadata['Array Type'] = 'Structured Array'
            metadata['Field Names'] = str(data.dtype.names)
        else:
            metadata['Array Type'] = 'Regular NumPy Array'
        
        # Statistical information (only for numeric arrays)
        if np.issubdtype(data.dtype, np.number) and data.size > 0:
            metadata['Min Value'] = np.min(data)
            metadata['Max Value'] = np.max(data)
            metadata['Mean Value'] = np.mean(data)
            metadata['Std Deviation'] = np.std(data)
            metadata['Median Value'] = np.median(data)
            
            # Check for NaN or Inf values
            if np.issubdtype(data.dtype, np.floating):
                metadata['Contains NaN'] = np.any(np.isnan(data))
                metadata['Contains Inf'] = np.any(np.isinf(data))
        
        # Sample data
        sample_data = get_sample_data(data)
        
        return metadata, sample_data
        
    except Exception as e:
        return {'Error': str(e)}, None

def get_sample_data(data, max_elements=100):
    """
    Get a sample of the data for display.
    """
    sample = {}
    
    if data.size == 0:
        sample['Note'] = 'Empty array'
        return sample
    
    # For small arrays, show everything
    if data.size <= max_elements:
        sample['Full Data'] = str(data)
    else:
        # For larger arrays, show samples
        if data.ndim == 1:
            # 1D array: show first and last elements
            sample['First 10 elements'] = str(data[:min(10, len(data))])
            sample['Last 10 elements'] = str(data[-min(10, len(data)):])
        elif data.ndim == 2:
            # 2D array: show corners
            rows, cols = data.shape
            sample['Top-left corner (5x5)'] = str(data[:min(5, rows), :min(5, cols)])
            if rows > 5 and cols > 5:
                sample['Bottom-right corner (5x5)'] = str(data[-min(5, rows):, -min(5, cols):])
        elif data.ndim == 3:
            # 3D array: show first and last slices
            sample['First slice shape'] = str(data[0].shape)
            sample['First slice sample'] = str(data[0, :min(3, data.shape[1]), :min(3, data.shape[2])])
            if data.shape[0] > 1:
                sample['Last slice sample'] = str(data[-1, :min(3, data.shape[1]), :min(3, data.shape[2])])
        else:
            # Higher dimensional arrays
            sample['First element'] = str(data.flat[0])
            sample['Shape info'] = f"Array with {data.ndim} dimensions"
    
    return sample

def save_metadata_to_txt(filepath, metadata, sample_data):
    """
    Save metadata and sample data to a text file.
    """
    txt_filename = filepath.stem + '_metadata.txt'
    txt_path = filepath.parent / txt_filename
    
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"METADATA ANALYSIS FOR: {filepath.name}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("FILE METADATA:\n")
        f.write("-" * 40 + "\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
        
        if sample_data:
            f.write("\n" + "=" * 80 + "\n")
            f.write("SAMPLE DATA:\n")
            f.write("-" * 40 + "\n")
            for key, value in sample_data.items():
                f.write(f"\n{key}:\n{value}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Analysis completed at: {np.datetime64('now')}\n")
    
    return txt_path

def analyze_folder(folder_path):
    """
    Analyze all .npy files in a folder.
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return
    
    # List of specific files to analyze
    target_files = [
        'depth_conf.npy',
        'depth_map.npy',
        'extrinsic.npy',
        'intrinsic.npy',
        'point_conf.npy',
        'point_map.npy',
        'points3d_unproj.npy'
    ]
    
    print(f"\nAnalyzing .npy files in: {folder.absolute()}")
    print("=" * 80)
    
    files_analyzed = 0
    
    for filename in target_files:
        filepath = folder / filename
        
        if filepath.exists():
            print(f"\nAnalyzing: {filename}")
            print("-" * 40)
            
            # Analyze the file
            metadata, sample_data = analyze_npy_file(filepath)
            
            if 'Error' in metadata:
                print(f"  Error: {metadata['Error']}")
            else:
                # Display key information
                print(f"  Shape: {metadata['Shape']}")
                print(f"  Data Type: {metadata['Data Type']}")
                print(f"  File Size: {metadata['File Size']}")
                
                # Save to text file
                txt_path = save_metadata_to_txt(filepath, metadata, sample_data)
                print(f"  Metadata saved to: {txt_path.name}")
                
                files_analyzed += 1
        else:
            print(f"\nFile not found: {filename}")
    
    print("\n" + "=" * 80)
    print(f"Analysis complete. {files_analyzed} files analyzed.")
    print(f"Metadata text files saved in: {folder.absolute()}")

def main():
    """
    Main function to run the script.
    """
    # You can modify this path to your folder location
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        # Default path - modify this to your folder path
        folder_path = input("Enter the path to the folder containing .npy files: ").strip()
    
    analyze_folder(folder_path)

if __name__ == "__main__":
    main()