import pandas as pd
import os

def add_basin_column(input_file, output_file, basin_value='1'):
    """
    Read a CSV file, add a 'basin' column with the specified value, and save to a new file.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the modified data
        basin_value: Value to set for all rows in the 'basin' column (default '1')
    """
    try:
        # Read the CSV file
        # Note: We're assuming the first row is the header
        df = pd.read_csv(input_file)
        
        # Add a 'basin' column with all values set to the specified value
        df['basin'] = basin_value
        
        # Save the modified dataframe to a new CSV file
        df.to_csv(output_file, index=False)
        
        print(f"Successfully processed {input_file}")
        print(f"Modified data saved to {output_file}")
        print(f"Added 'basin' column with value '{basin_value}' to all rows")
        
    except Exception as e:
        print(f"Error processing file: {e}")

# File paths - modify these to match your actual file paths
input_file = "datasets/SMFV2_Data.csv"  # Change this to your input file path
output_file = "datasets/SMFV2_Data_withbasin.csv"  # Change this to your desired output file path

# Run the function
add_basin_column(input_file, output_file)