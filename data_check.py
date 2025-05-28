import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_columns(csv_file_path):
    """
    Read CSV file and plot HG (FT), QR (CFS), and MAP (IN) as line plots
    
    Parameters:
    csv_file_path (str): Path to the CSV file
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Define the columns to plot
    columns_to_plot = ['HG (FT)', 'QR (CFS)', 'MAP (IN)']
    
    # Check if all required columns exist
    missing_columns = [col for col in columns_to_plot if col not in df.columns]
    if missing_columns:
        print(f"Warning: The following columns are missing from the CSV: {missing_columns}")
        columns_to_plot = [col for col in columns_to_plot if col in df.columns]
    
    if not columns_to_plot:
        print("No matching columns found in the CSV file.")
        return
    
    # Create subplots - one for each column
    fig, axes = plt.subplots(len(columns_to_plot), 1, figsize=(12, 4 * len(columns_to_plot)))
    
    # Handle case where there's only one subplot
    if len(columns_to_plot) == 1:
        axes = [axes]
    
    # Plot each column
    for i, column in enumerate(columns_to_plot):
        axes[i].plot(df.index, df[column], linewidth=2, marker='o', markersize=3)
        axes[i].set_title(f'{column} Over Time', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Data Point Index')
        axes[i].set_ylabel(column)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='both', which='major', labelsize=10)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def plot_csv_columns_combined(csv_file_path):
    """
    Alternative version: Plot all three columns on the same graph with separate y-axes
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Define the columns to plot
    columns_to_plot = ['HG (FT)', 'QR (CFS)', 'MAP (IN)']
    
    # Check if all required columns exist
    missing_columns = [col for col in columns_to_plot if col not in df.columns]
    if missing_columns:
        print(f"Warning: The following columns are missing from the CSV: {missing_columns}")
        columns_to_plot = [col for col in columns_to_plot if col in df.columns]
    
    if not columns_to_plot:
        print("No matching columns found in the CSV file.")
        return
    
    # Create figure with multiple y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green']
    
    # Plot first column on primary y-axis
    if 'HG (FT)' in columns_to_plot:
        ax1.plot(df.index, df['HG (FT)'], color=colors[0], linewidth=2, marker='o', markersize=3, label='HG (FT)')
        ax1.set_xlabel('Data Point Index')
        ax1.set_ylabel('HG (FT)', color=colors[0])
        ax1.tick_params(axis='y', labelcolor=colors[0])
    
    # Create second y-axis for QR (CFS)
    if 'QR (CFS)' in columns_to_plot:
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['QR (CFS)'], color=colors[1], linewidth=2, marker='s', markersize=3, label='QR (CFS)')
        ax2.set_ylabel('QR (CFS)', color=colors[1])
        ax2.tick_params(axis='y', labelcolor=colors[1])
    
    # Create third y-axis for MAP (IN)
    if 'MAP (IN)' in columns_to_plot:
        ax3 = ax1.twinx()
        # Offset the third y-axis
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(df.index, df['MAP (IN)'], color=colors[2], linewidth=2, marker='^', markersize=3, label='MAP (IN)')
        ax3.set_ylabel('MAP (IN)', color=colors[2])
        ax3.tick_params(axis='y', labelcolor=colors[2])
    
    plt.title('HG (FT), QR (CFS), and MAP (IN) Over Time', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Replace 'your_file.csv' with the actual path to your CSV file
    csv_file_path = 'datasets/SMFV2_Data_withbasin.csv'
    
    print("Plotting each column separately:")
    plot_csv_columns(csv_file_path)
    
    print("\nPlotting all columns on the same graph:")
    plot_csv_columns_combined(csv_file_path)
    
    # Optional: Display basic statistics
    df = pd.read_csv(csv_file_path)
    columns_of_interest = ['HG (FT)', 'QR (CFS)', 'MAP (IN)']
    existing_columns = [col for col in columns_of_interest if col in df.columns]
    
    if existing_columns:
        print("\nBasic Statistics:")
        print(df[existing_columns].describe())