import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_train_QR(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Parse the Time column as datetime
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Set Time as index for easier plotting
    df.set_index('Time', inplace=True)
    
    # Calculate 80% split point BEFORE clearing NaN values
    split_index = int(len(df) * 0.8)
    
    # Take only the first 80% of data (training set)
    df_train = df.iloc[:split_index]

    # Replace -999.0 with NaN and drop those rows from training set
    df_train = df_train.replace(-999.0, pd.NA)
    df_train = df_train.dropna(subset=['QR (CFS)'])
    
    # Define the columns to plot
    columns_to_plot = ['QR (CFS)']
    
    # Create subplots - one for each column
    fig, axes = plt.subplots(len(columns_to_plot), 1, figsize=(12, 8 * len(columns_to_plot)))
    
    # Handle case where there's only one subplot
    if len(columns_to_plot) == 1:
        axes = [axes]
    
    for i, column in enumerate(columns_to_plot):
        # Get start and end dates from the training data
        start_date = df_train.index.min()
        end_date = df_train.index.max()
        start_year = start_date.year
        end_year = end_date.year
        
        # Print exact start and end dates
        print(f"Training data period: {start_date.strftime('%B %d, %Y at %H:%M')} to {end_date.strftime('%B %d, %Y at %H:%M')}")
        
        axes[i].plot(df_train.index, df_train[column], linewidth=1)
        axes[i].set_title(f'Streamflow -- training data ({start_year} to {end_year})', fontsize=22, fontweight='bold') 
        axes[i].set_xlabel('Time (Years)', fontsize=22)
        axes[i].set_ylabel("Streamflow (CFS)", fontsize=22)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='both', which='major', labelsize=20)
        
        # Format x-axis to show years nicely
        axes[i].tick_params(axis='x', rotation=45)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig("3_train_QR.png")

def plot_csv_test_QR(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Parse the Time column as datetime
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Set Time as index for easier plotting
    df.set_index('Time', inplace=True)
    
    # Calculate 80% split point BEFORE clearing NaN values
    split_index = int(len(df) * 0.9)
    
    # Take only the first 80% of data (training set)
    df_train = df.iloc[split_index:]

    # Replace -999.0 with NaN and drop those rows from training set
    df_train = df_train.replace(-999.0, pd.NA)
    df_train = df_train.dropna(subset=['QR (CFS)'])
    
    # Define the columns to plot
    columns_to_plot = ['QR (CFS)']
    
    # Create subplots - one for each column
    fig, axes = plt.subplots(len(columns_to_plot), 1, figsize=(12, 8 * len(columns_to_plot)))
    
    # Handle case where there's only one subplot
    if len(columns_to_plot) == 1:
        axes = [axes]
    
    for i, column in enumerate(columns_to_plot):
        # Get start and end dates from the training data
        start_date = df_train.index.min()
        end_date = df_train.index.max()
        start_year = start_date.year
        end_year = end_date.year
        
        # Print exact start and end dates
        print(f"Training data period: {start_date.strftime('%B %d, %Y at %H:%M')} to {end_date.strftime('%B %d, %Y at %H:%M')}")
        
        axes[i].plot(df_train.index, df_train[column], linewidth=1)
        axes[i].set_title(f'Streamflow -- testing data ({start_year} to {end_year})', fontsize=22, fontweight='bold') 
        axes[i].set_xlabel('Time (Years)', fontsize=22)
        axes[i].set_ylabel("Streamflow (CFS)", fontsize=22)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='both', which='major', labelsize=20)
        
        # Format x-axis to show years nicely
        axes[i].tick_params(axis='x', rotation=45)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig("3_test_QR.png")

def plot_csv_QR_MAP(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    df = df[:24*30*3]
    # Parse the Time column as datetime
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Set Time as index for easier plotting
    df.set_index('Time', inplace=True)
    
    df_train = df

    # Replace -999.0 with NaN and drop those rows from training set
    df_train = df_train.replace(-999.0, pd.NA)
    df_train = df_train.dropna(subset=['HG (FT)', 'QR (CFS)', 'MAP (IN)'])
    
    # Define the columns to plot
    columns_to_plot = ['HG (FT)', 'QR (CFS)', 'MAP (IN)']
    
    # Create subplots - one for each column
    fig, axes = plt.subplots(len(columns_to_plot), 1, figsize=(12, 4 * len(columns_to_plot)))
    
    # Handle case where there's only one subplot
    if len(columns_to_plot) == 1:
        axes = [axes]
    
    for i, column in enumerate(columns_to_plot):
        # Get start and end dates from the training data
        start_date = df_train.index.min()
        end_date = df_train.index.max()
        start_year = start_date.year
        end_year = end_date.year
        
        # Print exact start and end dates
        print(f"Training data period: {start_date.strftime('%B %d, %Y at %H:%M')} to {end_date.strftime('%B %d, %Y at %H:%M')}")
        
        axes[i].plot(df_train.index, df_train[column], linewidth=2)

        if column == 'QR (CFS)':
            # axes[i].set_title(f'Streamflow -- data ({start_year} to {end_year})', fontsize=22, fontweight='bold') 
            axes[i].set_ylabel("Streamflow  R (CFS)", fontsize=22)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='both', which='major', labelsize=20)
            # Remove x-axis labels for QR subplot
            axes[i].tick_params(axis='x', labelbottom=False)
        elif column == 'MAP (IN)':
            # axes[i].set_title(f'Mean Annual Precipitation -- data ({start_year} to {end_year})', fontsize=22, fontweight='bold') 
            axes[i].set_xlabel('Time (year-month)', fontsize=22)
            axes[i].set_ylabel("Precipitation (IN)", fontsize=22)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='both', which='major', labelsize=18)
            import matplotlib.dates as mdates
            axes[i].xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Every 3 months
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Year-Month format
            axes[i].tick_params(axis='x', rotation=45)
        elif column == 'HG (FT)':
            # axes[i].set_title(f'Height Above Ground -- data ({start_year} to {end_year})', fontsize=22, fontweight='bold') 
            axes[i].set_ylabel("Water Level (FT)", fontsize=22)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='both', which='major', labelsize=20)
            # Remove x-axis labels for HG subplot
            axes[i].tick_params(axis='x', labelbottom=False)

    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig("4_QR_MAP.png")

# def plot_csv_columns(csv_file_path):
#     """
#     Read CSV file and plot HG (FT), QR (CFS), and MAP (IN) as line plots
    
#     Parameters:
#     csv_file_path (str): Path to the CSV file
#     """
    
#     # Read the CSV file
#     df = pd.read_csv(csv_file_path)
    
#     # Define the columns to plot
#     columns_to_plot = ['HG (FT)', 'QR (CFS)', 'MAP (IN)']
    
#     # Check if all required columns exist
#     missing_columns = [col for col in columns_to_plot if col not in df.columns]
#     if missing_columns:
#         print(f"Warning: The following columns are missing from the CSV: {missing_columns}")
#         columns_to_plot = [col for col in columns_to_plot if col in df.columns]
    
#     if not columns_to_plot:
#         print("No matching columns found in the CSV file.")
#         return
    
#     # Create subplots - one for each column
#     fig, axes = plt.subplots(len(columns_to_plot), 1, figsize=(12, 4 * len(columns_to_plot)))
    
#     # Handle case where there's only one subplot
#     if len(columns_to_plot) == 1:
#         axes = [axes]
#     df = df
#     # Plot each column
#     for i, column in enumerate(columns_to_plot):
#         axes[i].plot(df.index, df[column], linewidth=1)
#         axes[i].set_title(f'{column} Over Time', fontsize=14, fontweight='bold')
#         axes[i].set_xlabel('Data Point Index')
#         axes[i].set_ylabel(column)
#         axes[i].grid(True, alpha=0.3)
#         axes[i].tick_params(axis='both', which='major', labelsize=10)
    
#     # Adjust layout to prevent overlap
#     plt.tight_layout()
#     plt.savefig("4_QR_MAP.png")

if __name__ == "__main__":
    # Replace 'your_file.csv' with the actual path to your CSV file
    csv_file_path = 'datasets/SMFV2_Data_withbasin.csv'
    plot_csv_train_QR(csv_file_path)
    plot_csv_test_QR(csv_file_path)

    plot_csv_QR_MAP(csv_file_path)
    
    
    # Optional: Display basic statistics
    df = pd.read_csv(csv_file_path)
    columns_of_interest = ['HG (FT)', 'QR (CFS)', 'MAP (IN)']
    existing_columns = [col for col in columns_of_interest if col in df.columns]
    
    if existing_columns:
        print("\nBasic Statistics:")
        print(df[existing_columns].describe())



# def plot_csv_columns_combined(csv_file_path):
#     """
#     Alternative version: Plot all three columns on the same graph with separate y-axes
#     """
    
#     # Read the CSV file
#     df = pd.read_csv(csv_file_path)
    
#     # Define the columns to plot
#     columns_to_plot = ['HG (FT)', 'QR (CFS)', 'MAP (IN)']
    
#     # Check if all required columns exist
#     missing_columns = [col for col in columns_to_plot if col not in df.columns]
#     if missing_columns:
#         print(f"Warning: The following columns are missing from the CSV: {missing_columns}")
#         columns_to_plot = [col for col in columns_to_plot if col in df.columns]
    
#     if not columns_to_plot:
#         print("No matching columns found in the CSV file.")
#         return
    
#     # Create figure with multiple y-axes
#     fig, ax1 = plt.subplots(figsize=(12, 8))
    
#     colors = ['blue', 'red', 'green']
    
#     # Plot first column on primary y-axis
#     if 'HG (FT)' in columns_to_plot:
#         ax1.plot(df.index, df['HG (FT)'], color=colors[0], linewidth=2, marker='o', markersize=3, label='HG (FT)')
#         ax1.set_xlabel('Data Point Index')
#         ax1.set_ylabel('HG (FT)', color=colors[0])
#         ax1.tick_params(axis='y', labelcolor=colors[0])
    
#     # Create second y-axis for QR (CFS)
#     if 'QR (CFS)' in columns_to_plot:
#         ax2 = ax1.twinx()
#         ax2.plot(df.index, df['QR (CFS)'], color=colors[1], linewidth=2, marker='s', markersize=3, label='QR (CFS)')
#         ax2.set_ylabel('QR (CFS)', color=colors[1])
#         ax2.tick_params(axis='y', labelcolor=colors[1])
    
#     # Create third y-axis for MAP (IN)
#     if 'MAP (IN)' in columns_to_plot:
#         ax3 = ax1.twinx()
#         # Offset the third y-axis
#         ax3.spines['right'].set_position(('outward', 60))
#         ax3.plot(df.index, df['MAP (IN)'], color=colors[2], linewidth=2, marker='^', markersize=3, label='MAP (IN)')
#         ax3.set_ylabel('MAP (IN)', color=colors[2])
#         ax3.tick_params(axis='y', labelcolor=colors[2])
    
#     plt.title('HG (FT), QR (CFS), and MAP (IN) Over Time', fontsize=16, fontweight='bold')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()