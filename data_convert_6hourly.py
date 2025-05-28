import pandas as pd
from pathlib import Path
import os
INFILE  = "/home/qic69/projects/stream_forcasting/datasets/SMFV2_Data_withbasin.csv"
OUTFILE = "/home/qic69/projects/stream_forcasting/datasets/SMFV2_Data_withbasin_6hourly.csv"



def main() -> None:
    
    df = pd.read_csv(
        INFILE,
        parse_dates=["Time"],
        date_format="%m/%d/%y %H:%M",      # new in pandas ≥2.2
    )

    df = df.set_index("Time")

    numeric_cols = ["HG (FT)", "QR (CFS)", "MAP (IN)"]
    mask = (df[numeric_cols] == -999.0).any(axis=1)
    print(mask.sum(), "rows with -999.0, dropping them")
    df = df[~mask]
 
    six_hour_avg = (
        df[numeric_cols]
        .resample("6h", closed="left", label="left")
        .mean()
    )

    six_hour_avg["basin"] = (
        df["basin"]
        .resample("6h", closed="left", label="left")
        .first()
    )
    six_hour_avg = six_hour_avg.dropna(how="any")
    six_hour_avg.reset_index().to_csv(
        OUTFILE,
        index=False,
    )

    print(f"✓ 6-hour averages written to {os.path.abspath(OUTFILE)}")


if __name__ == "__main__":
    main()
