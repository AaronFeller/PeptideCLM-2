import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

def main():
    input_file = "ARVF_all_smiles_data_split_canonical_rdkit_descriptors.parquet"
    output_file = "ARVF_all_smiles_data_split_canonical_rdkit_descriptors_normalized.parquet"
    chunk_size = 100000

    print("First pass: calculating statistics...")
    pf = pq.ParquetFile(input_file)
    total_rows = pf.metadata.num_rows
    
    # Initialize running statistics
    sum_values = None
    sum_squares = None
    count = None
    
    # Get column names from first chunk
    first_chunk = next(pf.iter_batches(batch_size=1)).to_pandas()
    column_names = first_chunk.columns

    # First pass with detailed error checking
    for batch in tqdm(pf.iter_batches(batch_size=chunk_size)):
        df_chunk = batch.to_pandas()
        chunk_values = df_chunk.values
        
        # Print info about problematic values
        inf_mask = np.isinf(chunk_values)
        nan_mask = np.isnan(chunk_values)
        if np.any(inf_mask) or np.any(nan_mask):
            print(f"\nFound problematic values in chunk:")
            for col_idx in range(chunk_values.shape[1]):
                n_inf = np.sum(inf_mask[:, col_idx])
                n_nan = np.sum(nan_mask[:, col_idx])
                if n_inf > 0 or n_nan > 0:
                    print(f"Column {column_names[col_idx]}: {n_inf} inf values, {n_nan} NaN values")

        # Replace inf with nan
        chunk_values = np.nan_to_num(chunk_values, nan=np.nan, posinf=np.nan, neginf=np.nan)
        
        # Count non-NaN values per column
        chunk_count = (~np.isnan(chunk_values)).sum(axis=0)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunk_sum = np.nansum(chunk_values, axis=0)
            chunk_sum_squares = np.nansum(chunk_values ** 2, axis=0)
        
        # Update running sums
        if sum_values is None:
            sum_values = chunk_sum
            sum_squares = chunk_sum_squares
            count = chunk_count
        else:
            sum_values += chunk_sum
            sum_squares += chunk_sum_squares
            count += chunk_count

    # Calculate statistics with detailed reporting
    print("\nCalculating final statistics...")
    means = np.zeros_like(sum_values)
    stds = np.ones_like(sum_values)
    
    for i in range(len(means)):
        if count[i] > 0:
            means[i] = sum_values[i] / count[i]
            var = (sum_squares[i] / count[i]) - (means[i] ** 2)
            # Handle negative variance due to numerical instability
            stds[i] = np.sqrt(max(var, 0))
        else:
            print(f"Warning: Column {column_names[i]} has no valid values")
            means[i] = 0
            stds[i] = 1

    # Report on statistics
    print("\nStatistics summary:")
    for i, col in enumerate(column_names):
        print(f"\nColumn: {col}")
        print(f"  Mean: {means[i]}")
        print(f"  Std: {stds[i]}")
        print(f"  Valid values: {count[i]}/{total_rows}")
        if np.isnan(means[i]) or np.isnan(stds[i]):
            print("  WARNING: Invalid statistics!")

    # Save normalization parameters
    np.savez('normalization_params.npz', 
             means=means, 
             stds=stds,
             column_names=column_names)

    # Second pass with careful normalization
    print("\nSecond pass: normalizing data...")
    writer = None

    normalization_params = np.load('normalization_params.npz')
    means = normalization_params['means']
    stds = normalization_params['stds']
    # column_names = normalization_params['column_names']

        # Protect against zero or very small standard deviations
    eps = 1e-8
    safe_stds = np.where(np.abs(stds) < eps, 1.0, stds)

    for batch in tqdm(pf.iter_batches(batch_size=chunk_size)):
        df_chunk = batch.to_pandas()
        chunk_values = df_chunk.values
        
        # Replace inf with nan
        chunk_values = np.nan_to_num(chunk_values, nan=np.nan, posinf=np.nan, neginf=np.nan)
        
        # Normalize with protection against invalid values
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_values = (chunk_values - means[np.newaxis, :]) / safe_stds[np.newaxis, :]
        
        # Check for any remaining invalid values
        invalid_mask = ~np.isfinite(normalized_values)
        if np.any(invalid_mask):
            print(f"\nFound {np.sum(invalid_mask)} invalid values after normalization")
            # Replace invalid values with 0 or another sentinel value
            normalized_values[invalid_mask] = 0
            
        # Convert back to DataFrame
        normalized_chunk = pd.DataFrame(normalized_values, 
                                      columns=df_chunk.columns, 
                                      index=df_chunk.index)
        
        # Optional: Add some validation
        if np.any(np.isinf(normalized_values)) or np.any(np.isnan(normalized_values)):
            print("\nWarning: Invalid values in normalized data!")
            print(f"Inf values: {np.sum(np.isinf(normalized_values))}")
            print(f"NaN values: {np.sum(np.isnan(normalized_values))}")
        
        # Write normalized chunk
        if writer is None:
            schema = pa.Schema.from_pandas(normalized_chunk)
            writer = pq.ParquetWriter(output_file, schema)
        
        writer.write_table(pa.Table.from_pandas(normalized_chunk))

    if writer is not None:
        writer.close()

    print("\nNormalization complete!")
    print("Checking final statistics...")
    
    # Optional: Read a small sample to verify normalization
    sample_df = pd.read_parquet(output_file, columns=df_chunk.columns)
    print("\nSample statistics of normalized data:")
    print("Means range:", sample_df.mean().describe())
    print("Stds range:", sample_df.std().describe())

    print("Done!")

if __name__ == "__main__":
    main()