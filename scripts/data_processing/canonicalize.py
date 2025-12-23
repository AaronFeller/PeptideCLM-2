from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, Manager
from functools import partial

def canonicalize_smiles(smiles, error_count):
    """Process a single SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        error_count.value += 1
        return None
    except Exception as e:
        error_count.value += 1
        return None

def process_chunk(args):
    """Process a chunk of SMILES strings."""
    chunk, error_count = args
    # Create partial function with error_count
    func = partial(canonicalize_smiles, error_count=error_count)
    chunk['canonical_smiles'] = chunk['smiles'].apply(func)
    return chunk

def main():
    # Number of processes to use
    num_processes = 24  # Adjust based on your CPU cores
    
    # Read the parquet file
    print("Reading parquet file...")
    smiles_data_split = pd.read_parquet('ARVF_all_smiles_data_split.parquet')
    total_molecules = len(smiles_data_split)
    print(f"Total molecules to process: {total_molecules:,}")

    # Process in chunks of 100k
    chunk_size = 100000
    chunks = np.array_split(smiles_data_split, max(1, total_molecules // chunk_size))
    
    # Create a shared counter for errors
    with Manager() as manager:
        error_count = manager.Value('i', 0)
        
        # Create process pool
        print(f"Starting processing with {num_processes} processes...")
        with Pool(processes=num_processes) as pool:
            # Create iterable of (chunk, error_count) tuples
            chunk_args = [(chunk, error_count) for chunk in chunks]
            
            # Process chunks with progress bar
            processed_chunks = list(tqdm(
                pool.imap(process_chunk, chunk_args),
                total=len(chunks),
                desc="Processing chunks",
                unit="chunk"
            ))

        # Get final error count
        final_error_count = error_count.value

    # Combine processed chunks
    print("Combining processed data...")
    result = pd.concat(processed_chunks, ignore_index=True)
    
    # Print statistics
    total_processed = len(result)
    successful = result['canonical_smiles'].notna().sum()
    failed = total_processed - successful
    
    print("\nProcessing Statistics:")
    print(f"Total molecules processed: {total_processed:,}")
    print(f"Successful canonicalizations: {successful:,}")
    print(f"Failed canonicalizations: {failed:,}")
    print(f"Error count: {final_error_count:,}")
    
    # Save results
    print("\nSaving results...")
    # Filter for successful canonicalizations
    result = result[result['canonical_smiles'].notna()]
    # Replace smiles with canonical_smiles and drop original smiles column
    result = result.drop(columns=['smiles']).rename(columns={'canonical_smiles': 'smiles'})
    print(f"{len(result):,} canonical SMILES strings saved.")
    result.to_parquet('ARVF_all_smiles_data_split_canonical.parquet', index=False)
    print("Done!")

if __name__ == "__main__":
    main()