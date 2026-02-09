from pathlib import Path
from typing import Dict, Type

import pandas as pd
from rich.console import Console

console = Console()


class WellDataLoader:
    
    def __init__(self, config: Type):
        self.config = config
        self.dataset_name = config.DATASET_NAME
        self.wells: Dict[str, pd.DataFrame] = {}
    
    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        processed_data = self.preprocess_well(df.copy())
        return processed_data
    
    def load_from_directory(self, directory: str = None, split: str = 'train') -> Dict[str, pd.DataFrame]:
        if directory is None:
            if split == 'train':
                directory = Path(self.config.TRAIN_PATH)
            elif split == 'val':
                directory = Path(self.config.VAL_PATH)
            elif split == 'test':
                directory = Path(self.config.TEST_PATH)
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        else:
            directory = Path(directory)
        
        if not directory.exists():
            console.print(f"[red]Error: Directory does not exist {directory}[/red]")
            return {}
        
        wells = {}
        
        csv_files = list(directory.glob("*.csv"))
        
        if not csv_files:
            console.print(f"[yellow]Warning: No CSV files found in directory {directory}[/yellow]")
            return wells
        
        console.print(f"[blue]Found {len(csv_files)} CSV files in {directory}[/blue]")
        
        for file_path in csv_files:
            well_id = file_path.stem
            
            try:
                well_data = self.load_from_csv(str(file_path))
                wells[well_id] = well_data
                console.print(f"[green]OK Loaded: {well_id} ({len(well_data)} rows)[/green]")
            except Exception as e:
                console.print(f"[red]X Load failed: {well_id} - {str(e)}[/red]")
        
        self.wells = wells
        console.print(f"[green]OK Total loaded {len(wells)} wells[/green]")
        
        return wells
    
    def preprocess_well(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.config.FEATURE_COLUMNS:
            if col not in df.columns:
                console.print(f"[yellow]Warning: Missing feature column {col}, filling with 0[/yellow]")
                df[col] = 0.0
        
        if self.config.TARGET_COLUMN not in df.columns:
            console.print(f"[yellow]Warning: Missing target column {self.config.TARGET_COLUMN}[/yellow]")
        
        return df
    
    def load_all_splits(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        console.print(f"[blue]Start loading all splits for {self.dataset_name} dataset[/blue]")
        
        all_data = {}
        
        for split in ['train', 'val', 'test']:
            console.print(f"\n[cyan]Loading {split} data...[/cyan]")
            all_data[split] = self.load_from_directory(split=split)
        
        console.print(f"\n[green]OK {self.dataset_name} dataset loading complete[/green]")
        console.print(f"  Train: {len(all_data['train'])} wells")
        console.print(f"  Val: {len(all_data['val'])} wells")
        console.print(f"  Test: {len(all_data['test'])} wells")
        
        return all_data
    
    def get_well(self, well_id: str) -> pd.DataFrame | None:
        return self.wells.get(well_id)
    
    def get_all_wells(self) -> Dict[str, pd.DataFrame]:
        return self.wells


DataLoader = WellDataLoader