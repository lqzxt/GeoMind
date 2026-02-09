from typing import Dict, List, Tuple

class FaciesConfig:
    DATASET_NAME = "facies"
    TRAIN_PATH = "well_dataset/facies/train"
    VAL_PATH = "well_dataset/facies/val"
    TEST_PATH = "well_dataset/facies/test"
    
    @staticmethod
    def get_model_path(split: str, model_name: str = None) -> str:
        if model_name is None:
            base_paths = {
                'train': "well_dataset/facies/train",
                'val': "well_dataset/facies/val",
                'test': "well_dataset/facies/test"
            }
            return base_paths.get(split, "well_dataset/facies/train")
        else:
            return f"well_dataset/facies/{split}_{model_name}"
    
    TARGET_COLUMN = "Facies"
    
    FEATURE_COLUMNS = [
        'Depth',
        'GR',
        'ILD_log10',
        'DeltaPHI',
        'PHIND',
        'PE',
        'NM_M'
    ]
    
    NUM_CLASSES = 9
    
    LABEL_MAPPING = {
        1: "Nonmarine sandstone",
        2: "Nonmarine coarse siltstone",
        3: "Nonmarine fine siltstone",
        4: "Marine siltstone and shale",
        5: "Mudstone (limestone)",
        6: "Wackestone (limestone)",
        7: "Dolomite",
        8: "Packstone-grainstone (limestone)",
        9: "Phylloid-algal bafflestone (limestone)"
    }
    
    CLASS_NAMES = [
        "Nonmarine sandstone",
        "Nonmarine coarse siltstone",
        "Nonmarine fine siltstone",
        "Marine siltstone and shale",
        "Mudstone (limestone)",
        "Wackestone (limestone)",
        "Dolomite",
        "Packstone-grainstone (limestone)",
        "Phylloid-algal bafflestone (limestone)"
    ]
    
    FEATURE_DESCRIPTIONS = {
        'Depth': 'Depth in feet',
        'GR': 'Gamma Ray - measures natural radioactivity',
        'ILD_log10': 'Deep induction resistivity (log10 scale)',
        'DeltaPHI': 'Difference between neutron and density porosity',
        'PHIND': 'Neutron-density porosity',
        'PE': 'Photoelectric effect',
        'NM_M': 'Nonmarine-marine indicator'
    }
    
    NORMALIZATION_CONFIG = {
        "method": "standard",
        "features_to_normalize": ['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M']
    }

class Force2020Config:
    DATASET_NAME = "FORCE-2020"
    TRAIN_PATH = "well_dataset/FORCE-2020/train"
    VAL_PATH = "well_dataset/FORCE-2020/dev"
    TEST_PATH = "well_dataset/FORCE-2020/test"
    
    @staticmethod
    def get_model_path(split: str, model_name: str = None) -> str:
        if model_name is None:
            base_paths = {
                'train': "well_dataset/FORCE-2020/train",
                'val': "well_dataset/FORCE-2020/dev",
                'test': "well_dataset/FORCE-2020/test"
            }
            return base_paths.get(split, "well_dataset/FORCE-2020/train")
        else:
            actual_split = 'dev' if split == 'val' else split
            return f"well_dataset/FORCE-2020/{actual_split}_{model_name}"
    
    TARGET_COLUMN = "label"
    
    FEATURE_COLUMNS = [
        'DEPTH',
        'CALI',
        'DTC',
        'GR',
        'NPHI',
        'RHOB',
        'X_LOC',
        'Y_LOC',
        'Z_LOC'
    ]
    
    NUM_CLASSES = 5
    
    LABEL_MAPPING = {
        "Shale": "Shale",
        "Sandstone": "Sandstone",
        "Limestone": "Limestone",
        "Marl": "Marl",
        "Sandstone/Shale": "Sandstone/Shale"
    }
    
    CLASS_NAMES = ["Shale", "Sandstone", "Limestone", "Marl", "Sandstone/Shale"]
    
    FEATURE_DESCRIPTIONS = {
        'DEPTH': 'Depth',
        'CALI': 'Caliper - borehole diameter',
        'DTC': 'Sonic compressional slowness',
        'GR': 'Gamma Ray',
        'NPHI': 'Neutron porosity',
        'RHOB': 'Bulk density',
        'X_LOC': 'X coordinate',
        'Y_LOC': 'Y coordinate',
        'Z_LOC': 'Z coordinate (depth)'
    }
    
    NORMALIZATION_CONFIG = {
        "method": "standard",
        "features_to_normalize": ['DEPTH', 'CALI', 'DTC', 'GR', 'NPHI', 'RHOB']
    }

class GeolinkConfig:
    DATASET_NAME = "geolink"
    TRAIN_PATH = "well_dataset/geolink/train"
    VAL_PATH = "well_dataset/geolink/val"
    TEST_PATH = "well_dataset/geolink/test"
    
    @staticmethod
    def get_model_path(split: str, model_name: str = None) -> str:
        if model_name is None:
            base_paths = {
                'train': "well_dataset/geolink/train",
                'val': "well_dataset/geolink/val",
                'test': "well_dataset/geolink/test"
            }
            return base_paths.get(split, "well_dataset/geolink/train")
        else:
            return f"well_dataset/geolink/{split}_{model_name}"
    
    TARGET_COLUMN = "Lithology_geolink"
    
    FEATURE_COLUMNS = [
        'DEPT',
        'CALI',
        'NPHI',
        'RHOB',
        'GR',
        'DTC',
        'RDEP',
        'RSHA',
        'RMED'
    ]
    
    NUM_CLASSES = 11
    
    LABEL_MAPPING = {
        1: "Sand",
        2: "Silty Sand",
        3: "Cross Bedded Sand",
        4: "Shaly Sandstone",
        5: "Silt",
        6: "Shaly Silt",
        7: "Silty Shale",
        11: "Limestone",
        12: "Argillaceous Limestone",
        13: "Marlstone",
        16: "Sandy Limestone"
    }
    
    CLASS_NAMES = [
        "Sand",
        "Silty Sand",
        "Cross Bedded Sand",
        "Shaly Sandstone",
        "Silt",
        "Shaly Silt",
        "Silty Shale",
        "Limestone",
        "Argillaceous Limestone",
        "Marlstone",
        "Sandy Limestone"
    ]
    
    FEATURE_DESCRIPTIONS = {
        'DEPT': 'Depth',
        'CALI': 'Caliper - borehole diameter',
        'NPHI': 'Neutron porosity',
        'RHOB': 'Bulk density',
        'GR': 'Gamma Ray',
        'DTC': 'Sonic compressional slowness',
        'RDEP': 'Deep resistivity',
        'RSHA': 'Shallow resistivity',
        'RMED': 'Medium resistivity'
    }
    
    NORMALIZATION_CONFIG = {
        "method": "standard",
        "features_to_normalize": ['DEPT', 'CALI', 'NPHI', 'RHOB', 'GR', 'DTC', 'RDEP', 'RSHA', 'RMED']
    }

class SEAMConfig:
    DATASET_NAME = "SEAM"
    TRAIN_PATH = "well_dataset/SEAM/train"
    VAL_PATH = "well_dataset/SEAM/val"
    TEST_PATH = "well_dataset/SEAM/test"
    
    @staticmethod
    def get_model_path(split: str, model_name: str = None) -> str:
        if model_name is None:
            base_paths = {
                'train': "well_dataset/SEAM/train",
                'val': "well_dataset/SEAM/val",
                'test': "well_dataset/SEAM/test"
            }
            return base_paths.get(split, "well_dataset/SEAM/train")
        else:
            return f"well_dataset/SEAM/{split}_{model_name}"
    
    TARGET_COLUMN = "LayerGeoIndex"
    
    FEATURE_COLUMNS = [
        'Depth',
        'BedDipX',
        'BedDipY',
        'PorosityTotal',
        'ResistivityHorizontal',
        'TTIDipX',
        'TTIDipY',
        'Vp',
        'VsElasticOriginal'
    ]
    
    NUM_CLASSES = 7
    
    LABEL_MAPPING = {
        2: "Mother salt",
        3: "Cretaceous",
        4: "Oligocene–Paleocene",
        5: "Lower Miocene",
        8: "Pliocene",
        9: "Pleistocene",
        10: "Water"
    }
    
    CLASS_NAMES = [
        "Mother salt",
        "Cretaceous",
        "Oligocene–Paleocene",
        "Lower Miocene",
        "Pliocene",
        "Pleistocene",
        "Water"
    ]
    
    FEATURE_DESCRIPTIONS = {
        'Depth': 'Depth',
        'BedDipX': 'Bed dip in X direction',
        'BedDipY': 'Bed dip in Y direction',
        'PorosityTotal': 'Total porosity',
        'ResistivityHorizontal': 'Horizontal resistivity',
        'TTIDipX': 'TTI dip in X direction',
        'TTIDipY': 'TTI dip in Y direction',
        'Vp': 'P-wave velocity',
        'VsElasticOriginal': 'Original S-wave velocity'
    }
    
    NORMALIZATION_CONFIG = {
        "method": "standard",
        "features_to_normalize": [
            'Depth', 'BedDipX', 'BedDipY', 'PorosityTotal', 'ResistivityHorizontal',
            'TTIDipX', 'TTIDipY', 'Vp', 'VsElasticOriginal'
        ]
    }

def get_dataset_config(dataset_name: str):
    configs = {
        'facies': FaciesConfig,
        'FORCE-2020': Force2020Config,
        'geolink': GeolinkConfig,
        'SEAM': SEAMConfig
    }
    
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")
    
    return configs[dataset_name]