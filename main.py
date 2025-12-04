from src.preprocessing.data_cleaner import save_processed_dataset
from src.preprocessing.normalizer import normalize_dataset
from src.preprocessing.binarizer import generate_binary_dataset
from src.pipeline import run_pipeline

if __name__ == "__main__":
    print("="*60)
    print("ADAPTIVE LEARNING SYSTEM - PIPELINE MODE")
    print("="*60)
    
    print("\n[PREPROCESSING] Loading & processing datasets...")
    save_processed_dataset()
    normalize_dataset()
    generate_binary_dataset()
    
    print("\n[PIPELINE] Running Hebbian → ART1 → MLP pipeline...")
    pipeline, results = run_pipeline()
    
    print("\n" + "="*60)
    print("PIPELINE TRAINING COMPLETE")
    print("="*60)
    print(f"Hebbian features shape: {results['hebbian_features_shape']}")
    print(f"ART1 features shape: {results['art1_features_shape']}")
    print(f"ART1 clusters formed: {results['art1_clusters'].max() + 1}")
    print("="*60)