\"\"\"Run the whole P2 pipeline in order:  python -m src.p2_run_all\"\"\"
from src import p2_tabular, p2_hybrid, p2_shap

if __name__ == \"__main__\":
    p2_tabular.main()   # tabular benchmark + best_tabular_probs.npy
    p2_hybrid.main()    # GNN-embedding hybrid (needs P3 embeddings)
    p2_shap.main()      # SHAP + false-negative analysis
