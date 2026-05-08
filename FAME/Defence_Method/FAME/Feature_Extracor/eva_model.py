import os
import sys
project_path = os.getcwd()
sys.path.append(project_path)


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from Defence_Method.FAME.Feature_Extracor.model import CF_Extractor, ProjectionHead
from DataTool_Code.DataLoader import Load_Data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split

def evaluate_features(features, labels):
    print("\n" + "="*30)
    print("Computing feature quality metrics...")
    
    #  1. k-NN accuracy.
    #  Split features into train/test for evaluation.
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=1)  # 1-NN is the most sensitive proximity metric
    knn.fit(X_train, y_train)
    knn_acc = knn.score(X_test, y_test)
    print(f"-> Top-1 k-NN Accuracy: {knn_acc:.4f}")
    
    #  2. Silhouette score.
    #  Subsample for efficiency on large datasets.
    if len(features) > 10000:
        sample_idx = np.random.choice(len(features), 10000, replace=False)
        s_score = silhouette_score(features[sample_idx], labels[sample_idx])
    else:
        s_score = silhouette_score(features, labels)
    print(f"-> Silhouette Score: {s_score:.4f}  (range -1 to 1, higher is better)")
    
    #  3. Calinski-Harabasz index.
    ch_score = calinski_harabasz_score(features, labels)
    print(f"-> Calinski-Harabasz Index: {ch_score:.2f} (higher is better)")
    print("="*30 + "\n")


# --- Font configuration for cross-platform compatibility ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_models_and_extract_features(test_x, extractor_path, proj_path):
    """Load pre-trained models and extract L2-normalized 128-dim embeddings."""
    print("Loading models and extracting features...")
    
    # get input shape, e.g. (2000, 1)
    input_shape = test_x.shape[1:]
    
    # instantiate and build models
    extractor = CF_Extractor.build(input_shape)
    projection_head = ProjectionHead.build()  # default input_dim=256
    
    # load pre-trained weights
    extractor.load_weights(extractor_path)
    projection_head.load_weights(proj_path)
    
    # forward pass to extract features (use batch_size to avoid OOM on large test sets)
    print("-> Extracting 256-dim representation features (r)...")
    r_features = extractor.predict(test_x, batch_size=64)
    
    print("-> Extracting 128-dim projected features (e)...")
    e_features = projection_head.predict(r_features, batch_size=64)
    
    # L2-normalize to match the hypersphere geometry used during SupCon training
    e_normalized = tf.math.l2_normalize(e_features, axis=1).numpy()
    
    return e_normalized

def plot_tsne_clusters(features, labels, save_path=f"tsne_visualization.pdf"):
    """Reduce features with t-SNE and plot a 2D scatter diagram (supports many classes)."""
    print("Running t-SNE reduction (this may take a while)...")
    
    # Step 1: reduce to 2D with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    print("t-SNE complete. Plotting scatter diagram...")
    plt.figure(figsize=(14, 10))  # wider canvas to accommodate the side legend
    
    # Step 2: get unique class labels
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    
    # Step 3: generate a distinct colormap (nipy_spectral has better color separation than rainbow)
    cmap = plt.get_cmap('nipy_spectral')
    # uniformly sample colors across the colormap
    colors = [cmap(i / num_classes) for i in range(num_classes)]
    
    # Step 4: plot each class as a scatter
    for i, cls in enumerate(unique_classes):
        idx = (labels == cls)
        plt.scatter(
            features_2d[idx, 0], 
            features_2d[idx, 1], 
            color=colors[i],      # color sampled from colormap
            label=f'Class {int(cls)}', 
            alpha=0.6,            # semi-transparent to show overlaps
            edgecolors='none',    # no edge color to reduce visual clutter
            s=10                  # small markers to avoid crowding
        )
    
    # Step 5: configure chart labels and title
    plt.title(f't-SNE Visualization of {num_classes} Classes (Contrastive Features)', fontsize=15, pad=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    
    # Step 6: legend layout for many classes
    # ncol=4: spread into 4 columns to prevent overflow
    # fontsize=6: small font to fit many class labels
    # handletextpad: reduce gap between marker and text
    plt.legend(
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        fontsize=6, 
        ncol=4, 
        markerscale=2,         # enlarge legend markers for readability
        handletextpad=0.1
    )
    
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Step 7: save figure (bbox_inches='tight' prevents the side legend from being clipped)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    # plt.show()

if __name__ == "__main__":
    # load evaluation data
    PreTrain_DataSet_Name='AWF100'
    Tar_DataSet_Name='AWF200'
    test_x, test_y = Load_Data(Tar_DataSet_Name, 'test')
    
    # ================= 2. Model Paths =================
    save_dir = "Defence_Method/FAME/Feature_Extracor/File_Save"
    extractor_path = os.path.join(save_dir, f"FeaturExtractor_in_{PreTrain_DataSet_Name}.h5")
    proj_path = os.path.join(save_dir, f"ProjectionHead_in_{PreTrain_DataSet_Name}.h5")
    
    # ================= 3. Run Workflow =================
    if os.path.exists(extractor_path) and os.path.exists(proj_path):
        features_128d = load_models_and_extract_features(test_x, extractor_path, proj_path)
        plot_save_path = os.path.join(save_dir, f"tsne_visualization_{PreTrain_DataSet_Name}_to_{Tar_DataSet_Name}.pdf")
        plot_tsne_clusters(features_128d, test_y, save_path=plot_save_path)
        evaluate_features(features_128d, test_y)
    else:
        print(f"Model weights not found. Please check {extractor_path} and {proj_path}.")
    
