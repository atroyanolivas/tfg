from tensorflow.keras.models import load_model

pca_mlp = load_model("./test/models/PCA_MLP.keras")

pca_mlp.summary()