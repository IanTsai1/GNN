import numpy as np
from scipy.sparse import csr_matrix
from GNNs_unsupervised import GNN

def load_data():
    # Load saved features and edge index
    raw_features = np.load('final_features_tensor.npy')
    edge_index = np.load('edge_index.npy')
    edge_weight = np.load('edge_weight.npy')

    # Construct the adjacency matrix
    row = edge_index[0]
    col = edge_index[1]
    #TODO: TRY TO ADD CUSTOM EDGE WEIGHTS TO THE MODEL
    adj_matrix = csr_matrix((np.ones(len(row)), (row, col)), shape=(raw_features.shape[0], raw_features.shape[0]))
    #adj_matrix = csr_matrix((edge_weight, (row, col)), shape=(raw_features.shape[0], raw_features.shape[0]))

    return adj_matrix, raw_features

def example_with_my_data():
    # Load the adjacency matrix and features
    adj_matrix, raw_features = load_data()

    # Set up GNN (assuming it's unsupervised and uses 'gat' as the model type)
    gnn = GNN(adj_matrix, features=raw_features, supervised=False, model='gat', device='cpu')

    # Train the model
    gnn.fit()

    # Get node embeddings
    embs = gnn.generate_embeddings()

    # Since this is unsupervised, you can use embeddings for downstream tasks, clustering, etc.
    # For now, we just print out the embeddings
    #print(embs)
    np.save('node_embeddings.npy', embs)

if __name__ == "__main__":
    example_with_my_data()
