import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
 
 
def main():    
    embeddings_file = 'data/word_embeddings.p'
    embed_dict = load_embeddings(embeddings_file)
    
    vocabulary = embed_dict.keys()
    word_vec = np.array(embed_dict.values())

    ############################################################################
    # You should modify this part by selecting a subset of word embeddings 
    # for better visualization
    ############################################################################
    
    print(word_vec.shape)  # 1067, 100  
    old_word_vec = word_vec
    word_vec = word_vec[800:1000, :]
    print(len(vocabulary)) ## 1067
    print(vocabulary[800:1000])
    i = vocabulary.index('dinner')
    k = vocabulary.index('rainy')
    print(i,k)
    result = cosine_similarity(old_word_vec[i, :], old_word_vec[k, :])
    print(result)
    vocabulary = vocabulary[800:1000]

    ############################################################################

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)    
    Y = tsne.fit_transform(word_vec)
 
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()


def cosine_similarity(A,B):
    '''
    (nparray, nparray) -> float
    Returns the cosine similarity of vector A and B in order to determine 
    how sintactically similar they are in a n-vector space.
    '''
    return (np.dot(A,B) / (np.linalg.norm(A) * np.linalg.norm(B)))

    
def load_embeddings(file_name):
    """ Load in the embeddings """
    return pickle.load(open(file_name, 'rb'))


if __name__ == '__main__':    
    main()
