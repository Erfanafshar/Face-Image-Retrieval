from numpy import load
from os import listdir
from numpy import asarray
from numpy import savez_compressed
from numpy import concatenate

# ##################>>>>>>>>>>>>> set these before run
cnn_name = 'vgg'
# ##################

folder = 'embedding file parts/'
output_file_name = cnn_name + '_embedding.npz'


# enumerate files
def search_folder(folder_func):
    it = 0
    all_embeddings_func = []
    for filename in listdir(folder_func):
        path = folder_func + filename

        data = load(path, allow_pickle=True)
        detected_face = data['arr_0']
        print(detected_face.shape)
        if len(all_embeddings_func) == 0:
            all_embeddings_func = detected_face
        else:
            all_embeddings_func = concatenate((all_embeddings_func, detected_face))
        it += 1
        print(it)
    return asarray(all_embeddings_func)


def concat():
    all_embeddings = search_folder(folder)
    print(all_embeddings.shape)
    # save arrays to one file in compressed format
    savez_compressed(output_file_name, all_embeddings)


if __name__ == "__main__":
    concat()
