from os import listdir
from PIL import Image
from numpy import asarray
from numpy import savez_compressed
from mtcnn.mtcnn import MTCNN

# ##################>>>>>>>>>>>>> set this before run
file_idx = '1'
# ##################

folder = 'dataset' + file_idx + '/'


def add_result(detected_faces_func, face_array_func):
    detected_faces_func.append(face_array_func)


# extract a single face from a given photograph
def extract_face(results_list, detector_func, file_name):
    image = Image.open(file_name)
    image = image.convert('RGB')
    pixels = asarray(image)

    # detect faces in image with mtcnn detector
    results = detector_func.detect_faces(pixels)

    print(results)
    add_result(results_list, results)


def search_folder(detector_func):
    # just a handler to follow execution
    it = 0
    # each element of this array is one face
    results_list = []
    for filename in listdir(folder):
        path = folder + filename
        extract_face(results_list, detector_func, path)

        # follow execution
        it += 1
        print(it)
    return asarray(results_list)  # possible error


def detect_faces():
    # load mtcnn model for face detection
    detector = MTCNN()

    # search dataset folder images for face detection
    all_results = search_folder(detector)
    return all_results


def save_embeddings():
    # get info of all detected faces inside dataset
    all_results = detect_faces()

    print(all_results.shape)
    file_name = 'detection_results_' + file_idx + '.npz'
    savez_compressed(file_name, all_results)


if __name__ == "__main__":
    save_embeddings()
