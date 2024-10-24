from numpy import asarray
from numpy import savez_compressed
from numpy import load
from PIL import Image
from os import remove
from os import listdir
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from numpy import expand_dims


# ##################>>>>>>>>>>>>> set these before run
file_idx = '1'
save_image_start_index = 9877
# ##################

save_image = False
do_remove = False
save_image_index = save_image_start_index
folder = 'dataset' + file_idx + '/'
min_confidence = 0.95
min_face_width = 70
min_face_height = 100
required_size = (160, 160)
model_input = (160, 160, 3)


def save_face(path, face_save):
    global save_image_index
    image_save = Image.fromarray(face_save)
    image_save = image_save.resize((200, 200))
    slash_index = path.rfind('/')
    point_index = path.rfind('.')
    file_name = path[:slash_index + 1] + str(save_image_index) + path[point_index:]
    image_save.save(file_name)
    save_image_index += 1


def add_face(detected_faces_func, face_array_func):
    detected_faces_func.append(face_array_func)


def validate_face(confidence, height, width, pixels, x1, x2, y1, y2, img_width, img_height):
    # check if confidence of the image in enough
    if confidence > min_confidence:
        # check if image size is enough for embedding production and user perception
        if width > min_face_width and height > min_face_height:
            margin_ub = height // 5
            margin_lr = width // 5
            # check if the image can be cropped with sufficient border
            if y1 - margin_ub > 0 and x1 - margin_lr > 0 and y2 + margin_ub < img_height and x2 + margin_lr < img_width:
                face_save = pixels[y1 - margin_ub:y2 + margin_ub, x1 - margin_lr:x2 + margin_lr]
                return face_save
    return None


# extract a single face from a given photograph
def extract_face(detected_faces_func, this_results_f, file_name):
    global save_image_index
    image = Image.open(file_name)
    image = image.convert('RGB')
    img_width, img_height = image.size
    pixels = asarray(image)
    results = this_results_f
    if results:
        for idx in range(len(results)):
            x1, y1, width, height = results[idx]['box']
            confidence = results[idx]['confidence']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            # check if this face is validate in multiple criteria
            f_save = validate_face(confidence, height, width, pixels, x1, x2, y1, y2, img_width, img_height)
            if f_save is not None:
                face = pixels[y1:y2, x1:x2]
                image = Image.fromarray(face)
                image = image.resize(required_size)
                face_array = asarray(image)

                # add face to list of faces
                add_face(detected_faces_func, face_array)

                # save face with appropriate prefix after original image name
                if save_image:
                    save_face(file_name, f_save)
                else:
                    save_image_index += 1


def search_folder(all_results_f):
    # just a handler to follow execution
    it = 0
    # each element of this array is one face
    detected_faces = []
    for filename in listdir(folder):
        path = folder + filename
        extract_face(detected_faces, all_results_f[it], path)
        if do_remove:
            remove(path)

        # follow execution
        it += 1
        print(it)
    return asarray(detected_faces)  # possible error


def detect_faces(all_results_f):
    # search dataset folder images for face detection
    detected_faces = search_folder(all_results_f)
    return detected_faces


# get the face embedding for one face
def get_embedding(model_func, face_pixels_func):
    samples = expand_dims(face_pixels_func, axis=0)
    y_hat = model_func.predict(samples)
    return y_hat[0]


# convert each face in the train set to an embedding
def convert_to_embedding(detected_face_func, model_func):
    i = 1
    new_detected_face = []
    for face_pixels in detected_face_func:
        embedding = get_embedding(model_func, face_pixels)
        print(i)
        i += 1
        new_detected_face.append(embedding)
    return new_detected_face


def save_embeddings():
    # get info of all detected faces inside dataset
    rtv_file_name = 'detection_results_' + file_idx + '.npz'
    data = load(rtv_file_name, allow_pickle=True)  # possible error
    all_results = data['arr_0']

    print(all_results.shape)

    # produce detected faces from results
    detected_face = detect_faces(all_results)
    detected_face = asarray(detected_face, 'float32')
    # load facenet model
    samples = preprocess_input(detected_face, version=2)
    model = VGGFace(model='vgg16', include_top=False, input_shape=model_input, pooling='avg')
    print('Model loaded')

    # get embedding of detected faces
    embeddings = convert_to_embedding(samples, model)
    embeddings = asarray(embeddings)
    print(embeddings.shape)

    save_image_end_index = save_image_index - 1

    # save arrays to one file in compressed format
    file_name = 'vgg_embedding_' + file_idx + '_[' + str(save_image_start_index) \
                + "-" + str(save_image_end_index) + ']' + '.npz'
    savez_compressed(file_name, embeddings)


if __name__ == "__main__":
    save_embeddings()
