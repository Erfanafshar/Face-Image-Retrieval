# backend
from http.server import HTTPServer, SimpleHTTPRequestHandler
import base64

# AI
from PIL import Image
from gensim import similarities
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from arcface import ArcFace
from keras.models import load_model
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from numpy import expand_dims
from numpy import load
from numpy import mean
from numpy import std

import datetime
import threading

# for real case
dataset_path = "dataset\\"
facenet_emb_path = "facenet_embedding\\"
arcface_emb_path = "arcface_embedding\\"
vgg_emb_path = "vgg_embedding\\"

# for development
# dataset_path = "dataset_d\\"
# facenet_emb_path = "facenet_embedding_d\\"
# arcface_emb_path = "arcface_embedding_d\\"
# vgg_emb_path = "vgg_embedding_d\\"

# for test
# dataset_path = "dataset_t\\"
# facenet_emb_path = "facenet_embedding_t\\"
# arcface_emb_path = "arcface_embedding_t\\"
# vgg_emb_path = "vgg_embedding_t\\"

host_name = "localhost"
tmp_file_name = ""
server_port = 8080
model_input = (160, 160, 3)

num_rtv_img = 12
num_img_folder = 13        # min -> 1 & max -> 13 % max with test -> 14
models_coefficients = [0.44, 0.29, 0.27]    # facenet arcface vggface

model_f = None
model_a = None
model_v = None
detector = None
index_f = []
index_a = []
index_v = []


class SearchThread(threading.Thread):
    def __init__(self, name, model_name):
        threading.Thread.__init__(self)
        self.name = name
        self.model_name = model_name
        self._return = None

    def run(self):
        print("Starting " + self.name)
        self._return = do_search(self.model_name, False)
        print("Exiting " + self.name)

    def join(self, timeout=None):
        threading.Thread.join(self)
        return self._return


class SearchThread2(threading.Thread):
    def __init__(self, name, model_name):
        threading.Thread.__init__(self)
        self.name = name
        self.model_name = model_name
        self._return = None

    def run(self):
        print("Starting " + self.name)
        self._return = do_search(self.model_name, True)
        print("Exiting " + self.name)

    def join(self, timeout=None):
        threading.Thread.join(self)
        return self._return


def shape_emb_data(dataset_embedding_func):
    lst = []
    for vec in dataset_embedding_func:
        ls = []
        for i in range(len(vec)):
            ls.append((i, vec[i]))
        lst.append(ls)
    return lst


def load_models():
    global model_f, model_a, model_v, detector
    model_f = load_model('facenet_keras.h5')
    model_a = ArcFace.ArcFace()
    model_v = VGGFace(model='vgg16', include_top=False, input_shape=model_input, pooling='avg')
    detector = MTCNN()


def create_index(file_name, feature_num):
    data = load(file_name)
    dataset_embedding = data['arr_0']
    dataset_embedding_shaped = shape_emb_data(dataset_embedding)
    index = similarities.MatrixSimilarity(dataset_embedding_shaped, num_features=feature_num)
    return index


def create_facenet_indexes():
    global index_f
    for i in range(num_img_folder):
        index_f.append(create_index(facenet_emb_path + 'facenet_embedding_' + str(i) + '.npz', 128))


def create_arcface_indexes():
    global index_a
    for i in range(num_img_folder):
        index_a.append(create_index(arcface_emb_path + 'arcface_embedding_' + str(i) + '.npz', 512))


def create_vgg_indexes():
    global index_v
    for i in range(num_img_folder):
        index_v.append(create_index(vgg_emb_path + 'vgg_embedding_' + str(i) + '.npz', 512))


def load_models_and_data():
    load_models()
    create_facenet_indexes()
    create_arcface_indexes()
    create_vgg_indexes()


def get_search_flags(selected_option, facenet_check, arcface_check, vgg_check):
    if selected_option == "standard":  # previous -> advance
        # coefficients of [facenet arcface vgg]
        search_mode = models_coefficients

        # determine multiple cnn or not
        multi_mode = False

        return search_mode, multi_mode

    if selected_option == "multiple":  # previous -> basic
        # binary value of [facenet arcface vgg]
        search_mode = [-1]

        # determine multiple cnn or not
        multi_mode = False

        if facenet_check == "false":
            if arcface_check == "false":
                if vgg_check == "false":
                    search_mode = [0]
                else:
                    search_mode = [1]
            else:
                if vgg_check == "false":
                    search_mode = [2]
                else:
                    search_mode = [3]
                    multi_mode = True

        else:
            if arcface_check == "false":
                if vgg_check == "false":
                    search_mode = [4]
                else:
                    search_mode = [5]
                    multi_mode = True
            else:
                if vgg_check == "false":
                    search_mode = [6]
                    multi_mode = True
                else:
                    search_mode = [7]
                    multi_mode = True
        return search_mode, multi_mode

    if selected_option == "weighted":  # previous -> expert
        # coefficients of [facenet arcface vgg]
        search_mode = [int(facenet_check) / 100, int(arcface_check) / 100, int(vgg_check) / 100]

        # determine multiple cnn or not
        multi_mode = False

        return search_mode, multi_mode


class Server(SimpleHTTPRequestHandler):
    def do_POST(self):
        global tmp_file_name
        tmp_file_name = "temp"
        # start time of the request
        t1 = datetime.datetime.now()

        # read input image
        content_length = int(self.headers['Content-Length'])
        received_data = self.rfile.read(content_length)

        file_ext = self.headers["file_extension"]
        if file_ext == "":
            file_ext = "jpg"
        tmp_file_name += ("." + file_ext)
        # save input image
        with open(tmp_file_name, 'wb') as outfile:
            outfile.write(received_data)

        # set search mode
        selected_option = self.headers["option"]
        facenet_check = self.headers["facenet"]
        arcface_check = self.headers["arcface"]
        vgg_check = self.headers["vgg"]

        search_mode, multi_mode = get_search_flags(selected_option, facenet_check, arcface_check, vgg_check)

        # search for similar images in dataset
        each_score_data, total_score_data = "", ""
        if selected_option == "multiple":
            retrieved_images_data = face_search(search_mode)
        else:
            retrieved_images_data, each_score_data, total_score_data = face_search(search_mode)

        # no face detected in image
        if retrieved_images_data is None:
            sending_str = ""
            sending_date = sending_str.encode('ISO-8859-1')

        # creating appropriate formed string to send images to user
        else:
            if multi_mode:
                sending_str = ""
                for lst in retrieved_images_data:
                    for itm in lst:
                        sending_str += itm + "%%%%%"
                    sending_str = sending_str[:-5]
                    sending_str += "$$$$$"
                sending_str = sending_str[:-5]
                sending_date = sending_str.encode('ISO-8859-1')
            else:
                sending_str = ""
                for itm in retrieved_images_data:
                    sending_str += itm + "%%%%%"
                sending_str = sending_str[:-5]
                sending_date = sending_str.encode('ISO-8859-1')

        # send response headers to user
        self.send_response(200)
        self.send_header('Content-type', 'image/jpeg')

        if selected_option == "weighted":
            self.send_header('each_score', each_score_data)
            self.send_header('total_score', total_score_data)

        self.end_headers()
        self.wfile.write(sending_date)

        # end time of request
        dt = datetime.datetime.now() - t1
        # calculating response time of request
        print("response time: " + str(dt))


class Embedding:
    def __init__(self, mdl_name):
        self.model_name = mdl_name
        if self.model_name == 'facenet':
            self.required_size = (160, 160)
        elif self.model_name == 'arcface':
            self.required_size = (112, 112)
        elif self.model_name == 'vgg':
            self.required_size = (160, 160)

    def extract_face(self):
        image = Image.open(tmp_file_name)
        image = image.convert('RGB')
        pixels = asarray(image)
        results = detector.detect_faces(pixels)
        if not results:
            print("no face detected in image")
            return None
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(self.required_size)
        face_array = asarray(image)
        return face_array

    def get_embedding(self, face_pixels):
        if self.model_name == 'facenet':
            face_pixels = face_pixels.astype('float32')
            mean, std = face_pixels.mean(), face_pixels.std()
            face_pixels = (face_pixels - mean) / std

            samples = expand_dims(face_pixels, axis=0)
            y_hat = model_f.predict(samples)
            return y_hat[0]

        elif self.model_name == 'arcface':
            y_hat = model_a.calc_emb(face_pixels)
            return y_hat

        elif self.model_name == 'vgg':
            face_pixels = face_pixels.astype('float32')
            samples = expand_dims(face_pixels, axis=0)
            sample = preprocess_input(samples, version=2)

            y_hat = model_v.predict(sample)
            return y_hat[0]

    def embedding(self):
        detected_face = self.extract_face()
        if detected_face is None:
            return None
        new_embedding = self.get_embedding(detected_face)
        new_embedding = asarray(new_embedding)
        return new_embedding


def shape_emb_new(new_embedding):
    lst = []
    for i in range(len(new_embedding)):
        lst.append((i, new_embedding[i]))
    return lst


def resort(tops, tops_index):
    for j in range(1, len(tops)):
        if tops[0] < tops[j]:
            if j == 1:
                break
            else:
                tmp = tops[0]
                tops[0: j - 1] = tops[1: j]
                tops[j - 1] = tmp

                tmp_index = tops_index[0]
                tops_index[0: j - 1] = tops_index[1: j]
                tops_index[j - 1] = tmp_index
                break
        else:
            if j == len(tops) - 1:
                tmp = tops[0]
                tops[0: j] = tops[1: j + 1]
                tops[j] = tmp

                tmp_index = tops_index[0]
                tops_index[0: j] = tops_index[1: j + 1]
                tops_index[j] = tmp_index


def get_top_indexes(lst, cnn_name):
    tops = [-10 for _ in range(num_rtv_img)]
    tops_index = [-1 for _ in range(num_rtv_img)]
    if cnn_name == 'facenet':
        threshold = 0.25
    elif cnn_name == 'arcface':
        threshold = 0.20
    elif cnn_name == 'vgg':
        threshold = 0.35
    else:
        threshold = 1.00

    for i in range(len(lst)):
        if threshold < lst[i]:
            if tops[0] < lst[i]:
                tops[0] = lst[i]
                tops_index[0] = i
                resort(tops, tops_index)
    return [tops_index, tops]


# possible error (if faces with same score exist)
def get_final_idx_vals(top_indexes, top_values):
    tmp = []
    tmp_index = []
    tmp_position = [0 for _ in range(num_img_folder)]
    final_folder_index = []
    final_index = []
    final_values = []

    # initialize
    for i in range(num_img_folder):
        tmp.append(top_values[i][0])
        tmp_index.append(top_indexes[i][0])

    # loop
    for cnt in range(num_rtv_img):
        # find max index in tmp
        max_index = tmp.index(max(tmp))

        # insert max index to final
        final_folder_index.append(max_index)
        final_index.append(tmp_index[max_index])
        final_values.append(tmp[max_index])

        # update tmp position
        tmp_position[max_index] += 1

        if (cnt + 1) == num_rtv_img:
            break

        # replace next item in tmp list
        tmp[max_index] = top_values[max_index][tmp_position[max_index]]
        tmp_index[max_index] = top_indexes[max_index][tmp_position[max_index]]

    print(final_folder_index)
    print(final_index)
    print(final_values)
    print()
    return final_folder_index, final_index, final_values


def search(final_folder_index, final_index):
    images_data = []
    for i in range(num_rtv_img):
        image_path = dataset_path + "dataset" + str(final_folder_index[i]) + "\\" + str(final_index[i]) + '.jpg'

        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            images_data.append(image_data)
    return images_data


def do_search(model_name, just_index):
    embedding_instance = Embedding(model_name)
    new_embedding = embedding_instance.embedding()
    if new_embedding is None:
        return None

    # shape new embedding for gensim
    new_embedding_shaped = shape_emb_new(new_embedding)

    # search for most similar images in dataset
    sims = []
    if model_name == 'facenet':
        for i in range(num_img_folder):
            sims.append(index_f[i][new_embedding_shaped])

    elif model_name == 'arcface':
        for i in range(num_img_folder):
            sims.append(index_a[i][new_embedding_shaped])

    elif model_name == 'vgg':
        for i in range(num_img_folder):
            sims.append(index_v[i][new_embedding_shaped])

    else:
        print("wrong model name")
        return None

    top_indexes = []
    top_values = []
    for i in range(num_img_folder):
        top_indexes_values = get_top_indexes(sims[i], model_name)
        tmp_idx = top_indexes_values[0]
        tmp_idx.reverse()
        top_indexes.append(tmp_idx)

        tmp_val = top_indexes_values[1]
        tmp_val.reverse()
        top_values.append(tmp_val)

    print(model_name)
    final_folder_index, final_index, final_values = get_final_idx_vals(top_indexes, top_values)

    if just_index:  # combination methods
        return [final_folder_index, final_index, final_values]
    else:           # basic mode
        # get files of images from dataset
        images_data = search(final_folder_index, final_index)
        return images_data


def combination_search(index_value_f, index_value_a, index_value_v, coefs):
    # 0.standardize raw scores of models
    # means = [mean(index_value_f[2]), mean(index_value_a[2]), mean(index_value_v[2])]
    # stds = [std(index_value_f[2]), std(index_value_a[2]), std(index_value_v[2])]
    # standard_vals1 = (index_value_f[2] - means[0]) / stds[0]
    # standard_vals1 = standard_vals1 - 2 * min(standard_vals1)
    #
    # standard_vals2 = (index_value_a[2] - means[1]) / stds[1]
    # standard_vals2 = standard_vals2 - 2 * min(standard_vals2)
    #
    # standard_vals3 = (index_value_v[2] - means[2]) / stds[2]
    # standard_vals3 = standard_vals3 - 2 * min(standard_vals3)

    # 0.standardize with percentage
    maxes = [max(index_value_f[2]), max(index_value_a[2]), max(index_value_v[2])]
    standard_vals1 = index_value_f[2] / maxes[0]
    standard_vals2 = index_value_a[2] / maxes[1]
    standard_vals3 = index_value_v[2] / maxes[2]

    # 1.create list of all indexes with all values
    idx_val_dict = {}

    # info extraction
    folder_index = index_value_f[0] + index_value_a[0] + index_value_v[0]
    indexes = index_value_f[1] + index_value_a[1] + index_value_v[1]
    values = standard_vals1.tolist() + standard_vals2.tolist() + standard_vals3.tolist()

    # dict creation loop -> (length between 12 and 36)
    cnt = 0
    for fi, i, v in zip(folder_index, indexes, values):
        key = (fi, i)
        sub_idx = cnt // num_rtv_img

        # if key is available update value
        if key in idx_val_dict:
            value = idx_val_dict[key]
            value[sub_idx] = v
            idx_val_dict[key] = value

        # if key is not available insert value
        else:
            value = [0, 0, 0]
            value[sub_idx] = v
            idx_val_dict[key] = value
        cnt += 1

    # 2.multiply coefficients and provide final score for each index
    idx_score_dict = {}
    for key, val in idx_val_dict.items():
        val = idx_val_dict[key]
        score = coefs[0] * val[0] + coefs[1] * val[1] + coefs[2] * val[2]
        idx_score_dict[key] = score

    # 3.find top 12 scores
    idx_score_dict_copy = idx_score_dict.copy()
    final_idxes = []
    max_score = -1
    max_idx = -1

    for _ in range(num_rtv_img):
        for idx, score in idx_score_dict_copy.items():
            if score > max_score:
                max_score = score
                max_idx = idx

        max_score = -1
        final_idxes.append(max_idx)
        idx_score_dict_copy.pop(max_idx)

    # 4.return image files
    final_folder_index = []
    final_index = []
    for ffi, fi in final_idxes:
        final_folder_index.append(ffi)
        final_index.append(fi)

    # 5.fill arrays with all scores of top 12 image for user
    each_score, total_score = [], []
    for key in final_idxes:
        each_score.append(idx_val_dict[key])
        total_score.append(idx_score_dict[key])

    images_data = search(final_folder_index, final_index)
    return images_data, each_score, total_score


def face_search(search_mode):
    if len(search_mode) == 1:
        # 0 model
        if search_mode == [0]:
            print("error no model selected ")
            return None

        # 1 model
        if search_mode == [1]:
            search_thread = SearchThread('1', 'vgg')
            search_thread.start()
            image_data_f = search_thread.join()
            return image_data_f

        if search_mode == [2]:
            search_thread = SearchThread('1', 'arcface')
            search_thread.start()
            image_data_a = search_thread.join()
            return image_data_a

        if search_mode == [4]:
            search_thread = SearchThread('1', 'facenet')
            search_thread.start()
            image_data_v = search_thread.join()
            return image_data_v

        # 2 model
        if search_mode == [3]:
            search_thread_0 = SearchThread('1', 'arcface')
            search_thread_1 = SearchThread('2', 'vgg')

            search_thread_0.start()
            search_thread_1.start()

            images_data_a = search_thread_0.join()
            images_data_v = search_thread_1.join()

            images_data = [images_data_a, images_data_v]
            return images_data

        if search_mode == [5]:
            search_thread_0 = SearchThread('1', 'facenet')
            search_thread_1 = SearchThread('2', 'vgg')

            search_thread_0.start()
            search_thread_1.start()

            images_data_f = search_thread_0.join()
            images_data_v = search_thread_1.join()

            images_data = [images_data_f, images_data_v]
            return images_data

        if search_mode == [6]:
            search_thread_0 = SearchThread('1', 'facenet')
            search_thread_1 = SearchThread('2', 'arcface')

            search_thread_0.start()
            search_thread_1.start()

            images_data_f = search_thread_0.join()
            images_data_a = search_thread_1.join()

            images_data = [images_data_f, images_data_a]
            return images_data

        # 3 model
        if search_mode == [7]:
            search_thread_0 = SearchThread('1', 'facenet')
            search_thread_1 = SearchThread('2', 'arcface')
            search_thread_2 = SearchThread('3', 'vgg')

            search_thread_0.start()
            search_thread_1.start()
            search_thread_2.start()

            images_data_f = search_thread_0.join()
            images_data_a = search_thread_1.join()
            images_data_v = search_thread_2.join()

            images_data = [images_data_f, images_data_a, images_data_v]
            return images_data

    if len(search_mode) == 3:
        search_thread_0 = SearchThread2('1', 'facenet')
        search_thread_1 = SearchThread2('2', 'arcface')
        search_thread_2 = SearchThread2('3', 'vgg')

        search_thread_0.start()
        search_thread_1.start()
        search_thread_2.start()

        index_value_f = search_thread_0.join()
        index_value_a = search_thread_1.join()
        index_value_v = search_thread_2.join()

        images_data, each_score, total_score = combination_search(index_value_f, index_value_a, index_value_v, search_mode)
        return images_data, each_score, total_score


if __name__ == "__main__":
    load_models_and_data()
    webServer = HTTPServer((host_name, server_port), Server)
    print("Server started https://%s:%s" % (host_name, server_port))
    webServer.serve_forever()
