#!/usr/bin/python3
import time
import sys
import os
import cv2
import getpass
import pickle
from pathlib import Path
import face_recognition
from os import walk


param_length = len(sys.argv)


def log_error(msg, _quit = True):
    print("\n")
    print("-- PARAMETER ERROR --\n"*2)
    print(" %s " % msg)
    print("\n")
    print("-- PARAMETER ERROR --\n"*2)
    print("\n")
    if _quit:
        quit()
    else:
        return False


def log_debug(msg):
    print("\n------- %s -------\n" % msg)


def log_warning(msg):
    print("\n WARNING ------- %s -------" % msg)


def dir_exists(path_str):
    path = Path(path_str)
    if path.exists():
        return True
    return False


def get_timestamp():
    return int(time.time() * 1000)


def delete_file(file_path):
    os.remove(file_path)
    if file_exists(file_path):
        raise Exception('unable to delete file: %s' % file_path)


def file_exists(file_name):
    try:
        with open(file_name) as f:
            return file_name
    except OSError as e:
        return False


def file_exists_and_not_empty(file_name):
    if file_exists(file_name) and os.stat(file_name).st_size > 0:
        return True
    return False


def read_images_in_dir(path_to_read):
    dir_name, subdir_name, file_names = next(walk(path_to_read))
    images = [item for item in file_names if '.jpeg' in item[-5:] or '.jpg' in item[-4:] or 'png' in item[-4:]]
    return images, dir_name


def write_to_pickle(known_face_encodings, known_face_metadata, data_file):
    with open(data_file, mode='wb') as f:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, f)


def read_pickle(pickle_file, exception_if_fail=True):
    try:
        with open(pickle_file, 'rb') as f:
            known_face_encodings, known_face_metadata = pickle.load(f)
            return known_face_encodings, known_face_metadata
    except OSError as e:
        if exception_if_fail:
            log_error("Unable to open pickle_file: {}, original exception {}".format(pickle_file, str(e)))
        else:
            return [], []


def new_face_metadata(face_image, name=None, camera_id=None, confidence=None, print_name=False, image_group=None):
    """
    Add a new person to our list of known faces
    """
    # if image_group and not image_group in com.IMAGE_GROUPS:
    # com.log_error("Image type most be one of the followings or None: {}".format(com.IMAGE_GROUPS))

    today_now = get_timestamp()

    if name is None:
        name = camera_id + '_' + str(today_now)
    else:
        if print_name:
            print('Saving face: {} in group: {}'.format(name, image_group))

    return {
        'name': name,
        'face_id': 0,
        'camera_id': camera_id,
        'first_seen': today_now,
        'first_seen_this_interaction': today_now,
        'image': False,
        'image_group': image_group,
        'confidence': confidence,
        'last_seen': today_now,
        'seen_count': 1,
        'seen_frames': 1
    }


def encode_face_image(face_obj, face_name, camera_id, confidence, print_name, default_sample=0,
                      model=None, image_group=None):
    # covert the array into cv2 default color format
    # THIS ALREADY DONE IN CROP
    # rgb_frame = cv2.cvtColor(face_obj, cv2.COLOR_RGB2BGR)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = face_obj[:, :, ::-1]

    # try to get the location of the face if there is one
    # face_location = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=2, model='cnn')
    if model is not None:
        #print("face_location with: {}/{}".format(default_sample, model))
        # face_location = face_recognition.face_locations(rgb_small_frame, default_sample, model='cnn')
        face_location = face_recognition.face_locations(rgb_small_frame, 1, model='cnn')
        # face_location = face_recognition.face_locations(rgb_small_frame)
    else:
        print("face_location with default model: {}".format(model))
        face_location = face_recognition.face_locations(rgb_small_frame)

    # if got a face, loads the image, else ignores it
    if face_location:
        # Grab the image of the face from the current frame of video
        top, right, bottom, left = face_location[0]
        face_image = rgb_small_frame[top:bottom, left:right]
        face_image = cv2.resize(face_image, (150, 150))
        encoding = face_recognition.face_encodings(face_image)

        # if encoding empty we assume the image was already treated
        if len(encoding) == 0:
            encoding = face_recognition.face_encodings(rgb_small_frame)

        if encoding:
            face_metadata_dict = new_face_metadata(face_obj, face_name, camera_id, confidence, print_name, image_group)
            return encoding[0], face_metadata_dict

    return None, None


def encode_and_update_face_image(face_obj, name, face_encodings, face_metadata, default_sample=0, model=None,
                                 image_group=None):
    new_encoding, new_metadata = encode_face_image(face_obj, name, None, None, True, default_sample, model, image_group)

    if new_encoding is not None:
        face_encodings.append(new_encoding)
        face_metadata.append(new_metadata)
        # if we are able to encode and get the metadata we return a third value indicating success with "True" value
        return face_encodings, face_metadata, True

    # if we are failed to encode and get the metadata we return a third value indicating failure with "False" value
    return face_encodings, face_metadata, False


def encode_known_faces_from_images_in_dir(image_path, output_file, image_group=None, append=False):
    '''
    Esta funccion codifica los rostros encotrados en las imagenes presentes en el diretorio especificado
    '''
    if dir_exists(image_path) is False:
        log_error("Directory '{}' does not exist".format(image_path))

    files, root = read_images_in_dir(image_path)

    known_face_encodings = []
    known_face_metadata = []
    if append and file_exists(output_file):
        known_face_encodings, known_face_metadata = read_pickle(output_file)

    write_to_file_counter = 0
    model = None
    for file_name in files:
        # load the image into face_recognition library
        source_info = {}
        face_obj = face_recognition.load_image_file(root + '/' + file_name)
        name = os.path.splitext(file_name)[0]
        known_face_encodings, known_face_metadata, encoding_result = encode_and_update_face_image(
            face_obj, name, known_face_encodings, known_face_metadata, 0, model, image_group)

        if encoding_result is False:
            model = "cnn"
            known_face_encodings, known_face_metadata, encoding_result = encode_and_update_face_image(
                face_obj, name, known_face_encodings, known_face_metadata, 1, model, image_group)

        if encoding_result:
            write_to_file_counter += 1
            write_to_pickle(known_face_encodings, known_face_metadata, output_file)
        else:
            log_warning("Unable to process a face from image: {}".format(name))

    return write_to_file_counter

msg = ' Usage: ' + sys.argv[0] + ' -new    ABSOLUT_PATH_TO_DIRECTORY_OF_FACES\n'
msg = msg+'  Usage: ' + sys.argv[0] + ' -update ABSOLUT_PATH_TO_DBS'

del sys.argv[0]

if sys.argv[0] != '-new' and sys.argv[0] != '-update':
    log_error(msg)

action = sys.argv[0]
del sys.argv[0]

home_path = os.getenv('HOME')
dbs_dir = home_path + '/dbs'
faces_dir = home_path + '/faces'
config_dir = home_path + '/config'
user_name = getpass.getuser()
base_path_length = len(home_path+'/')
list_type = ''

# valida que el path del directorio este dentro de faces/whitelist or blacklist
if action == '-new':
    directory_path = str(sys.argv[0]).strip()
    relative_path = directory_path[base_path_length:]

    #validating if the directory path contains whitelist or blacklist
    if not (relative_path[:15] == 'faces/whitelist' or relative_path[:15] == 'faces/blacklist'):
        log_error("face's directory "+relative_path+" must be inside faces and inside whitelist or blacklist")

    #get list type
    list_type = 'black'
    if relative_path[6:11] == 'white':
        list_type = 'white'


    if directory_path[0:1] != '/':
        log_error("Directory path should be a full path not a relative path")

    if directory_path[-1:] != '/':
        directory_path = directory_path + '/'

    directory_name = ''
    for item in str(sys.argv[0]).strip().rstrip().split('/'):
        if len(item) > 0:
            directory_name = item

    if param_length != 3:
        log_error(msg)

    # verificar que todos los elementos dentro del directorio sean archivos
    file_count = 0
    for item in os.listdir(directory_path):
        is_file = os.path.isdir(item)
        if is_file is True:
            log_error("There should only be files under {}:".format(directory_path))
        # verificar que el archivo no este vacio
        if file_exists_and_not_empty(directory_path+item):
            file_count += 1

    if file_count == 0:
        log_error("directory {} cannot be empty".format(directory_path))

    #validar que el directorio target exista
    target_dir = dbs_dir+"/"+list_type+'list'
    if not dir_exists(target_dir):
        log_error('Target directory "{}" does not exist'.format(target_dir))

    # verificar si la ruta destino existe, si existe eliminarla para que sea recreada
    db_file_full_path = dbs_dir+"/"+list_type+"list/"+directory_name+'.dat'
    if file_exists(db_file_full_path):
        delete_file(db_file_full_path)

    # reading files and making their numerical representation
    faces_dir = home_path + '/' + relative_path
    result = encode_known_faces_from_images_in_dir(faces_dir, db_file_full_path, list_type)

    if result == 0:
        log_error('Unable to create any numerical face representation from the files in: {}'.format(faces_dir))
    log_debug('Saving -{}- numerical faces and metadatas in: {}'.format(result, db_file_full_path))
elif action == '-update':
    dbs_paths = set()
    for item in sys.argv:
        if not file_exists_and_not_empty(item):
            log_error("file {} does not exist or is empty".format(item))

        is_dir = os.path.isdir(item)
        if is_dir is True:
            log_error("There should only be db files: {}".format(item))

        dbs_paths.add(item)

        trigger_file = config_dir+'/dbs_to_update.txt'
        log_debug("writing in {}".format(trigger_file))
        with open(trigger_file, 'w') as f:
            for line in dbs_paths:
                f.write(line + '\n')
