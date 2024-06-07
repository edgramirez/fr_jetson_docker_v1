#!/usr/bin/env python3

################################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

import sys
import gi
import configparser
from os import path
sys.path.append('../')
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
from ctypes import *
import time
import math
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
import numpy as np
import pyds
import cv2
import os
import os.path
from datetime import datetime, timedelta
import threading
from random import randrange
import random
import face_recognition

import lib.biblioteca as biblio
import lib.server as srv
import lib.common as com
import lib.validate as validate
import lib.service_variables as sv
import lib.json_methods as jm

# 6-nov-2021
# Validar este arreglo saved_count para que se utiliza...
global PGIE_CLASS_ID_FACE
PGIE_CLASS_ID_FACE = 0

global PGIE_CLASS_ID_MAKE
PGIE_CLASS_ID_MAKE = 2

MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_FACE = 0

# 6-Nov-2021
# Variables not necessary for this model
#
# PGIE_CLASS_ID_PLATE = 1
# PGIE_CLASS_ID_MAKE = 2
# PGIE_CLASS_ID_MODEL = 3

MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080
GST_CAPS_FEATURES_NVMM = "memory:NVMM"

# 6-nov-2021
# Arreglo no utilizado en este modelo
# pgie_classes_str= ["face", "Placa", "Marca","Modelo"]

CURRENT_DIR = os.getcwd()

# 0 cualquier cosa es reconocida como rostro, 1 es la maxima confidencia de que ese objeto es un rostro
DEEPSTREAM_FACE_RECOGNITION_MINIMUM_CONFIDENCE = .86
# bytes, permite elegir solo frames de un tamaño adecuado
FRAME_SIZE = 1024*20

fps_streams = {}
frame_count = {}
saved_count = {}
call_order_of_keys = []


global BASE_DIRECTORY
global BASE_INPUT_DB_DIRECTORY
global DEMO
global WHITELIST_DB_NAME
global WHITELIST_DB_DIRECTORY
global BLACKLIST_DB_NAME
global BLACKLIST_DB_DIRECTORY


#################  Model and service functions  #################

def set_action_common_variables(service_name):
    global BASE_DIRECTORY, BASE_INPUT_DB_DIRECTORY, DEMO, WHITELIST_DB_NAME, \
        WHITELIST_DB_DIRECTORY, BLACKLIST_DB_NAME, BLACKLIST_DB_DIRECTORY

    if service_name == "whiteList" or service_name == "blackList":
        BASE_DIRECTORY = os.path.expanduser('~') + "/faceRecognition"
        BASE_INPUT_DB_DIRECTORY = BASE_DIRECTORY+"/data_input"
        DEMO = False

    if service_name == "whiteList":
        try:
            WHITELIST_DB_NAME = "WhiteList.dat"
            WHITELIST_DB_DIRECTORY = BASE_INPUT_DB_DIRECTORY + '/whitelist_db'
        except AttributeError as e:
            com.log_error("whitelist service parameters are not defined in definition.py file")
    elif service_name == "blackList":
        try:
            BLACKLIST_DB_NAME = "BlackList.dat"
            BLACKLIST_DB_DIRECTORY = BASE_INPUT_DB_DIRECTORY + '/blacklist_db'
        except AttributeError as e:
            com.log_error("blackList service parameters are not defined in definition.py file")


def set_header(token_file=None):
    if token_file is None:
        token_file =  "/home/kayros/.token"

    if com.file_exists(token_file):
        token_handler = com.open_file(token_file, 'r+')
        sv.header = {'Content-type': 'application/json', 'X-KAIROS-TOKEN': token_handler.read().split('\n')[0]}
        com.log_debug('Header correctly set')
        return sv.header
    com.log_error('Unable to read token')


def config_blacklist(srv_camera_id):
    set_blacklist_url(srv_camera_id)
    set_blacklist_db(srv_camera_id)
    set_delta(srv_camera_id, 10)


def config_whitelist(srv_camera_id):
    set_whitelist_url(srv_camera_id)
    set_whitelist_db(srv_camera_id)
    set_delta(srv_camera_id, 10)


def config_age_and_gender(srv_camera_id):
    if len(sv.gender_age_dict) == 0:
        set_age_gender_config()

    set_age_and_gender_url(srv_camera_id)


def set_blacklist_db(camera_service_id):
    global BLACKLIST_DB_NAME, BLACKLIST_DB_DIRECTORY

    service_name = 'blackList'
    camera_mac = camera_service_id.split('_')[1]

    # si la db esa definida usar esa si no usar la default
    for item in sv.scfg[camera_mac]['services']:
        if camera_service_id in item:
            if 'dbName' in item[camera_service_id][service_name]:
                search_db_name = BLACKLIST_DB_DIRECTORY + '/' + item[camera_service_id][service_name]['dbName']
            else:
                search_db_name = BLACKLIST_DB_DIRECTORY + '/' + BLACKLIST_DB_NAME

            if com.file_exists_and_not_empty(search_db_name):
                # Guarda el nombre de la db de whiteList
                set_known_faces_db_name(camera_service_id, search_db_name)
                # Extrae de la db de blackList
                # Estos valores son fijos y solo se leen una sola vez desde el archivo de base de datos o serializado
                sv.blacklist_encodings, sv.blacklist_metas = com.read_pickle(get_known_faces_db_name(camera_service_id),
                                                                             False)
                # Carga los datos en sus dictionarios globales correspondientes
                return True

    com.log_error('Unable to setup blacklist input/output service variables - blacklist search db "{}" does not exists'.
                  format(search_db_name))


def set_whitelist_db(camera_service_id):
    global WHITELIST_DB_NAME, WHITELIST_DB_DIRECTORY

    service_name = 'whiteList'
    camera_mac = camera_service_id.split('_')[1]

    # si la db esa definida usar esa si no usar la default
    for item in sv.scfg[camera_mac]['services']:
        if camera_service_id in item:
            if 'dbName' in item[camera_service_id][service_name]:
                search_db_name = WHITELIST_DB_DIRECTORY + '/' + item[camera_service_id][service_name]['dbName']
            else:
                search_db_name = WHITELIST_DB_DIRECTORY + '/' + WHITELIST_DB_NAME

            # check the DB file exists and is not empty
            if com.file_exists_and_not_empty(search_db_name):
                # Guarda el nombre de la db de whiteList
                set_known_faces_db_name(camera_service_id, search_db_name)
                # Extrae de la db de blackList
                sv.whitelist_encodings, sv.whitelist_metas = com.read_pickle(get_known_faces_db_name(camera_service_id),
                                                                             False)
                # Carga los datos en sus dictionarios globales correspondientes
                return True

    com.log_error('Unable to setup whitelist input/output service variables - whitelist search db "{}" does not exists'.
                  format(search_db_name))


def set_age_gender_config():
    baseDir = 'configs/'
    faceProto = baseDir + "opencv_face_detector.pbtxt"
    faceModel = baseDir + "opencv_face_detector_uint8.pb"
    ageProto = baseDir + "age_deploy.prototxt"
    ageModel = baseDir + "age_net.caffemodel"
    genderProto = baseDir + "gender_deploy.prototxt"
    genderModel = baseDir + "gender_net.caffemodel"

    sv.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    sv.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    sv.genderList = ['Male', 'Female']

    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    sv.gender_age_dict.update({"ageNet": ageNet})
    sv.gender_age_dict.update({"genderNet": genderNet})


def set_blacklist_url(camera_service_id):
    if camera_service_id not in sv.urls:
        com.log_debug("Setting up endpoint for service is: {}".format("Blacklist - "+camera_service_id))
        for item in sv.scfg[camera_service_id[7:24]]['services']:
            if camera_service_id in item:
                sv.urls.update({camera_service_id: item[camera_service_id]['blackList']['endpoint']})


def set_whitelist_url(camera_service_id):
    if camera_service_id not in sv.urls:
        com.log_debug("Setting up endpoint for service is: {}".format("Whitelist - "+camera_service_id))
        for item in sv.scfg[camera_service_id[7:24]]['services']:
            if camera_service_id in item:
                sv.urls.update({camera_service_id: item[camera_service_id]['whiteList']['endpoint']})


def set_age_and_gender_url(camera_service_id):
    if camera_service_id not in sv.urls:
        com.log_debug("Setting up endpoint for service is: {}".format("Age and Gender - "+camera_service_id))
        for item in sv.scfg[camera_service_id[7:24]]['services']:
            if camera_service_id in item:
                sv.urls.update({camera_service_id: item[camera_service_id]['ageAndGender']['endpoint']})


def get_service_url(camera_service_id):
    global WHITELIST_DB_NAME, WHITELIST_DB_DIRECTORY, BLACKLIST_DB_NAME, BLACKLIST_DB_DIRECTORY

    if camera_service_id in sv.urls:
        return sv.urls[camera_service_id]

    com.log_error("Unable to get service endpoint for id: {} / url list = {}".format(camera_service_id, sv.urls))


def set_action_helper(srv_camera_id, service_name):
    if service_name in com.SERVICE_DEFINITION[com.SERVICES[service_name]]:
        config_age_and_gender(srv_camera_id)
        return True
    else:
        com.log_error("Service '"+service_name+"' not defined")


def set_action(srv_camera_id, service_name):
    """
    Esta function transfiere la configuration de los parametros hacia los servicios activos por cada camara
    """
    execute_actions = False
    if service_name in com.SERVICES:
        sv.action.update({srv_camera_id: service_name})
        com.log_debug('Set "{}" variables for service id: {}'.format(service_name, srv_camera_id))
        if service_name == 'find':
            if service_name == com.SERVICE_DEFINITION[com.SERVICES[service_name]]:
                com.log_error("Service de find no definido aun")
            else:
                com.log_error("Service '"+service_name+"' not defined")
        elif service_name == 'blackList':
            if service_name in com.SERVICE_DEFINITION[com.SERVICES[service_name]] and BLACKLIST_DB_NAME:
                config_blacklist(srv_camera_id)
                execute_actions = True
            else:
                com.log_error("Service '"+service_name+"' not defined")
        elif service_name == 'whiteList':
            if service_name in com.SERVICE_DEFINITION[com.SERVICES[service_name]] and WHITELIST_DB_NAME:
                config_whitelist(srv_camera_id)
                execute_actions = True
            else:
                com.log_error("Service '"+service_name+"' not defined")
        elif service_name == 'ageAndGender':
            if service_name in com.SERVICE_DEFINITION[com.SERVICES[service_name]]:
                config_age_and_gender(srv_camera_id)
                execute_actions = True
            else:
                com.log_error("Service '"+service_name+"' not defined")
        elif service_name == 'aforo':
            execute_actions = set_action_helper(srv_camera_id, service_name)

        if execute_actions:
            return True

    com.log_error('Unable to set up value: {}, must be one of this: {}'.format(service_name, com.SERVICES))


#####################  Non specific to Model or Service functions  #####################

def set_config(scfg):
    number_sources = 0
    for camera_mac in scfg:
        call_order_of_keys.append(camera_mac)
        number_sources += 1
        for service_id in scfg[camera_mac]:
            if service_id == "source":
                continue
            for item in scfg[camera_mac][service_id]:
                for service_id_inner in item:
                    for service_name in item[service_id_inner]:
                        set_action_common_variables(service_name)
                        set_camera_mac_address(service_id_inner, camera_mac)
                        set_action(service_id_inner, service_name)
    return number_sources


def set_camera_mac_address(camera_service_id, camera_mac_address):
    if camera_service_id not in sv.service_camera_mac_address:
        sv.service_camera_mac_address.update({camera_service_id: camera_mac_address})
    else:
        sv.service_camera_mac_address[camera_service_id] = camera_mac_address


def get_camera_mac_address(camera_service_id):
    if camera_service_id not in sv.service_camera_mac_address:
        com.log_error("Unable to found mac address value for service id: {}".format(camera_service_id))
    else:
        return sv.service_camera_mac_address[camera_service_id]


def get_not_applicable_id(camera_service_id, abort = True):
    if camera_service_id in sv.not_applicable_id:
        return sv.not_applicable_id[camera_service_id]

    if abort:
        com.log_error('get_not_applicable_id() - No value found for camera_service_id: {}'.format(camera_service_id))
    else:
        return []


def update_not_applicable_id(camera_service_id, new_value, best_index = None):
    if best_index is not None:
        sv.not_applicable_id[camera_service_id][best_index] = new_value
    else:
        # if key is not already in the global dictionary 
        if camera_service_id not in sv.not_applicable_id:
            sv.not_applicable_id.update({camera_service_id: {new_value}})
        else:
            sv.not_applicable_id[camera_service_id].add(new_value)


def get_camera_service_id(camera_service_id):
    global call_order_of_keys
    return call_order_of_keys[camera_service_id]


def set_delta(camera_service_id, secs):
    if not isinstance(secs, int):
        com.log_error('delta time value must be integer')

    if camera_service_id in sv.delta_time:
        sv.delta_time[camera_service_id] = secs
    else:
        sv.delta_time.update({camera_service_id: secs})


##############################################


def set_known_faces_db_name(camera_service_id, value):
    sv.input_file.update({camera_service_id: value})


def set_metadata(camera_service_id, metadata):
    sv.known_face_metadata.update({camera_service_id: metadata})


def set_encoding(camera_service_id, encodings):
    sv.known_face_encodings.update({camera_service_id: encodings})


def set_known_faces_db(camera_service_id, encodings, metadata):
    set_encoding(camera_service_id, encodings)
    set_metadata(camera_service_id, metadata)


def get_face_detection_url(camera_service_id):
    if camera_service_id in sv.face_detection_url:
        return sv.face_detection_url[camera_service_id]

    com.log_error('get_face_detection_url() - No value found for camera_service_id: {}'.format(camera_service_id))


def get_known_faces_db_name(camera_service_id):
    if camera_service_id in sv.input_file:
        return sv.input_file[camera_service_id]

    com.log_error('get_known_faces_db_name() - No value found for camera_service_id: {}'.format(camera_service_id))


def get_delta(camera_service_id):
    global delta_time
    return delta_time[camera_service_id]


def get_similarity(camera_service_id):
    return 0.7


def add_new_face_metadata(camera_service_id, face_image, confidence, difference, obj_id):
    """
    Add a new person to our list of known faces - Recurrencia
    """
    global known_face_metadata
    today_now = datetime.now()
    name = str(obj_id) + '_'+ camera_service_id + '_' + str(today_now)
    face_id = camera_service_id + '_' + str(com.get_timestamp())

    known_face_metadata[camera_service_id].append({
        'name': name,
        'face_id': face_id,
        'first_seen': today_now,
        'first_seen_this_interaction': today_now,
        'image': face_image,
        'confidence': [confidence],
        'difference': [difference],
        'last_seen': today_now,
        'seen_count': 1,
        'seen_frames': 1
    })

    # Json data format
    data = {
            'name': name,
            'faceId': face_id,
            'cameraID': camera_service_id,
            'faceType': 0,
            '#initialDate': today_now,
            'numberOfDetections': 1,
            'image': face_image
            }

    # background_result = threading.Thread(target=send_json, args=(data, 'POST', get_face_detection_url(),))
    # background_result.start()

    return known_face_metadata


def get_gender_and_age(image):
    blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), sv.MODEL_MEAN_VALUES, swapRB=False)

    sv.gender_age_dict["genderNet"].setInput(blob)
    gender_preds = sv.gender_age_dict["genderNet"].forward()
    # print(gender_preds[0].argmax())
    gender = sv.genderList[gender_preds[0].argmax()]
    print(f'Gender: {gender}')

    sv.gender_age_dict['ageNet'].setInput(blob)
    age_preds = sv.gender_age_dict["ageNet"].forward()
    age = sv.ageList[age_preds[0].argmax()]
    print(f'Age: {age[1:-1]} years')


def whitelist_process(camera_service_id, image_encoding, image_meta, obj_id):
    not_applicable_id = get_not_applicable_id(camera_service_id, abort=False)

    if obj_id in not_applicable_id:
        return False

    metadata, best_index, difference = biblio.lookup_known_face(image_encoding, sv.whitelist_encodings,
                                                                sv.whitelist_metas)

    # WhiteList reporta cuando no hay coincidencias
    if best_index is None:
        # Actualizando la lista de ids de los rostros no coincidentes
        update_not_applicable_id(camera_service_id, obj_id)

        # create the payload
        data = {
                "clientId": "Estadio Olmeca. SA de CV",
                "cameraId": get_camera_mac_address(camera_service_id),
                "listType": 'whiteList',
                "matchedId": None,
                "matchedName": None 
                }
        # jm.send_json(sv.header, data, 'POST', get_service_url(camera_service_id))
        background_result = threading.Thread(target=jm.send_json, args=(sv.header, data, 'POST',
                                                                        get_service_url(camera_service_id),))
        background_result.start()
        print('Rostro con id: {}, streaming {}, no esta en la White list. Reportando incidente: {}'.
              format(obj_id, camera_service_id, data))
        return True

    return False


def blacklist_process(camera_service_id, image_encoding, image_meta, obj_id):
    not_applicable_id = get_not_applicable_id(camera_service_id, abort=False)

    if obj_id in not_applicable_id:
        return False

    metadata, best_index, difference = biblio.lookup_known_face(image_encoding, sv.blacklist_encodings, sv.blacklist_metas)

    # BlackList reporta cuando hay coincidencias
    if best_index is not None:
        # Actualizando la lista de ids de los rostros no coincidentes
        update_not_applicable_id(camera_service_id, obj_id)
        return False

    # create the payload
    data = {
            "clientId": "Estadio Olmeca. SA de CV",
            "cameraId": get_camera_mac_address(camera_service_id),
            "listType": 'blackList',
            "matchedId": None,
            "matchedName": None 
            }
    # jm.send_json(sv.header, data, 'POST', get_service_url(camera_service_id))
    background_result = threading.Thread(target=jm.send_json, args=(sv.header, data, 'POST',
                                                                    get_service_url(camera_service_id),))
    background_result.start()
    print('Rostro con id: {}, streaming {}, esta en la Black list. Reportando incidente: {}'.
          format(obj_id, camera_service_id, data))
    return True


def tiler_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0		# Faltaba del archivo Original deepstream_imagedata_multistream
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        com.log_debug("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    # contador = 0
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num

        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        is_first_obj = True
        save_image = False
        obj_counter = {
        PGIE_CLASS_ID_FACE:0
        }

        #6-Nov-2021
        # Se sacan variables no utilizadas para este modelo
        #PGIE_CLASS_ID_PLATE:0,
        #PGIE_CLASS_ID_MAKE:0,
        #PGIE_CLASS_ID_MODEL:0
        
        # Este bloque se movio aquí , estaba afuera del while más externo y provocaba 
        # que el servicio no empatara con el stream correspondiente

        camera_id = get_camera_service_id(pyds.NvDsFrameMeta.cast(l_frame.data).pad_index)
        config = sv.scfg[camera_id]
        
        id_set = set()

        while l_obj is not None:
            # contador += 1

            try: 
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            # Periodically check for objects with borderline confidence value that may be false positive detections.
            # If such detections are found, register the frame with bboxes and confidence value.

            # Save the annotated frame to file.
            if obj_meta.class_id == 0 and obj_meta.confidence > DEEPSTREAM_FACE_RECOGNITION_MINIMUM_CONFIDENCE:
                # Getting Image data using nvbufsurface
                # the input should be address of buffer and batch_id
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                frame_image = crop_and_get_faces_locations(n_frame, obj_meta, obj_meta.confidence)

                # 6- nov-2021   -   El tamaño del frame mayor que 0 evita recuadros de rostros vacios
                # se se aumenta el tamaño se estará selecionando frames mas grandes y visibles
                if frame_image.size > FRAME_SIZE: 
                    name = None
                    # print("Guardando ",obj_meta.object_id)
                    # 11 de Enero 2022 ERM
                    # El object ID es por streaming, d/
                    
                    id_set.add(obj_meta.object_id)
                    set_active_ids_x_camera(camera_id, obj_meta.object_id, frame_number)
                    previous_treated_elements = get_treated_face_ids(camera_id)
                    
                    if obj_meta.object_id not in previous_treated_elements:
                        print("Trying to process image ................", obj_meta.object_id, obj_meta.confidence)
                        # tratar de generar un codificado de la imagen
                        image_encoding, image_meta = biblio.encoding_image_from_source(camera_id, frame_image,
                                                                                       obj_meta.confidence)
                        
                        # si se logro codificar (es decir, no es vacio "[]"), entonces se registra como rostro
                        # analizado y solo se vuelve a analizar si la confianza es mayor
                        if image_meta:
                            print("Image successfully processed .........", obj_meta.object_id, obj_meta.confidence)
                            add_to_treated_face_ids(camera_id, obj_meta.object_id)

                            for camera_service_id, service_name in sv.action.items():

                                if service_name in com.SERVICE_DEFINITION[com.SERVICES['whiteList']]:
                                    whitelist_process(camera_service_id, image_encoding, image_meta, obj_meta.object_id)
                                elif service_name in com.SERVICE_DEFINITION[com.SERVICES['blackList']]:
                                    blacklist_process(camera_service_id, image_encoding, image_meta, obj_meta.object_id)
                                elif service_name in com.SERVICE_DEFINITION[com.SERVICES['ageAndGender']]:
                                    get_gender_and_age(frame_image)
            try: 
                l_obj = l_obj.next
            except StopIteration:
                break

        # Get frame rate through this probe
        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()

        # si no leemos ningun rostro no ejecutamos el bloque
        if frame_number % 23 == 0:
            current_active_ids_x_camera = get_active_ids_x_camera(camera_id)
            if current_active_ids_x_camera is not None:
                if len(id_set) == 0:
                    for value in current_active_ids_x_camera:
                        set_inactive_ids_x_camera(camera_id, value)
                else:
                    for value in id_set:
                        set_inactive_ids_x_camera(camera_id, value)

                inactive_items = get_inactive_ids_x_camera(camera_id)

                elements_to_delete = [key for key in inactive_items if inactive_items[key] > 2]

                for item in elements_to_delete:
                    remove_inactive_ids_x_camera(camera_id, item)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def set_active_ids_x_camera(camera_id, id_from_source, frame_number):
    if camera_id not in sv.scfg:
        com.log_error("wrong camera_id: {} - This id is not part of the configuration\n".format(camera_id))

    if camera_id not in sv.active_ids_per_camera:
        sv.active_ids_per_camera.update({camera_id: {id_from_source}})
    else:
        sv.active_ids_per_camera[camera_id].add(id_from_source)


def remove_inactive_ids_x_camera(camera_id, item_to_delete):
    if camera_id in sv.active_ids_per_camera:
        sv.active_ids_per_camera[camera_id].remove(item_to_delete)
        del sv.inactive_ids_per_camera[camera_id][item_to_delete]


def get_active_ids_x_camera(camera_id):
    if camera_id in sv.active_ids_per_camera:
        return sv.active_ids_per_camera[camera_id]


def set_inactive_ids_x_camera(camera_id, id_from_source):
    if camera_id not in sv.scfg:
        com.log_error("wrong camera_id: {} - This id is not part of the configuration\n".format(camera_id))

    if camera_id not in sv.inactive_ids_per_camera:
        sv.inactive_ids_per_camera.update({camera_id: {id_from_source: 1}})
    else:
        if id_from_source not in sv.inactive_ids_per_camera[camera_id]:
            sv.inactive_ids_per_camera[camera_id].update({id_from_source: 1})
        else:
            sv.inactive_ids_per_camera[camera_id][id_from_source] += 1


def get_inactive_ids_x_camera(camera_id):
    if camera_id in sv.inactive_ids_per_camera:
        return sv.inactive_ids_per_camera[camera_id]

    return {}


def write_to_db(face_metadata, face_encodings, output_db_name):
    biblio.write_to_pickle(face_encodings, face_metadata, output_db_name)


def add_to_treated_face_ids(camera_id, id_already_treated):
    if camera_id in sv.treated_ids:
        sv.treated_ids[camera_id].add(id_already_treated)
    else:
        sv.treated_ids.update({camera_id: {id_already_treated}})


def get_treated_face_ids(camera_id):
    if camera_id in sv.treated_ids:
        return sv.treated_ids[camera_id]

    return {}


def main():
    global call_order_of_keys

    sv.header = set_header()
    sv.scfg = srv.get_server_info(sv.header)
    com.log_debug("Final configuration: {}".format(sv.scfg))

    number_sources = set_config(sv.scfg)
    is_live = False

    com.log_debug("Number of sources :{}".format(number_sources))
    print("\n------ Fps_streams: ------ \n", fps_streams)

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    com.log_debug("Creating Pipeline")
    pipeline = Gst.Pipeline()

    if not pipeline:
        com.log_error(" Unable to create Pipeline")
    com.log_debug("Creating streamux")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        com.log_error(" Unable to create NvStreamMux")

    pipeline.add(streammux)

    # Load pipeline based on the configuration
    i = 0
    for ordered_key in call_order_of_keys:
        fps_streams["stream{0}".format(i)]=GETFPS(i)    
        frame_count["stream_"+str(i)] = 0
        saved_count["stream_"+str(i)] = 0

        # Defining only 1 active source per camera 
        for service_id in sv.scfg[ordered_key]:
            uri_name = sv.scfg[ordered_key]['source']

            com.log_debug("Creating source_bin: {}.- {} with uri_name: {}\n".format(i, ordered_key, uri_name))
            
            if uri_name.find("rtsp://") == 0 :
                is_live = True
            
            source_bin = create_source_bin(i, uri_name)
            if not source_bin:
                com.log_error("Unable to create source bin")
            
            pipeline.add(source_bin)
            padname = "sink_%u" % i
            
            sinkpad = streammux.get_request_pad(padname)
            if not sinkpad:
                com.log_error("Unable to create sink pad bin")
            srcpad = source_bin.get_static_pad("src")
            if not srcpad:
                com.log_error("Unable to create src pad bin")
            srcpad.link(sinkpad)
            i += 1
            break

    """
    -------- Configuration loaded --------
    """

    com.log_debug("Creating Pgie")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        com.log_error(" Unable to create pgie")
    
    # Creation of tracking to follow up the model face
    # April 21th
    # ERM
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        com.log_error(" Unable to create tracker")
    
    # Add nvvidconv1 and filter1 to convert the frames to RGBA
    # which is easier to work with in Python.

    com.log_debug("Creating nvvidconv1 ")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        com.log_error(" Unable to create nvvidconv1")
    com.log_debug("Creating filter1")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        com.log_error(" Unable to get the caps filter1")
    filter1.set_property("caps", caps1)
    com.log_debug("Creating tiler")

    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        com.log_error(" Unable to create tiler")
    com.log_debug("Creating nvvidconv")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        com.log_error(" Unable to create nvvidconv")
    com.log_debug("Creating nvosd")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        com.log_error(" Unable to create nvosd")
    if(is_aarch64()):
        com.log_debug("Creating transform")
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        if not transform:
            com.log_error(" Unable to create transform")

    com.log_debug("Creating EGLSink")

    # edgar: cambio esta linea para no desplegar video - 
    # 6-nov-2021
    # Reprogramar para que el elemento sink tome el valor nvegldessink (video output) o fakesink (Black hole for data)
    # dependiendo si estamos en modo DEMO o produccion respectivamente

    demo_status = DEMO
    if demo_status == "True":
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    else:
        sink = Gst.ElementFactory.make("fakesink", "fakesink")

    if not sink:
        com.log_error("Unable to create egl sink")

    if is_live:
        com.log_debug("At least one of the sources is live")
        streammux.set_property('live-source', 1)
    
    # Camaras meraki 720p
    # 6-Nov-2021
    # se definen las variables width y height con valores definidos al inicio

    streammux.set_property('width', MUXER_OUTPUT_WIDTH)
    streammux.set_property('height', MUXER_OUTPUT_HEIGHT)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)

    #fin de la definicion

    pgie.set_property('config-file-path', CURRENT_DIR + "/configs/pgie_config_facenet.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        com.log_debug("WARNING: Overriding infer-config batch-size '{}', with number of sources {}".
                      format(pgie_batch_size, number_sources))
        pgie.set_property("batch-size", number_sources)

    # Set properties of tracker
    # April 21th
    # ERM

    config = configparser.ConfigParser()
    config.read('configs/tracker_config.txt')
    config.sections()
    
    for key in config['tracker']:
        if key == 'tracker-width':
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        elif key == 'tracker-height':
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        elif key == 'gpu-id':
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        elif key == 'll-lib-file':
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        elif key == 'll-config-file':
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        elif key == 'enable-batch-process':
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)

    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    sink.set_property("sync", 0) # Sync on the clock
    sink.set_property("qos", 0) # faltaba archivo original deepstream_imagedata_multistream.py Generate Quality-of-Service events upstream

    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        # print("Architecture x86 ")
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)

    com.log_debug("Adding elements to Pipeline")

    # Add tracker in pipeline
    # April 21th
    # ERM

    pipeline.add(pgie)
    pipeline.add(tracker)     # Tracker
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    if is_aarch64():
        pipeline.add(transform)
    pipeline.add(sink)

    com.log_debug("Linking elements in the Pipeline")

    # 6-nov-2021
    # Revision de elementos del pipeline
    # el filtro después del tiler no me hace sentido
    # por el momento se queda como el archivo original deepstream_imagedata_multistream.py

    streammux.link(pgie)
    pgie.link(tracker)        # se añade para tracker
    # pgie.link(nvvidconv1)     se modifica
    tracker.link(nvvidconv1)  # se añade para ligar tracker con los demas elementos
    nvvidconv1.link(filter1)
    filter1.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    if is_aarch64():
        nvosd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(sink)
    '''
    streammux.link(pgie)
    pgie.link(tracker)        # se añade para tracker
    # pgie.link(nvvidconv1)     se modifica
    tracker.link(nvvidconv1)  # se añade para ligar tracker con los demas elementos
    nvvidconv1.link(filter1)
    filter1.link(tiler)
    #tiler.link(nvvidconv)
    tiler.link(nvosd)
    #nvvidconv.link(nvosd)
    if is_aarch64():
        nvosd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(sink)
    '''

    # create an event loop and feed gstreamer bus messages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    tiler_sink_pad = tiler.get_static_pad("sink")
    if not tiler_sink_pad:
        com.log_error(" Unable to get src pad")
    else:
        tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)

    # List the sources
    com.log_debug("Now playing...")
    for camera_mac_address in sv.scfg:
        for service_id in sv.scfg[camera_mac_address]:
            if service_id == "source":
                com.log_debug("Now playing ... {}".format(sv.scfg[camera_mac_address]["source"]))

    com.log_debug("Starting pipeline")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except Exception as e:
        com.log_debug("------  ORIGINAL Exception in execution: {} ------".format(str(e)))
        pass

    # cleanup
    com.log_debug("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)


##################  SCRIPT FUNCTIONS ######## #########


def crop_and_get_faces_locations(n_frame, obj_meta, confidence):
    # convert python array into numy array format.
    frame_image = np.array(n_frame, copy=True, order='C')

    # convert the array into cv2 default color format
    rgb_frame = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)

    # draw rectangle and crop the face
    crop_image = draw_bounding_boxes(rgb_frame, obj_meta, confidence)

    return crop_image


def draw_box_around_face(obj_locations, obj_labels, image):
    # Draw a box around each face, body, car or any kind of object along with its labels
    for (top, right, bottom, left), face_label in zip(obj_locations, obj_labels):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)


def draw_bounding_boxes(image, obj_meta, confidence):
    # confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    # obj_name = pgie_classes_str[obj_meta.class_id]
    # image=cv2.rectangle(image,(left,top),(left+width,top+height),(0,0,255,0),2)
    # image=cv2.line(image, (left,top),(left+width,top+height), (0,255,0), 9)
    # Note that on some systems cv2.putText erroneously draws horizontal lines across the image
    # image=cv2.putText(image,obj_name+',C='+str(confidence),(left-5,top-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255,0),2)
    # image = cv2.putText(image, obj_name, (left-5,top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255,0), 2)
    crop_image = image[top-20:top+height+20, left-20:left+width+20]
    return crop_image


def cb_newpad(decodebin, decoder_src_pad, data):
    com.log_debug("In cb_newpad")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if gstname.find("video")!=-1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                com.log_error("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            com.log_error(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, object_, name, user_data):
    com.log_debug("Decodebin child added:"+name)

    if name.find("decodebin") != -1:
        object_.connect("child-added", decodebin_child_added, user_data)
    if is_aarch64() and name.find("nvv4l2decoder") != -1:
        com.log_debug("Setting buff api_version")
        object_.set_property("bufapi-version", True)


def create_source_bin(index, uri):
    com.log_debug("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    # bin_name = "source-bin-%s" %index     diferente al original deestram_imagedata_multistream
    bin_name = "source-bin-%02d" %index
    com.log_debug(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        com.log_error(" Unable to create source bin")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        com.log_error(" Unable to create uri decode bin")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has been created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decoded bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        com.log_error(" Failed to add ghost pad in source bin")
        return None
    return nbin


if __name__ == '__main__':
    sys.exit(main())
