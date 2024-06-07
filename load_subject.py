#!/usr/bin/python3
import sys
import os
from pathlib import Path
import lib.common as com

param_length = len(sys.argv)
base_input_dir = com.BASE_INPUT_DB_DIRECTORY

msg = 'Usage: ' + sys.argv[0] + ' newBlackList | newWhiteList | addToBlackList | addToWhiteList | removeBlackList | ' \
                                'removeWhiteList '

if param_length < 2:
    com.log_error(msg)

if sys.argv[1] == 'newBlackList' or sys.argv[1] == 'addToBlackList':
    if param_length == 2:
        blacklist_face_images = base_input_dir + '/blacklist_faces'
        blacklist_results_dir = base_input_dir + '/blacklist_db'
        com.create_data_dir(blacklist_results_dir)
        com.create_data_dir(blacklist_face_images)
        try:
            blacklist_data = blacklist_results_dir + '/' + com.BLACKLIST_DB_NAME
            import lib.biblioteca as biblio
            if sys.argv[1] == 'newBlackList':
                biblio.encode_known_faces_from_images_in_dir(blacklist_face_images, blacklist_data, 'blacklist')
            else:
                biblio.encode_known_faces_from_images_in_dir(blacklist_face_images, blacklist_data, 'blacklist', True)
        except AttributeError:
            com.log_error("Configuration error - environment variable 'BLACKLIST_DB_NAME' not set")
        com.log_debug("Saving data in directory: {}".format(blacklist_results_dir))

    else:
        com.log_error(msg)

elif sys.argv[1] == 'newWhiteList' or sys.argv[1] == 'addToWhiteList':
    if param_length == 2:
        whitelist_face_images = base_input_dir + '/whitelist_faces'
        whitelist_results_dir = base_input_dir + '/whitelist_db'
        com.create_data_dir(whitelist_results_dir)
        com.create_data_dir(whitelist_face_images)
        try:
            whitelist_data = whitelist_results_dir + '/' + com.WHITELIST_DB_NAME
            import lib.biblioteca as biblio

            if sys.argv[1] == 'newWhiteList':
                biblio.encode_known_faces_from_images_in_dir(whitelist_face_images, whitelist_data, 'whitelist')
            else:
                biblio.encode_known_faces_from_images_in_dir(whitelist_face_images, whitelist_data, 'whitelist', True)
        except AttributeError:
            com.log_error("Configuration error - environment variable 'WHITELIST_DB_NAME' not set")
        com.log_debug("Saving data in directory: {}".format(whitelist_results_dir))

    else:
        com.log_error(msg)
else:
    com.log_error(msg)
