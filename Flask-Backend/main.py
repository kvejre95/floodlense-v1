import io
import os

import cv2
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from matplotlib import pyplot as plt
from sentinelhub import CRS, BBox, DataCollection, SHConfig, WmsRequest

from model import UNet_Model, DoubleConv, UNet
import logging
import torch
import torch.nn as nn

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s : %(message)s')

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

dmodel = UNet_Model()


@app.route("/")
def index():
    logging.debug("Index route accessed")
    return "API is Up and Running."


@app.route('/download_image', methods=["GET"])
def download_image():
    config = SHConfig()
    SH_INSTANCE_ID = os.environ.get('SH_INSTANCE_ID')
    
    config.instance_id = SH_INSTANCE_ID

    # Retrieve latitude and longitude from query parameters
    latitude = float(request.args.get('latitude', 41.9028))

    longitude = float(request.args.get('longitude', 12.4964))

    # Create bounding box around the specified coordinates
    # You might want to adjust the size of the bounding box depending on your requirements
    bbox_lat_offset = 0.023  # size, adjust as needed
    bbox_long_offset = 0.047  # size, adjust as needed
    bbox_coords_wgs84 = (longitude - bbox_long_offset,
                         latitude - bbox_lat_offset,
                         longitude + bbox_long_offset,
                         latitude + bbox_lat_offset)
    # bbox_coords_wgs84 = (12.44693, 41.870072,  12.541001,  41.917096) # Rome

    bbox = BBox(bbox=bbox_coords_wgs84, crs=CRS.WGS84)

    # Retrieve Sentinel-2 data for the specified bounding box
    wms_true_color_request = WmsRequest(
        data_collection=DataCollection.SENTINEL2_L1C,
        layer='TRUE-COLOR-S2L2A',
        bbox=bbox,
        time='latest',
        width=780,
        height=512,
        maxcc=0.2,
        config=config)

    wms_true_color_img = wms_true_color_request.get_data()

    # Take the last image from the list
    # np.save('array_file.npy', image)
    image = wms_true_color_img[-1][:,:,:3][..., ::-1]
    # image_rgb = image_temp[:, :, :3]
    # image = image_rgb[..., ::-1]
    logging.info(type(image), image.shape)
    # Get dimensions of the image
    height, width = image.shape[:2]

    # Calculate the right part of the image to crop to 512x512
    start_row, start_col = int((height - 512) / 2), int(width - 512)

    # Crop the image to 512x512 from the right part
    cropped_image = image[start_row:start_row + 512, start_col:start_col + 512]

    # Downscale the cropped image to 256x256
    resized_image = cv2.resize(cropped_image, (256, 256), interpolation=cv2.INTER_AREA)
    logging.info(resized_image.shape)

    # Convert the result back to RGB
    output = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Use a relative path in Replit
    save_directory = 'images'

    # Make sure the directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    timestamp = datetime.now()

    # Convert the datetime to a timestamp (int)
    timestamp_int = int(timestamp.timestamp())

    filename = f"satellite_image_{timestamp_int}.jpeg"

    # Full path for the image
    file_path = os.path.join(save_directory, filename)

    # Save the processed image
    plt.imsave(file_path, output, format='jpeg')

    dmodel.predict_data(file_path, timestamp_int)

    print(f"Image saved to {file_path}")
    url = f"http://localhost:5000/show_image/{timestamp_int}"

    # Return the Image url to view
    # return f"https://chatgptplugin-vejrekshitij1.replit.app/show_image/{timestamp_int}"
    return jsonify({"url": url})


@app.route('/show_image/<int:timestamp>')
def show_image(timestamp):
    # Get the list of files in the images folder
    image_folder = 'images'  # Change this to the path of your images folder
    image_files = os.listdir(image_folder)
    # Iterate through the files and check if any match the specified timestamp
    for filename in image_files:
        # Check if the filename matches the format "satellite_image_timestamp.png"
        if filename.startswith('output_') and filename.endswith(
                f'{timestamp}.png'):
            # Save the file path
            file_path = f'{image_folder}/{filename}'

            # Send the file as a response with appropriate headers
            return send_file(file_path, mimetype='image/png')

    # If no matching image is found, return an error response
    return jsonify({'error': 'Image not found for the specified timestamp'}), 404


@app.route('/show_folder_images')
def show_folder_images():
    image_folder = 'images'  # Change this to the path of your images folder
    image_files = os.listdir(image_folder)

    # Return a JSON object containing all filenames
    return {"filenames": image_files}


# Provide API for OpenAI with Json data to provide chatbot with description
@app.route('/.well-known/ai-plugin.json')
def serve_ai_plugin():
    return send_from_directory('.',
                               'ai-plugin.json',
                               mimetype='application/json')


# Provide API for OpenAI with yaml configuration file
@app.route('/openapi.yaml')
def serve_openapi_yaml():
    return send_from_directory('.', 'openapi.yaml', mimetype='text/yaml')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
