from flask import Flask, send_file, send_from_directory
import io
import numpy as np
from matplotlib import pyplot as plt
from sentinelhub import CRS, BBox, WmsRequest, DataCollection, SHConfig

app = Flask(__name__)


@app.route('/')
def download_image():
    config = SHConfig()
    config.instance_id = "a7daa129-9465-40ce-ad80-4f369c57922e"

    # It's assumed that you have set the instance ID and other configurations in your SHConfig already.

    betsiboka_coords_wgs84 = (46.16, -16.15, 46.51, -15.58)
    betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
    wms_true_color_request = WmsRequest(
        data_collection=DataCollection.SENTINEL2_L1C,
        layer='TRUE-COLOR-S2L2A',
        bbox=betsiboka_bbox,
        time='2017-12-15',
        width=512,
        height=856,
        config=config
    )

    wms_true_color_img = wms_true_color_request.get_data()
    image = wms_true_color_img[-1]  # Take the last image from the list

    # Save the image to a BytesIO object
    image_bytes = io.BytesIO()
    plt.imsave(image_bytes, image, format='png')
    image_bytes.seek(0)  # Go to the beginning of the IO stream

    return send_file(
        image_bytes,
        mimetype='image/png',
        as_attachment=True,
        download_name='satellite_image.png'
    )


@app.route('/.well-known/ai-plugin.json')
def serve_ai_plugin():
    return send_from_directory('.',
                               'ai-plugin.json',
                               mimetype='application/json')


@app.route('/openapi.yaml')
def serve_openapi_yaml():
    return send_from_directory('.', 'openapi.yaml', mimetype='text/yaml')


if __name__ == '__main__':
    app.run(debug=True)
