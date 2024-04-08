from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from typing import List
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import io
import uvicorn
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


# Please specify the path to the model
model_path = "./PM1_AI_CaseStudy/model_scripted.pt"

# Adopted and modified from https://github.com/aasimsani/model-quick-deploy

def load_model(model_path):
    # Load the model
    model = torch.jit.load(model_path)
    model.eval()

    # send the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model


def pre_process(image):
    # Load image
    img = cv2.imdecode(np.frombuffer(image.file.read(),
                                      np.uint8),
                        cv2.IMREAD_COLOR)
    # convert it to the correct format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_size = img.size

    # Transform it so that it can be used by the model
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
    imagenet_std = torch.tensor([0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize(size=(64, 64), interpolation=T.InterpolationMode.BICUBIC),
                           T.RandomHorizontalFlip(p=0.5),
                           T.ToTensor(),
                           T.Normalize(imagenet_mean, imagenet_std),
                           ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformed_img = transform(img).unsqueeze(0).to(device)

    # Create data dictionary to store the image and other information
    data = {"img": img, "img_size": img_size, "device": device, "transformed_img": transformed_img}

    # Return this data so it can be used in postprocessing
    return data


def post_process(data, prediction):
    # Get the image and the size of the image
    original = data["img"]
    original_size = data["img_size"]

    # Create a matplotlib canvas to render the images
    fig = Figure()
    fig.set_size_inches(original_size[0] / fig.dpi, original_size[1] / fig.dpi)
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    # Render the original image as foreground
    ax.imshow(np.asarray(original)[:, :, [2, 1, 0]])
    ax.set_title(prediction)
    ax.axis("off")
    canvas.draw()

    # Reshape output to be a numpy array
    width, height = fig.get_size_inches() * fig.get_dpi()
    width = int(width)
    height = int(height)
    output_image = np.frombuffer(canvas.tostring_rgb(),
                                dtype='uint8').reshape(height, width, 3)

    # Encode to png
    _, im_png = cv2.imencode(".png", output_image)

    return im_png


# Code from: https://fastapi.tiangolo.com/tutorial/request-files/
app = FastAPI()


@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    """ Create API endpoint to send image to and specify
     what type of file it'll take

    :param files: Get image files, defaults to File(...)
    :type files: List[UploadFile], optional
    :return: A list of png images
    :rtype: list(bytes)
    """

    # Class annotations
    classes = {
        0: "NonSelfie",
        1: "Selfie",
    }

    # Load the model
    model = load_model(model_path)

    for image in files:

        # Return preprocessed input batch and loaded image
        data = pre_process(image)

        # Run the model and postpocess the output
        with torch.no_grad():
            output = model(data["transformed_img"])
            prediction = torch.max(output.detach().cpu(), 1)[1].item()
            label = classes[prediction]


        # Post process and label the image with prediction
        output_image = post_process(data, label)
            
        return StreamingResponse(io.BytesIO(output_image.tobytes()),
                                 media_type="image/png")


@app.get("/")
async def main():
    """Create a basic home page to upload a file

    :return: HTML for homepage
    :rtype: HTMLResponse
    """

    content = """<body>
                <h3>Upload an image to learn if it is a Selfie or NonSelfie</h3>
                <form action="/uploadfiles/" enctype="multipart/form-data" method="post">
                    <input name="files" type="file" multiple>
                    <input type="submit">
                </form>
                </body> 
            """
    
    
    return HTMLResponse(content=content)

uvicorn.run(app)