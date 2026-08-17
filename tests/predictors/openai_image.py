#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

"""
OpenAI-compatible fake image generation model.

Instead of generating, the model resizes a fixed stock photo to the
requested dimensions and returns `num_images` copies per prompt.

Consumed by muna-server `tests/serving.rs`: `/v1/images/generations`
returns b64-encoded images, one per prompt, at the requested size.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna", "requests"]
# ///

from io import BytesIO
from muna import compile, Parameter
from muna.beta import Annotations
from PIL import Image
from requests import get
from typing import Annotated

image_response = get(
    "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
    headers={ "User-Agent": "muna/1.0" },
)
image = Image.open(BytesIO(image_response.content))

@compile(
    tag="@muna/test-openai-image",
    access="unlisted"
)
def image_model(
    prompt: Annotated[
        list[str],
        Parameter.Generic(description="Text descriptions of the desired images.")
    ],
    *,
    width: Annotated[int, Annotations.ImageWidth(
        description="Generated image width in pixels.",
        min=256,
        max=2048
    )]=1024,
    height: Annotated[int, Annotations.ImageHeight(
        description="Generated image height in pixels.",
        min=256,
        max=2048
    )]=1024,
    num_images: Annotated[int, Annotations.ImageCount(
        description="Number of images to generate per prompt.",
        min=1,
        max=4
    )]=1,
) -> Annotated[
    list[Image.Image],
    Parameter.Generic(description="Generated images.")
]:
    """
    Fake model compatible with the OpenAI Images API.
    """
    resized_image = image.resize((width, height))
    return [resized_image] * (num_images * len(prompt))

if __name__ == "__main__":
    results = image_model(["image of a dog"])
    results[0].show()
