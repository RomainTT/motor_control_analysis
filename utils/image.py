#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
from PIL import Image, ImageDraw, ImageFont


def fig2img(fig, bbox_inches=None, dpi=None):
    """Transform a matplotlib figure into a Pillow image"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=bbox_inches, transparent=False)
    buf.seek(0)
    img = Image.open(buf)
    return img


def ax2img(fig, ax, expand=None, dpi=None):
    """Transform a matplotlib axe into a Pillow image

    Args:
        fig: the matplotlib figure containing the axe.
        ax: the matplotlib axe to transform
        dpi (optional): int indicating the number of DPI.
          Default is matplotlib rcParams["savefig.dpi"].
        expand (optional): a tuple (left, right, up, down)
          containing inches to expand each side of the axe.
          example: (1,0,2,1) will expand the left side of 1 inch,
          the up side of 2 inches, and the down side of 1 inch.

    Returns:
        An image of the given axe (expanded if expand is given).
        The object is an instance of PIL.Image.
    """
    if type(dpi) is not float or dpi <= 0:
        ValueError("Wrong dpi value")
    if type(expand) is not list or len(expand) != 4:
        ValueError("Wrong expand value")
    extent = ax.get_window_extent()
    extent = extent.transformed(fig.dpi_scale_trans.inverted())
    if expand:
        extent.y0 -= expand[3]  # down
        extent.y1 += expand[2]  # up
        extent.x0 -= expand[0]  # left
        extent.x1 += expand[1]  # right
    img = fig2img(fig, dpi=dpi, bbox_inches=extent)  # Get the PIL image
    return img


def add_txt_on_img(img, txt, pos, size, color="black", font="FreeSans"):
    """Add some text on a PIL image.

    No object is returned as img is modified inplace.

    Args:
        img (PIL.Image): the image being the background.
        txt (str): the text to write.
        pos (int, int): Top left corner position (x,y) of the text, in pixels.
        size (int): the font size, in points.
        color (optional): color of the text. Can be a string with the name of
          the color, or a tuple (R,G,B). Default is 'black'.
        font (str) (optional): The name of the Font to use. Must be installed
          on the system. Default is 'FreeSans'.
          If font does not exist, OSError is raised.
    """
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font, size)
    draw.text(pos, txt, color, font)


def add_img_on_img(img, added_img, pos, resize=None):
    """Add an image on top of another image.

    No object is returned as img is modified inplace.

    Args:
        img (PIL.Image): the image being the background.
        added_img (PIL.Image): The image to add on top of img.
        pos: either a 2-tuple giving the upper left corner, a 4-tuple
             defining the left, upper, right, and lower pixel coordinate.
             If the size of the box does not match the size of added_img,
             then added_img is reshaped.
        resize (float): Ratio to resize added_img, keeping height/width ratio.
    """
    if resize:
        size = added_img.size
        new_size = (int(size[0] * resize), int(size[1] * resize))
        added_img = added_img.resize(new_size)
    img.paste(added_img, pos)
