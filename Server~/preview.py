from PIL import Image, ImageDraw, ImageFont

class Preview:
    def __init__(self):
      return
    
    def render(self, img, width, height, object_labels, object_centers, timestamp, content):
      draw = ImageDraw.Draw(image)

      assert(len(object_labels) == len(object_centers))
      # add each of the labels
      for i in range(len(object_labels)):
        draw.text(object_centers[i], object_labels[i])
      
      # save the image
      output_path = "/images/img" + timestamp
      image.save(output_path)