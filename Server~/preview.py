from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import cv2

class Preview:
    def __init__(self):
      self.img = None
      self.yolo_results = None
      self.timestamp = None
      self.llm_reply = None
      self.client_id = None
      self.imageCounter = 0
    
    def fit_text_in_two_lines(self, img, text, margin = 20):
      font=cv2.FONT_HERSHEY_SIMPLEX
      color=(255, 255, 255)
      thickness=2

      # Get image dimensions
      h, w = img.shape[:2]
      available_width = w - 2 * margin
      available_height = h - 2 * margin
      
      # Split text into two roughly equal parts at the middle
      words = text.split()
      middle_idx = len(words) // 2
      line1 = ' '.join(words[:middle_idx])
      line2 = ' '.join(words[middle_idx:])
      
      # Find the maximum font scale that allows both lines to fit
      max_scale = 0.1
      for scale in np.arange(0.1, 10.0, 0.1):
          # Get text sizes
          line1_size = cv2.getTextSize(line1, font, scale, thickness)[0]
          line2_size = cv2.getTextSize(line2, font, scale, thickness)[0]
          
          # Check if width fits
          if line1_size[0] > available_width or line2_size[0] > available_width:
              max_scale = scale - 0.1
              break
              
          # Check if height fits (two lines plus spacing)
          line_height = max(line1_size[1], line2_size[1])
          if 2 * line_height > available_height:
              max_scale = scale - 0.1
              break
              
          max_scale = scale
      
      # Get final sizes with the selected scale
      line1_size = cv2.getTextSize(line1, font, max_scale, thickness)[0]
      line2_size = cv2.getTextSize(line2, font, max_scale, thickness)[0]
      line_height = max(line1_size[1], line2_size[1])
      
      # Calculate positions to center text
      x1 = (w - line1_size[0]) // 2
      x2 = (w - line2_size[0]) // 2
      
      y_center = h // 2
      y1 = y_center - line_height // 2
      y2 = y_center + line_height + 10  # Add spacing between lines
      
      # Draw text
      cv2.putText(img, line1, (x1, y1), font, max_scale, color, thickness)
      cv2.putText(img, line2, (x2, y2), font, max_scale, color, thickness)
      
      return img

    def addImg(self, img, yolo_results, timestamp, client_id):
      self.img = img.copy()
      self.yolo_results = yolo_results
      self.timestamp = timestamp
      self.client_id = client_id
      self.render()
    
    def addReply(self, llm_reply):
      self.llm_reply = llm_reply
      self.render()

    # preview.render(img, yolo_results, timestamp, llm_reply, client_id)
    def render(self):
      if(self.img is None):
        return

      # save image to disk
      img_path = "preview.jpg"
      self.imageCounter += 1
      width = self.img.shape[1]
      height = self.img.shape[0]

      # draw bounding boxes
      for result in self.yolo_results:
          bbox = result["bbox"]
          x1, y1, x2, y2 = bbox
          cv2.rectangle(self.img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
          cv2.putText(self.img, result["class_name"], (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
      
      # show llm content
      blank = np.zeros(shape=(100, width, 3), dtype=np.int16)
      combined = self.img
      if(self.llm_reply is not None):
        llm_img = self.fit_text_in_two_lines(blank, self.llm_reply)

        # TODO: also want to add instructions on the right

        combined = np.concatenate((self.img, llm_img), axis=0) # axis = 0 for vertical, 1 for horizontal 
      cv2.imwrite(img_path, combined)

      # render photo dynamically in popup (this doesn't work on nunga)
      # cv2.imshow("Dynamic Image", combined)
      # ^ TODO: can do this in a separate script/client, running parallel to main