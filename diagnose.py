from PIL import Image
from pipeline import colorize

img = Image.open("test.png")  # apni kisi test NIR image ka naam/path daalo

result = colorize(img, debug=True)
result.save("diagnostic_output.png")
print("Saved diagnostic_output.png")