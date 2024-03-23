import os
import glob

path = '/home/anton/work'
video_extension = 'png'
files = glob.glob(path + "/*." + video_extension)
file_names = [os.path.basename(x) for x in glob.glob(path + "/*." + video_extension)]

print(files)
print(file_names)
