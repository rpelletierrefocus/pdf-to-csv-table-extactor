import os
import argparse
from OCR import process_file,  PDFTOImage, setDebug


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--in", type=str, nargs='+', required=True,
                help="path to input images folder of scans")
ap.add_argument("-o", "--out", type=str, nargs='+', required=False,
                help="path to output images folder of result")
ap.add_argument('-d', action='store_true')
ap.add_argument('-dci', action='store_true') 
# don't change image
args = vars(ap.parse_args())
file_path = args["in"][0]
out_file_path = args["out"][0] if args["out"] is not None else "./out"

if args['d']:
    setDebug(True)

try:
    os.makedirs(os.path.join(file_path, "converted"))
    os.makedirs(out_file_path)

except FileExistsError:
    print("Directory ", out_file_path,  " already exists")


print(out_file_path)

included_extensions = ['jpg', 'jpeg', 'bmp', 'png', 'gif', 'pdf']
file_names = [fn for fn in os.listdir(file_path)
              if any(fn.endswith(ext) for ext in included_extensions)]

for file_name in file_names:
    filePath = os.path.join(file_path, file_name)
    try:
        if file_name.endswith('pdf'):
            outFilePath = os.path.join(
                file_path, "converted", file_name.replace('pdf', 'jpg'))
            filePath = PDFTOImage(filePath, outFilePath)
        process_file(filePath,
                     os.path.join(out_file_path, file_name), not args['dci'])
        print(file_name)
    except Exception as e:
        print("err:", e)
        continue
