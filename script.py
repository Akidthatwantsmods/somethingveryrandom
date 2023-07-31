import sys
import argparse
import shutil
import os

from jetson_inference import imageNet
from imagenet import process_image
from jetson_utils import videoSource, videoOutput, cudaFont, Log




parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--topK", type=int, default=1, help="show the topK number of class predictions (default: 1)")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)
	
categories: any = os.listdir(args.input)
total = 0
error = 0

def check_output(category, labels):
	for label in labels:
		if category in label:
			return True
		return False
	

for category in categories:
	category_folder_path: any = os.path.join(args.input, category)
	images: any = os.listdir(category_folder_path)
	total += len(images)
	output_category_folder_path = os.path.join(args.output, category)
	if os.path.exists(output_category_folder_path):
		shutil.rmtree(output_category_folder_path)	
	os.makedirs(output_category_folder_path, exist_ok = True)

	for image in images:
		image_path: any = os.path.join(category_folder_path, image) 
		output_path: any = os.path.join(output_category_folder_path, "test_{}".format(image))
		labels = process_image(image_path, output_path, args.network, args.topK)

		if not check_output(category, labels):
			error += 1

print("Accuracy: ", (total -error)/total)