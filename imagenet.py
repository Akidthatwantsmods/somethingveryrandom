#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse

from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log

def process_image(input, output, network, topK):
    # load the recognition network
    # net = imageNet(network, sys.argv)

    # note: to hard-code the paths to load a model, the following API can be used:
    #
    net = imageNet(model="model/resnet18.onnx", labels="model/labels.txt", 
                    input_blob="input_0", output_blob="output_0")

    # create video sources & outputs
    input = videoSource(input, argv=sys.argv)
    output = videoOutput(output, argv=sys.argv)
    font = cudaFont()

    # process frames until EOS or the user exits
    
        # capture the next image
    img = input.Capture()

    if img is None: # timeout
        return  

    # classify the image and get the topK predictions
    # if you only want the top class, you can simply run:
    #   class_id, confidence = net.Classify(img)
    predictions = net.Classify(img, topK=topK)

    # draw predicted class labels
    for n, (classID, confidence) in enumerate(predictions):
        classLabel = net.GetClassDesc(classID)
        confidence *= 100.0
        if classLabel == "Class #1":
            classLabel = "Happy"
        else:
            classLabel = "Unhappy"

        print(f"imagenet:  {confidence:05.2f}% class #{classID} ({classLabel})")

        font.OverlayText(img, text=f"{confidence:05.2f}% {classLabel}", 
                        x=2, y=2 + n * (font.GetSize() + 2),
                        color=font.White, background=font.Gray10)
                        
    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    return classLabel
