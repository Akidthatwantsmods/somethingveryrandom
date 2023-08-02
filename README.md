# Project Name = Bob the smile classifier

 it detects happy and unhappy faces
I wanted to do this because I like to do simple things thats all really
you can do this to detect who finds a joke funny and who does not, and for the future development of this project will include a live video stream

## The Algorithm

I use imagenet to detect the smile and unsmiling faces
I trained my model on colab on the dataset I downloaded on kaggle: https://www.kaggle.com/datasets/ghousethanedar/smiledetection?resource=download
I exported and tested the model on jetson nano
here is the colab: https://colab.research.google.com/drive/192QnxdF8Juk2wwc0I9ICik3jt7NmCs6N?usp=sharing
and here is the link to my second video: https://youtu.be/ZayF8yCt0fY

## Running this project

1.  to retrain the model please use https://colab.research.google.com/drive/192QnxdF8Juk2wwc0I9ICik3jt7NmCs6N
2. to test this model use {imagenet.py --model=model/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=face/labels.txt face/test/happy/[Example.jpg] [Example2.jpg]} login to your jetson nano to use it
