#! /bin/bash

opencv_annotation --annotations=annotations.txt --images=images/ 

opencv_createsamples -info annotations.txt -bg bg.txt -vec positives.txt -w 36 -h 36  

opencv_traincascade -data cascade_dir -vec positives.txt -bg bg.txt -numPos 73 -numNeg 4 -w 36 -h 36 