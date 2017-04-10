#! /bin/bash

opencv_annotation --annotations=annotations.txt --images=positives/ 

opencv_createsamples -info annotations.txt -bg bg.txt -vec positives.txt -w 24 -h 24  

opencv_traincascade -data cascade_dir -vec positives.txt -bg bg.txt -numPos 114 -numNeg 4 -w 24 -h 24 