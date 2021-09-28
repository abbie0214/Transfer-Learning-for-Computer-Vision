# Mid-term Project: Domain Adaptation Part

Transfer Learning for Computer Vision: domain adaptation code

To use it:  
1. uncomment the correspingding code of main.py in line 32-57.  
eg. If you want to do domain apadtation from domain W to domian A, uncomment:

```
####  W - > A ####
name='W to A'
input_filename_train= 'webcam_webcam.csv'
input_filename_test = 'webcam_amazon.csv'
``` 
and leave other options comment.  

2. run main_unsupervised.py or main_supervised.py for unsupervised and supervised domain adaptation experiment. 

    
A jupter demo illustrates the training and testing process. You can directly run it.
