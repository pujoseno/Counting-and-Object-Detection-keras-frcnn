# Counting-and-Object-Detection-keras-frcnn
How to practice Counting and Object Detection keras frcnn

# Case
In this case we practice about Counting and Objecct detection using keras and frcnn. In frcnn model we have class are background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor.

Region-based Deep Convolutional Networks are exciting tools, enabling software developers to solve many interesting problems. The presented solution just scratches the surface. By fine-tuning the network for the particular data set or using transfer learning from other trained models, we can achieve high accuracy and speed while detecting objects.

### Structure
* Counting and Object Detection
  * keras_frcnn
    * config.pickle
    * config.py
    * data_augment.py
    * data_generator.py
    * FixedBatchNormalization.py
    * losses.py
    * pascal_voc_parser.py
    * resnet.py
    * roi_helpers.py
    * RoiPoolingConv.py
    * simple_parser.py 
  * outputs
  * pre-trained model
    * pretrainedmodel.hdf5
  * Videos
    * test.mp4
  * test_frcnn.py
  * train_frcnn.py
  
### Dependencies
  * Keras
  * Imutils
  * cv2
  * Numpy
                  
### How to Run
  * Add input_dir and ouput_dir directory in test_frcnn.py
    parser.add_option("-d", "--input_dir", dest="input_dir", default=~~"F:/counting~~/outputs")
    parser.add_option("-u", "--output_dir", dest="output_dir", default=~~"F:/counting~~/outputs")
  * run
    python test_frcnn.py --input_file videos/test.mp4 --output_file outputs/output4.mp4 --frame_rate=25
    
