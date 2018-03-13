
# Vehicle Detection Project

## Introduction

To solve this task at first I used HOG features and SVM classifier and it gave me some good results (you can check what is done in this case by checking [project.py](project.py) file). However, I was very concerned about the speed of sliding window approach - it took way more time that I could imagine. In addition, extractig HOG features and training classifier was slow too. I waited hours to get some results from GridSearchCV method to tune my model parameters. 
Finally, I decided to take another approach. This approach is based on convolutional neural nets, in particular one type of architecture for object detection - SSD (Single Shot Detector). By using the SSD I didn't have to tune parameters for feature extraction, the false positives were removed easily by increasing the prediction confidence threshold and the model was already pre-trained on the 'car' class, so no additional training was required. 

[image1]: ./output_images/test1.jpg "test1.jpg"
[image2]: ./output_images/test2.jpg "test2.jpg"
[image3]: ./output_images/test3.jpg "test3.jpg"
[image4]: ./output_images/test4.jpg "test4.jpg"
[image5]: ./output_images/test5.jpg "test5.jpg"
[image6]: ./output_images/test6.jpg "test6.jpg"
[video1]: ./project_video.mp4


### 1. Network Architecture

The SSD approach ([link](https://arxiv.org/pdf/1512.02325.pdf)) is based on a feed-forward convolutional network that produces a fixed-size collection of bounding boxes and scores for the presence of object class instances in those boxes, followed by a non-maximum suppression step to produce the final detections.

The architecture consists of some early network layers which are based on a standard architecture (VGG-16). This architecture is followed by convolutional feature layers which decrease in size progressively and allow predictions of detections at multiple scales. Each added feature layer can produce a fixed set of detection predictions using a set of convolutional filters. For a feature layer, the basic element for predicting parameters of a potential detection is a small kernel that produces either a score for a category, or a shape offset relative to the default box coordinates.

The code for this project you can find inside [VehicleDetectionSSD.ipynb](VehicleDetectionSSD.ipynb) file.

### 2. Model training

The network has been initially trained on the [MS COCO](http://cocodataset.org/#home) dataset. I used pretrained weights for 80 classes from [here](https://drive.google.com/open?id=1vmEF7FUsWfHquXyCqO17UaXOPpRbwsdj). Then, I sub-sampled (or up-sampled) the weight tensors of all the classification layers to predict only five classes of objects which we need for this project: car, truck, bicyclist, motorcycle, bus. In the notebook you'll find the code which downloads the original weights, subsamples them for only those 5 classes above, makes predictions and draws bounding boxes for detected objects from test images, test video and project video. Keras SSD model for this project can be found in [this repo](https://github.com/pierluigiferrari/ssd_keras).

### 3. Model prediction and post processing

Once the weights of the pre-trained network have been loaded and subsampled, 
the predictions and drawing of the bounding boxes is performed in `process_data` function. Preprocessing of the images to fit model inputs are held inside `load_preprocess_image` and `preprocess_image`  fuctions. Actual prediction takes place in `detect_vehicles` function, which returns a list of predictions for every picture in the form `[label, confidence, xmin, ymin, xmax, ymax]` with a threshold of confidence 60%

```python
    confidence_threshold = 0.6    
    # Get detections with confidence higher than 0.6.
    y_pred = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
```
For lower values of threshold we can be sure that there won't be any vehicles missed, however that would also increase the number of false positives.

### 4. Results

Here are results of the proposed method on test images

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

### 5. Video Implementation

Project video and the result can be found [here][video1]

---

### Discussion

**Techniques used**

To solve the vehicle detection problem I used the SSD network architecture, subsampled for 5 classes of vehicles on the road. It provides huge benefits because it is fast, reliable and easy to implement.

**Where the pipeline might fail**

In most cases it will perform very well because the weights are pretrained on [MS COCO](http://cocodataset.org/#home) dataset. But it may fail on vehicles which are not from those five subsampled classes.

**How to improve**

Good way to improve the model would be to fine tune this subsampled model on Udacity's dataset, but it performs well in most cases even without additional data.