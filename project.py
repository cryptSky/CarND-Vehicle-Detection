import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import glob
import time
from sklearn.svm import LinearSVC, SVC
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from sklearn.model_selection import train_test_split
from lesson_functions import *
from moviepy.editor import VideoFileClip
from sklearn import svm
import csv
from sklearn.model_selection import GridSearchCV
import os
from collections import deque

class VehicleTracker:

    def __init__(self, n):
        self.n = n
        self.recent_bboxes = deque(maxlen=n)
    
    def list_standard_data(self):
        non_vehicles = []
        non_vehicle_path = "./non_vehicles/"
        for folder in os.listdir(non_vehicle_path):
            non_vehicles.extend(glob.glob(non_vehicle_path+folder+'/*.png'))
            #non_vehicles.extend([non_vehicle_path+folder+fname for fname in os.listdir(non_vehicle_path + folder)])
            
        vehicles = []
        vehicle_path = "./vehicles/"
        for folder in os.listdir(vehicle_path):
            vehicles.extend(glob.glob(vehicle_path+folder+'/*.png'))
            #vehicles.extend([vehicle_path+folder+fname for fname in os.listdir(vehicle_path + folder)])
            
        return vehicles, non_vehicles
    
    def list_crowdai_data(self):
        path = './object-detection-crowdai/'
        csv_path = path + 'labels.csv'
        result_folder = './crowdai_vehicles/'
        
        with open(csv_path, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            filename = ''
            index = 0
            
            for row in reader:
                if row[4] != filename:
                    img = mpimg.imread(path + row[4])
                    filename = row[4]
                    index = 0
                if row[5].lower() == 'car' or row[5].lower() == 'truck':                   
                    xmin,xmax,ymin,ymax = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    new_img = img[xmin:xmax, ymin:ymax]                
                    cv2.imwrite(filename+'_'+str(index), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    index += 1
    
    def plot_color_hist(self, rh):
        # Plot a figure with all three bar charts
        if rh is not None:
            fig = plt.figure(figsize=(12,3))
            plt.subplot(131)
            plt.bar(bincen, rh[0])
            plt.xlim(0, 256)
            plt.title('R Histogram')
            plt.subplot(132)
            plt.bar(bincen, gh[0])
            plt.xlim(0, 256)
            plt.title('G Histogram')
            plt.subplot(133)
            plt.bar(bincen, bh[0])
            plt.xlim(0, 256)
            plt.title('B Histogram')
            fig.tight_layout()
        else:
            print('Your function is returning None for at least one variable...')
            
    def color_spacex(self, img):
        # Select a small fraction of pixels to plot by subsampling it
        scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
        img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
    
        # Convert subsampled image to desired color space(s)
        img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
        img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting
        
        # Plot and show
        plot3d(img_small_RGB, img_small_rgb)
        plt.show()
    
        plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
        plt.show()
        
       
    # Define a function to return some characteristics of the dataset 
    def data_look(car_list, notcar_list):
        data_dict = {}
        # Define a key in data_dict "n_cars" and store the number of car images
        data_dict["n_cars"] = len(car_list)
        # Define a key "n_notcars" and store the number of notcar images
        data_dict["n_notcars"] = len(notcar_list)
        # Read in a test image, either car or notcar
        example_img = mpimg.imread(car_list[0])
        # Define a key "image_shape" and store the test image shape 3-tuple
        data_dict["image_shape"] = example_img.shape
        # Define a key "data_type" and store the data type of the test image.
        data_dict["data_type"] = example_img.dtype
        # Return data_dict
        return data_dict
        
    def param_search(self):
        with open("features.p", "rb" ) as handle:
            features = pickle.load(handle)
            
        params =  {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf', 'linear']}
        svc = svm.SVC()
        clf = GridSearchCV(svc, params, cv=5, n_jobs=4, verbose=10)
        clf.fit(np.concatenate((features["X_train"], features["X_test"])), np.concatenate((features["y_train"], features["y_test"])))
        
        print(clf.best_params_)
    
    def extract_features(self):
        # Divide up into cars and notcars
        cars, notcars = self.list_standard_data()
        
        ### TODO: Tweak these parameters and see how the results change.
        spatial = 32
        histbin = 32
        
        colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 11
        pix_per_cell = 14
        cell_per_block = 2
        hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
        
        print('Using:',orient,'orientations',pix_per_cell,
            'pixels per cell and', cell_per_block,'cells per block')        
                
  
     
        t=time.time()
        car_features = extract_features(cars, color_space=colorspace, 
                                spatial_size=(spatial, spatial),
                                hist_bins=histbin, hist_range=(0, 256),
                                orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                hog_channel=hog_channel)
        notcar_features = extract_features(notcars, color_space=colorspace, 
                                spatial_size=(spatial, spatial),
                                hist_bins=histbin, hist_range=(0, 256),
                                orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                hog_channel=hog_channel)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to extract HOG features...')
        
        car_features = np.nan_to_num(car_features)
        notcar_features = np.nan_to_num(notcar_features)
        
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        
        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state)
                                
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)
        
        features = {}
        features["X_train"] = X_train
        features["y_train"] = y_train
        features["X_test"] = X_test
        features["y_test"] = y_test
        features["orient"] = orient
        features["pix_per_cell"] = pix_per_cell
        features["cell_per_block"] = cell_per_block
        features["spatial_size"] = spatial
        features["hist_bins"] = histbin
        features["scaler"] = X_scaler
        
        filename = 'features_new.p'
        print('Saved features data to {0} ...'.format(filename))
        with open(filename, 'wb') as file:
            pickle.dump(features, file)
    
    def classify(self):

        with open("features_new.p", "rb" ) as handle:
            features = pickle.load(handle)
            
        X_train = features["X_train"]
        y_train = features["y_train"]
        X_test = features["X_test"] 
        y_test = features["y_test"]        

        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC 
        svc = SVC(C=100, kernel='rbf')
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        
        n_predict = 10
        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these',n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
        
        model_data = {}
        model_data["svc"] = svc
        model_data["orient"] = features["orient"]
        model_data["pix_per_cell"] = features["pix_per_cell"]
        model_data["cell_per_block"] = features["cell_per_block"]
        model_data["spatial_size"] = features["spatial_size"]
        model_data["hist_bins"] = features["hist_bins"]
        model_data["scaler"] = features["scaler"]
        
        filename = 'svc_pickle.p'
        print('Saved model data to {0} ...'.format(filename))
        with open(filename, 'wb') as file:
            pickle.dump(model_data, file)
        
    def find_cars(self, img):
         
        # get attributes of our svc object
        svc = self.model_data["svc"]
        X_scaler = self.model_data["scaler"]
        orient = self.model_data["orient"]
        pix_per_cell = self.model_data["pix_per_cell"]
        cell_per_block = self.model_data["cell_per_block"]
        spatial_size = self.model_data["spatial_size"]
        hist_bins = self.model_data["hist_bins"]
                
        ystart = 400
        ystop = 656
        scale = 1.5
            
        #out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        out_img, bboxes_all = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        
        #print(bboxes_all)
        
        out_img, bboxes = self.draw_real_boxes(out_img, bboxes_all)
        
        #plt.imshow(out_img)
        #plt.show()
        
        return out_img
    
    def get_real_bboxes(self, image):
        final_heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
        for bboxes in self.recent_bboxes:
            final_heat = add_heat(final_heat,bboxes)
           
        #print(np.max(final_heat))
           
        # Apply threshold to help remove false positives
        if len(self.recent_bboxes) >= self.n:
            final_heat = apply_threshold(final_heat, self.n - 1)
        
        # Visualize the heatmap when displaying    
        heatmap = np.clip(final_heat, 0, 255)
        
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        final_image, real_bboxes = get_labeled_bboxes(np.copy(image), labels, True)
        
        return final_image, real_bboxes
    
    def draw_real_boxes(self, image, box_list):
        # Read in a pickle file with bboxes saved
        # Each item in the "all_bboxes" list will contain a 
        # list of boxes for one of the images shown above
        # box_list = pickle.load( open( "bbox_pickle.p", "rb" ))
        
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
                
        # Add heat to each box in box list
        heat = add_heat(heat,box_list)
        
        #print(np.max(heat))
        #print(heat[heat >= 7])
        
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 3)
        
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img, recent_bboxes = get_labeled_bboxes(image, labels)     
        
        self.recent_bboxes.appendleft(recent_bboxes)
        
        draw_img, real_bboxes = self.get_real_bboxes(image)
        
        return draw_img, real_bboxes
        
    
    def process_frame(self, img):
        #cv2.imwrite('test0.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        result = vt.find_cars(img)    
        #plt.imshow(result)
        #plt.show()
        return result        
       
        
    def process_video(self):
    
        white_output = 'result_project_video.mp4'
        ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
        ## To do so add .subclip(start_second,end_second) to the end of the line below
        ## Where start_second and end_second are integer values representing the start and end of the subclip
        ## You may also uncomment the following line for a subclip of the first 5 seconds
        ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
        clip1 = VideoFileClip("project_video.mp4")
        white_clip = clip1.fl_image(self.process_frame) #NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)
        white_clip.reader.close()
        white_clip.audio.reader.close_proc()  

        
if __name__ == "__main__":
    vt = VehicleTracker(6)
    
    #vt.extract_features()
    #vt.classify()
    #vt.param_search()
    
    #for image_name in glob.glob('./test_images/*.jpg'):
    #    image = mpimg.imread(image_name)
    #    res = vt.find_cars(image)
    #    plt.imshow(res)
    #    plt.show()
    
    #vt.color_spacex(image)
    #hist_features = color_hist(image, nbins=32, bins_range=(0, 256))
    #vt.train(, 'non_vehicles')

    # load a pe-trained svc model from a serialized (pickle) file
    with open("svc_pickle.p", "rb" ) as handle:
        vt.model_data = pickle.load(handle)    
    
    vt.process_video()