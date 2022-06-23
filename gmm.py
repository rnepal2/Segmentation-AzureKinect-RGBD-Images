# Image Segmentation
# Microsoft Azure Kinect RGB-Depth camera

# libraries
import os
from tqdm import tqdm
import glob
import time
import numpy as np 
import cv2
import pickle
from sklearn.mixture import GaussianMixture
import matplotlib
import matplotlib.pyplot as plt

# Depth and RBG images folder path
depth_dir = os.path.join(os.path.abspath('.'), 'AzureKinectData/depths')
image_dir = os.path.join(os.path.abspath('.'), 'AzureKinectData/rgbs') 

# Whether train or load trained model
train = False
# if not model saved: always train
if len(glob.glob("./*.pkl")) == 0:
    train = True

sample = cv2.imread(os.path.join(depth_dir, 'depth_0001.png'), cv2.IMREAD_UNCHANGED)
Lx, Ly = sample.shape
print('Input depth image shape: ', sample.shape)


# selects the list of n images (path) to use for training
def create_train_images(n=10):
    all_images_list = os.listdir(depth_dir)
    total_images = len(all_images_list)-1
    select_from = np.random.randint(0, total_images, n)
    image_list = []
    for i in select_from:
        image = all_images_list[i]
        image_list.append(image)
    return [os.path.join(depth_dir, image) for image in image_list]

# training images list
image_list = create_train_images(n=20)

save_model_as = "trained_model.pkl"
def train_model(train_data=image_list, save_model=True, filename=save_model_as, n_components=3):
    model = GaussianMixture(n_components=n_components, covariance_type='full', tol=1e-4, random_state=0)
    for i in tqdm(range(len(train_data))):
        X = cv2.imread(train_data[i], cv2.IMREAD_UNCHANGED)
        X = X.reshape(Lx*Ly, 1)
        model = model.fit(X)

    print(f"GMM model is trained on {len(train_data)} images.")
    if save_model:
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
    return model

def Model(train=train, train_data=image_list, n_components=3):
    if train:
        print(f"\n====Training the model====")
        model = train_model(train_data=train_data, save_model=True, filename=save_model_as, n_components=n_components)
    else:
        with open(save_model_as, 'rb') as file:
            model = pickle.load(file)
    return model


# create segments: in the same shape of the image: depth image 
def segment_image(model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # removing background using depth value info
    img[img < 80] = 0  # very near on the table
    img[img > 1400] = 0  # set everything beyond 1050 mm to zero
    
    img = img.reshape(Lx*Ly, 1)  # 2d matrix to an array to GMM input format 

    # predictions
    segments = model.predict(img)
    segments = segments.reshape(Lx, Ly)

    # keep only ROI - segment of interest only: from 0, 1, 2
    segments[segments == 1] = 0
    
    # clipping off top and bottom, carpet area in the front
    W = len(segments)
    H = len(segments[0])
    
    # segments[0:int(0.15*W), :] = 0
    # segments[int(0.85*W): , :] = 0
    segments[:, 0:int(0.15*H)] = 0
    segments[:, int(0.95*W) :] = 0
    
    segments = np.float64(segments)
    # Denoising 
    return segments #cv2.fastNlMeansDenoising(segments, None, 30.0, 7, 21)


# pass the axis and segments of the segmented image
def draw_bbox(ax, segments):
    # Iterate all colors in mask: we have only ROI in this case
    for color in np.unique(segments):
        # Color 0 is assumed to be background or artifacts
        if color == 0:
            continue

        # Determine bounding rectangle w.r.t. all pixels of the mask for the current color
        x, y, w, h = cv2.boundingRect(np.uint8(segments == color))

        # Draw bounding rectangle to color image
        rect = matplotlib.patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')

        # Show image with bounding box
        ax.add_patch(rect)
    return ax


def test_image_list(n=25):
    count = len(os.listdir(depth_dir))
    assert count == len(os.listdir(image_dir))
    skip = int(count/n)
    
    image_list = os.listdir(image_dir)[0:count:skip]
    depth_list = os.listdir(depth_dir)[0:count:skip]
    
    image_paths = [os.path.join(image_dir, image) for image in image_list]
    depth_paths = [os.path.join(depth_dir, image) for image in depth_list]
    
    return [(u, v) for (u, v) in zip(image_paths, depth_paths)]

test_images = test_image_list(n=25)

def test_plot():
    _, axs = plt.subplots(nrows=2, ncols=1)
    clusters = segment_image(model=Model(train=False, n_components=3), image_path=test_images[5])
    axs[0].imshow(cv2.imread(os.path.join(image_dir, 'image_0196.png')))
    axs[1].imshow(clusters, vmin=0, vmax=2)
    draw_bbox(axs[1], clusters)
    plt.show()

#test_plot()
#raise SystemExit

# Plotting the results
plt.ion()
class DynamicPlot:
    # sample image to initialize
    sample_image = cv2.imread(test_images[0][1], cv2.IMREAD_UNCHANGED)
    sample_rbg = cv2.cvtColor(cv2.imread(test_images[0][0]), cv2.COLOR_BGR2GRAY)
    
    print('depth: ', sample_image.shape)
    print('color: ', sample_rbg.shape)

    def __init__(self, model):
        #self.test_data = image_list
        self.segments = self.sample_image
        self.rbg = self.sample_rbg
        self.model = model
        self.figure, self.ax = plt.subplots(nrows=2, ncols=1)
        self.ax[0].imshow(self.rbg)
        self.ax[1].imshow(self.segments)

    def on_running(self, k):
        # finding the bbox edgepoints
        x = None
        for color in np.unique(self.segments):
                # Color 0 is assumed to be background or artifacts
            if color == 0:
                continue
            # Determine bounding rectangle w.r.t. all pixels of the mask for the current color
            x, y, w, h = cv2.boundingRect(np.uint8(self.segments == color))
        
        if x is not None: # to avoid the situation of no bbox found
            # modifying boundary line values to show bbox
            for i in range(x, x+w):
                try:
                    self.segments[y, i] = 10
                    self.segments[y+h, i] = 10
                except IndexError:
                    self.segments[y, i] = 10
                    self.segments[Lx-3, i] = 10

            for j in range(y, y+h):
                try:
                    self.segments[j, x] = 10
                    self.segments[j, x+w] = 10
                except IndexError:
                    self.segments[j, x] = 10
                    self.segments[j, Ly-3] = 10

        # update plot
        self.ax[0].imshow(self.rbg, aspect='equal')
        self.ax[1].imshow(self.segments, vmin=0, vmax=10)
        # update image and rescale
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        outpath = f"output/image_{k}.png"
        self.figure.savefig(outpath)


    def update(self, test_data):
        i = 0
        for image in tqdm(test_data):
            # load image: as np array: both rbg and depth images
            rbg_image = cv2.imread(image[0], cv2.IMREAD_UNCHANGED)
            self.rbg = cv2.cvtColor(rbg_image, cv2.COLOR_BGR2GRAY)
            self.segments = segment_image(self.model, image[1])
            # update segments and draw a bbox 
            self.on_running(i)
            # sleep to slow down little bit
            time.sleep(.05) 
            i += 1


# Run
if __name__ == "__main__":
    model = Model()
    d = DynamicPlot(model=model)
    d.update(test_data=test_images)


