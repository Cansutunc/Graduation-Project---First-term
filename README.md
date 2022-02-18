# Graduation Project First Part Focuses on Image Processing And Second Part will focus on Deep Learning...Next Deep Learning Repo will be uploaded soon
# Graduation-Project-Image Processing 
Project aim is to improve printed products from 3d printer and detecting failrues on the products to increase productivity. In that purpose, using image processing for detecting failrues on the printed objects. In this semester, focusing on the printed products. In image processing part, focusing on 2 main parts as pre image processing for making the images that are taken from cameras turning into processed images and developing algorithms for detecting failrues on the processed images. 

By using different algortihms that are made by myself for image processing such as Preprocessing Algortihms,Contour detection algorithms,Shape and Contour matching algorithms,Failrue detection algorihms, Feature extraction algorithms and more.

# Preprocessing
In image processing part we are focusing on 2 main parts as pre image processing for making the images that are taken from cameras turning into processed images and developing algorithms for detecting failrues on the processed images. 

In the image preprocessing stage, first applying filtering, threshold methods and connected object extraction on the image taken from the camera, respectively.
After performing these operations, the objects of interest on the image are made more specific and easy to process. 

In the study, the Gaussian filtering method, which uses a 5x5 core matrix, is used. Choosing a large size of the core matrix reduces the noise on the image and makes it blurry.
After the Gaussian filtering step, binarization of the gray image is processed next.
On the gray image, thresholding is applied and only the parts of the relevant objects are used.
The smallest (min) and maximum values (max) used in the thresholding process are determined as a result of experimental studies. By comparing whether the pixel values in the gray image are between min and max values, a new value assignment is performed for the binary image.
Then we decided to use otsu tresholding method, the function determines the optimal threshold value using the Otsu's algorithm and uses it instead of the specified thresh.

After thresholding, an image containing black and white colors is created. In the image, there are undesirable white dots in the black regions and undesirable black dots in the white regions. Morphological processing is applied in order to remove the noise on the obtained binary image. In the morphological process, 3x3, 5x5, etc., which are called structural elements on the binary image, which are given as input. square matrix is scrolled. In the morphological processing step, the image is updated by using neighboring pixel values in the structural element and binary image values.

In the morphological processing step, the image is updated by using neighboring pixel values in the structural element and binary image values.

In the proposed study, erosion and dilation morphological operations are applied on the binary image.


The erosion process is used to narrow the white areas on the binary image and to clear the white areas in the black areas.In the erosion operation, The core slides across the image. A pixel (1 or 0) in the original image is considered 1 only if all pixels below the core are 1, otherwise it will be eroded (made zero). Useful for removing small white noises.It is used for decreasing little white noises.

For the dilate operation, It is just opposite of erosion.The dilate process, on the other hand, expands the borders of the white areas, while at the same time clearing the black spots in the white area. Here, a pixel element is '1' if at least one pixel under the kernel is '1'. So it increases the white region in the image or size of foreground object increases. Normally, in cases like noise removal, erosion is followed by dilation. Because, erosion removes white noises, but it also shrinks our object. So we dilate it. Since noise is gone, they won't come back, but our object area increases. It is also useful in joining broken parts of an object.

# Feature Extraction

There are 2 main algorithm for calculating the similarity of the object images. 

One of them by using the IOU (Intersection Over Union) idea.
It is developed for calculating the area similarity of the objects for both camera and gives a result of similarity.
This result is used for both detecting if the printed object has defects and for detecting which type of defect object has.

The other main algorithm for comparing contour of the object images for similarity. This algorithm also used for both defect percentage and defect kind.

After finding defect percentage and defect type, showing the detected defect areas to the user. For that purpose using several algorithms.
