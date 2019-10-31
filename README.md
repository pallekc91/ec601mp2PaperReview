# ec601mp2PaperReview

## Introduction
__Multi-View 3D Object detection network for autonomous driving__\
A proposed high accuracy 3D multi-view object detection mechanism for autonomous driving. A system called MV3D (Multi view 3D) network which takes both LIDAR bird’s eye and point cloud and RGB image as input and predicts 3D boxes around objects including their orientation. It consists of 2 subnetworks, first which generates the 3D object proposal and the next for combining features from multiple views. It was tested on the KITTI benchmark and it outperformed the state-of-the-art by around 25% in the task of 3D localization and detection. It also performed 15% better against various LIDAR based state-of-the-art methods in 2D recognition. 
In comparison with general image-based methods, which typically generate 3D boxes and then perform region-based recognition using R-CNNs, a combination of LIDAR and image-based methods perform better in 3D object prediction. The reason for going for this method was to perform region-based feature fusion. 

The architecture of the system contains two major components, a 3D proposal network and a region-based feature fusion network. The network takes bird’s eye view, the front view of LIDAR point cloud and an RGB image as an input and generates a 3D object proposal in its first component using the bird’s eye view map and projects it to three views. A deep fusion network then combines region wise features obtained after ROI pooling for each view. The fused features are jointly used to predict object class and an oriented 3D box using regression. 

While existing work encodes LIDAR as a 3D grid or a front view map which later would require complex computations to extract features, this paper improvises the idea and represents LIDAR in a more compact way by projecting 3D point cloud to the bird’s eye view and the front view. Both the views share the encoded information regarding height, distance and density to form a much complete representation of information from LIDAR.  For detecting 3D images, instead of using front view map which has occlusion problem and loss of information regarding physical size, a bird’s eye view overcomes these challenges and also provides a better way obtaining accurate boxes around objects which doesn’t have high variance in the sense of vertical location. 
For region based fusion network, modal data of different resolution coming from different features are processed via ROI pooling to obtain vectors of equal lengths which later can be projected into the three respective views. Given the fusion features of the Multiview network and 3D proposals, 3D boxes and their orientation are generated using regression concentrating on the 8 corners of the 3d boxes. 

The implementation of the base network was built on 16-layer VGG net with some modifications in channels, pooling layers and addition of fully connected layers and up-sampling layers. 

## References
Multi-View 3D Object Detection Network for Autonomous Driving\
Xiaozhi Chen, Huimin Ma, Ji Wan, Bo Li, Tian Xia\
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017

## Analysis
Since the model is built on a well-established VGG layer, and it leverages the information from both LIDAR and RGB images and doesn’t stop at just 3D proposal from LIDAR, it further uses deep fusion which is better than early and late fusion to fuse features from different projections and runs a regression on the fused features to generate 3D representation of objects, I would regard this as an absolute strength of the model. Further, using this model and testing their results against the KITTI benchmark in predicting 2D object from the image and outperforming state-of-the-art models like Mono3D, 3DOP, VeloFCN would also prove the robustness and scalability of the model. The way they simplified representation of 3D boxes using 8-point corners instead of a 24D complex representation is also something to notice.

However, the metrics stated only reflect their performance on a single KITTI dataset which is used for autonomous driving, we cannot accept their statement that this model can be applied generically at all autonomous driving applications. Also, their dependence on multimodal data, in turn hardware, is high which doesn’t make this model applicable in all scenarios. Furthermore, no information of timing is mentioned which is a very important metric on object detection for autonomous driving. 

## Recommendations
Since the performance of the model against state-of-the-art techniques is verified and the results outperform the models by a fair margin, the adoption of the technique would be suggested. However, a more detailed understanding on the architecture and the required hardware and computation power should be thoroughly considered before adopting the model. Furthermore, since the verification of the model was done on a single dataset, more experiments should be performed on other datasets for the problem before applying this in 3D object detection for autonomous driving. Since, timing is one of the most important criteria for a model in such applications, it is important to get complete information regarding the same before adopting the model. Also, as the model suggests that the applications need not be only specific to autonomous driving but generic enough to apply to various other 3D object recognizing applications, a more detailed experiments must be conducted before adopting this model very specific to the application and its dynamics.

## Conclusion
The proposed model ranks better against the state-of-the-art techniques in 2D and 3D object recognition for autonomous driving given the discussed parameters. Its architecture optimization and application of various techniques like Multiview ROI pooling, deep fusion gives it an edge over the other competitive techniques. However, further information regarding some of the most important criteria to be considered for the application needs to be considered before concluding the adoptability of the technique. 

## Review of team's projects
### Future of Machine Learning Languages by Zhou Yuhao
#### Summary 
The author analysed and compared various machine learning languages like Python, Swift, Julia and Rust. He used various metrics for the comparison like learning difficulty, performance, libraries, parallel, mobile embedding and many more. On the whole it is a thorough research on the 4 ML languages
#### I Learned
That though python is still the most used ML language, its limitations are creating scope for development of many other ML languages and depending on the use case, moving to a different ML language like Julia or Swift can actually be considered.

### Snorkel: Another Alternative for Weak Supervision in Python by Yue Lie
#### Summary 
The author talks about the process of labelling a data using Snorkle which is a cost effective solution for a weak supervision ML problem
#### I Learned
I was able to understand the examples of unsupervised problems or rather weak supervised problem according to the user, and I also learned that snorkel is a good tool to label data cost effectively and mitigate the issue. Snorkle also has multiple use cases mentioned for different type of ML problems like information extraction, recommendation system etc.

### Difference between human and computer brain by Chen Linfeng
#### Summary
The author starts with identifying the difference between human brain and the current neural network focussing on the training of human brain and the neural network, the inputs coming from various sources and that the human brain can also do mistakes. He then concludes saying that we need to train the models not only using the APIs given by the companies, but also we need to build on top of them for better performance
#### I Learned
This was an exciting read, I learnt the difference of computer and human brain and that there is still a long way to go to mimic a human brain. Also the importance of building on top of the existing foundation for better performance.

### State of the art of objects segmentation methods based on machine learning algorithms by Zhan Bo
#### Summary
The author discusses various state of the art methods made by various people for object segmentation using algorithms like support vector machines, contour seed pairs learning based networks, orthogonal gamma distribution-based machine learning approach and many more
#### I learned
This review introduced me to object segmentation and the different approaches taken to do the same which were all alien to me before

### A Survey of dimensionality reduction techniques by Samyak Jain
#### Summary
The author extensively introduced many dimensionality reduction algorithms in practice and categorized them according to their shape of the function. He mainly talks about non-linear dimensionality techniques which can be applied on unsupervised problems and concludes that random projections is the best dimensionality reduction algorithm that can be applied on unsupervised problems. 
#### I Learned
The taxonomy of dimensionality reduction algorithms gave ma good perspective of the available algorithms and their application based on the input function. This also introduced me to random projections and the advantages of it applying it on unsupervised problems. It encourages me to use random projections in the future


