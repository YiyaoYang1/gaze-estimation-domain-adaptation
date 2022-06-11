# Domain Adaptation on Gaze Estimation
This work aims to realize unsupervised domain adaptation on gaze estimation.  
The source domain is MPIIFaceGaze dataset.   
The target domain is ColumbiaGaze.  
If images are from different distribution, feature extractor will map them to different clusters in feature space. A conditional GAN is used to pull the clusters together.

![7aa96c24eaa46fa1049430d04be8bea](https://user-images.githubusercontent.com/87518590/173186927-18a22587-8433-4ada-a5a8-3bf96ac8cbb5.png)

The feature extractor parameters are frozen, the classifier is trained on source domain.

![95c61169cc2a19d6a91ad0ede0a65dd](https://user-images.githubusercontent.com/87518590/173186910-fa266fa8-fd47-4591-9784-69955d840ee7.png)
