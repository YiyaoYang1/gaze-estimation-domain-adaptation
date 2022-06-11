# Domain Adaptation on Gaze Estimation
This work aims to realize unsupervised domain adaptation on gaze estimation from MPIIFaceGaze to ColumbiaGaze.  
If images are from different distribution, feature extractor will map them to different clusters in feature space.   
A conditional GAN is used to pull the clusters together.  


![7aa96c24eaa46fa1049430d04be8bea](https://user-images.githubusercontent.com/87518590/173186927-18a22587-8433-4ada-a5a8-3bf96ac8cbb5.png)
### cgan.py
>The feature  extractor in the original model works as a generator G(x), x represents input image.   
>An external MLP works as a discriminator D(x), to classify whether the extracted feature is from source domain or target domain, represented by one-hot encoding Y(x).  
>In each epoch, the discriminator is optimized first, the optimal target is to minimize ||D(G(x))-Y(x)|| for x in both domain.  
>Then the generator is optimized to confuse the discriminator, the optimal target is to minimize ||D(G(x))-Y'|| for x in target domain, Y' is the one-hot code for source domain.  
>In this way, images from target domain would be mapped to a cluster closer to the source domain's cluster in feature space.
### finetune.py
>The feature extractor parameters are frozen, the classifier is trained on source domain.  
>Since the feature extractor has been generalized, training on source domain can enhance performance on target domain.

![95c61169cc2a19d6a91ad0ede0a65dd](https://user-images.githubusercontent.com/87518590/173186910-fa266fa8-fd47-4591-9784-69955d840ee7.png)
