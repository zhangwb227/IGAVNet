## Improving Topological Continuity in Retinal Artery–Vein Segmentation via Graph Structure Modeling and Vector Encoding Loss

### 1. Motivation

<img src="fig1.svg" alt="fig1"/>

Fig. 1. Comparison of the sensitivity of different loss functions to misclassified vascular pixels in terms of spatial distribution and topological structure. Compared with existing loss functions, the proposed VE loss shows higher sensitivity to the spatial distribution of misclassified pixels and the topological consistency of vascular structures. Here, NDR  denotes the normalized difference, and Pred1 and Pred2 represent segmentation results with the same number of misclassified pixels but located at different spatial positions, while Pred1 and Pred3 represent segmentation results with topological structure errors.

### 2. Methods

 <img src="fig2.svg" alt="fig2"/>

Fig. 2. Overview of Our Method. (a) Overall architecture of IGAVNet. The skip connections between the encoder and decoder are equipped with the MSAM to suppress redundant features, while the VE loss is employed to constrain the consistency between predictions and ground truth. (b) Structure of the TCAM. The heterogeneous dual-branch design consists of a convolutional branch that activates local vascular features and a graph-modeling branch that constructs a vascular graph via the VSGM algorithm. Graph convolution is subsequently applied to enhance global vascular node representations. (c) Structure of the MSAM. A Softmax-based selection mechanism is applied along both channel and spatial dimensions to emphasize high-confidence vascular features while attenuating redundant nonvascular responses.

 <img src="fig3.svg" alt="fig3"/>

Fig. 3. VE Loss Structure Diagram. The vector encoding includes pixel distance weighting and class one-hot encoding. It calculates the cosine similarity between the predicted results and the labels, enhancing the model's sensitivity to error locations and thereby improving the accuracy of the model's segmentation of fine vessels.









