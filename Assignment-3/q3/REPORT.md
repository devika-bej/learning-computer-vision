# Question 3

## Mask to Box Function
Approach for converting the mask to boxes:
- Create a color to label map by observing the dataset (color value was taken using a color picker to get exact value)
- For each label, create a black n white mask only for that feature
- Similar to question 2, get the bounding box using skimage functions label and region props, which look for connected components in images
- Store the annotations in a list of dict format, for each different box, for each image
- The dict is of the form {label: coordinates}

## Dataset Analysis
- Frequency:
    - Feet and hands are most frequent - most images would have more than one occurence of each
    - Legs, which could have been equally frequent, is not - both legs mostly get combined into a single bounding box/mask
    - Hair and torso are equally frequent, face slightly less - not necessary image has a face mask (back view)
- Width, Height, Aspect Ratio, and Area
    - For all the graphs, the distribution is very clearly left skewed, most bounding boxes are smaller in size
    - For both height and width the majority of the values lie between 0 and 100, however the decline is steeper in height - most boxes are smaller in size 
    - The majority of the aspect ratios are distributed between 0 and 2, with the peak at around 0.5
- Distribution of box positions
    - The scatter plot plots the centers of the bounding boxes, classwise
    - As expected it mimics the position in a typical human - hair, face, torso, hands on both sides, legs, feet on the bottom
- Challenges
    - The major challenge in the task would be in identifying the smaller bounding boxes, which are also very frequent.
    - Based on the visualization, these include the hands, feet, and perhaps face
    - The feet feature had two different colors for mask initially, this was handled by merging them into a single class

## Model Selection
- The model used is RetinaNet, some of the reasons are documented
    - Provides a balance between speed and accuracy - faster than FasterRCNN and more accurate than YOLO
    - Uses focal loss, which multiplies a balancing factor to the standard cross entropy loss to reduce the loss for confident classes - keeping the focus on tougher classes
    - This loss is useful in cases of imbalance in class frequencies, blurry images, and smaller bounding boxes
    - The efficiency in terms of speed comes from the one stage architecture of RetinaNet as compared to FasterRCNN (separate RPN)

## Model Implementation
- The RetinaNet model is intialised with the weights of ResNet50's Feature Pyramid Network (FPN)
- The last FC layer is adjusted to handle only 7 classes

## Training Hyperparameters and Behaviour
- Optimizer: Adam
- Learning Rate: 0.001
- Number of epochs: 15
- Time taken: ~6 minutes per epoch, total of ~1.5 hours

## Results

| Metric     | IoU threshold = 0.5  | IoU threshold = 0.6  | IoU threshold = 0.7  | IoU threshold = 0.8  |
|------------|----------------------|----------------------|----------------------|----------------------|
| mAP        | 0.5405               | 0.4885               | 0.3936               | 0.2514               |
| hair mAP   | 0.64                 | 0.57                 | 0.57                 | 0.57                 |
| face mAP   | 0.78                 | 0.78                 | 0.64                 | 0.5                  |
| torso mAP  | 0.28                 | 0.28                 | 0.28                 | 0.28                 |
| hands mAP  | 0.57                 | 0.43                 | 0.43                 | 0.43                 |
| legs mAP   | 0.5                  | 0.36                 | 0.36                 | 0.36                 |
| feet mAP   | 0.57                 | 0.28                 | 0.14                 | 0.14                 |

## Qualitative Analysis
- Performs best on torso and legs
- Fails to identify hands most times, and feet a lot of times
- Faces are often hidden in case of _almost_ backside viewpoint, and the model fails to pick up on them
- Torso sometimes gets doubly predicted
- Images where certain features are partially hidden or occluded also do not get identified, especially common in faces and hands

## Potential Improvements
- Adjust and choose best anchor sizes to be able to capture both small boxes for hands and feet and large boxes of torso and legs - the huge variation in sizes is a challenge
- Potential improvement from architecture perspective would be in the FPN to be able to extract a proper hierarchy in features

## Prompts used
https://chatgpt.com/share/67d04b63-3544-800a-9bba-a18d0f18a13f
https://claude.ai/share/e1521e29-2cd2-4500-a7f7-d507412fc957
