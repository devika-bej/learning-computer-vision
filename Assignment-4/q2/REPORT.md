# Question 2 Report

## Tabulated results

| Model                     | Test mIoU | Max Val mIoU |
|---------------------------|-----------|--------------|
| Vanilla UNet              | 0.8707    | 0.8779       |
| UNet (No Skip Connections)| 0.8045    | 0.8116       |
| Residual UNet             | 0.8763    | 0.8829       |
| Gated Residual UNet       | 0.8756    | 0.8827       |

## U-Net without Skip Connections

### Visual differences compared to Vanilla U-Net

- Vanilla U-Net is able to capture finer details better than the one without skip connections
- These details include the prediction for the gate/railing, shape of the tree leaves, better understanding of street lights, etc

### Importance of skip connections in U-Net

- During the decoder section of the U-Net, the skip connections re-input the finer details/feature maps from the encoder section which were lost while downsampling to high level features and then upsampling that limited information
- This allows for combination of high level features (object semantics, gotten from upsampling) and low level features (edges, details and textures, gotten from the encoder section) to generate precise segmentation predictions
- Another advantage is also that the reuse of features helps speed up the learning process since network does not have to relearn the low level features again in decoder section

## Gated Residual U-Net

### Advantages of attention gates according to the paper

- The attention gates use the gating signal (g) to filter features from skip connections (xl); the output at the end are the attention coefficients
- xl is multiplied by these coefficients so that the model focuses only on relevant features from skip connections and not all
- Thus the attention gates help improve performance by doing this filtering of features

### Visual differences compared to Vanilla U-Net

- For the given dataset, there is not much visual difference between the results of the two models
- This is also backed by the mIOU values on test data (Vanilla U-Net - 0.8707, Gated - 0.8756)
- However, on close observation we can see that the gated model is slightly more detailed than vanilla model - better prediction of street light, more detailed cars, able to identify the bars at the side of the street, etc