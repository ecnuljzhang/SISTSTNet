# SISTSTNet: A New Script Independent Scene Text Style Transfer Network
The source code for 'SISTSTNet: A New Script Independent Scene Text Style Transfer Network' by Lingjun Zhang, Palaiahnakote Shivakumara, Lokesh Nandanwar, Umapada Pal, Yue Lu, Cheng-Lin Liu and Tong Lu. <br>
<img src="/pics/someresults.png" width="500px">
# Getting Start
## Libarary
| Package |	Version | Source |
| ------- | ------- | ------ |
| python | 3.7.11 | Conda |
| pip | 21.2.2 | Conda |
| pytorch | 1.10.1 | Conda |
| torchvision | 0.11.2 | Conda |
| catalyst | 20.4 | Pip |
| matplotlib | 3.5.1 | Conda |
| numpy | 1.21.2 | Conda |
| opencv | 4.4.0 | Conda |
| pillow | 6.2.2 | Conda |
| piq | 0.7.0 | Conda |
| scikit-image | 0.18.3 | Conda |
| scipy | 1.1.0 | Conda |
| tensorboard | 2.9.0 | Conda |

## Dataset
<img src="/pics/Proposed_dataset.png" width="500px">

### Folder Structure
The directory hierarchy is shown as follows: 
```
LITST-Dataset
|--- color masks
|          |--- color_1.jpg
|          |--- ...
|--- std_font
|          |--- English_1.jpg
|          |--- ...
|--- train
|          |--- font_style_1
|          |--- font_style_2
|                  |--- Hindi_1.jpg
|                  |--- ...
|          |--- ...
|--- valid
|          |--- font_style_3
|          |--- font_style_4
|                  |--- Hindi.jpg
|                  |--- ...
|          |--- ...
```
### Description
#### 'color_masks' folder 
It contains the color masks of the different solid and gradient colors used to produce the input source characters while training.
#### 'train' folder 
It contains the fonts styles for generating the source characters while training the model with the help of color masks.
#### 'test' folder 
It contains the fonts styles for generating the source characters while testing the model with the help of color masks.
#### 'std_font' folder 
It contains the standard target characters (Binary Image Standard font) used as input target in Target Encoder Network.

# How to run
## 1. Prepare dataset
Download proposed [dataset](https://drive.google.com/file/d/1K2evs9p3VLeKGWgPJkV-AahD1J5NOZLp/view?usp=sharing). 
## 2. Train
You can train on proposed dataset with the following code:
```
python train_proposed.py
```
Or you can also download the pretrained [model](https://drive.google.com/file/d/1XXwCE7tmMyELIKp4cRAo7b-DG0LN2I7z/view?usp=sharing).
## 3. Test
```
python test.py
```
You can define font_style_path and im_src, im_dist to get the generated characters for that particular font style.
