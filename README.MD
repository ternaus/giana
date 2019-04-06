# Code for the GIANA challenge based on the winning solutions for 2017, 2018

# Data
To obtain the dataset for this problem, you will need to contact the organizers of the Giana Challenge at
https://giana.grand-challenge.org/PolypDetection/

It will consist of the files:

 * Train: `cvcvideoclinicdbtrainvalid.rar`
 * Test: `cvcvideoclinicdbtestpart1.rar`, `cvcvideoclinicdbtestpart2.rar`
 
 
## Prepare train
Unpack the file `cvcvideoclinicdbtrainvalid.rar` to the `{data_path}` folder.

```bash
{data_path}/CVC-VideoClinicDBtrain_valid
    1
        1_1.png
        1_1_mask_png
        ...   
```

```bash
python -m dataset_utils.prepare_data -c configs/dataset.json
```

to convert the data to the structure

```bash
{data_path}/train
    1
        images
            1_1.png
         masks
            1_1.png
     ...
```
 