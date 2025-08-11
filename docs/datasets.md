
# Datasets

The datasets we use are put in the datasets/ folder.

### COCO

download the [COCO dataset](https://cocodataset.org/#home) and place them in the following way:

```
coco/
    train2017/
    val2017/
    annotations/
        instances_train2017.json 
        instances_val2017.json
        instances_train2017.1@5.0.json
        instances_val2017.1@17.0.json
        coco_cat_freq.json  // for multi-data balanced extraction(2 stage)
```

### LVIS

download the [LVIS dataset](https://www.lvisdataset.org/) and place them in the following way:

```
lvis_v1.0/
    train2017/  
    val2017/
    annotations/
        lvis_v0.5_train.json
        lvis_v0.5_val.json
        lvis_v0.5_train.1@5.0.json
        lvis_v0.5_val.1@4.0.json
```

Since some images in the lvis_v1.0 dataset are the same as those in the COCO dataset, you can use the COCO paths (such as train2017, val2017), but make sure to use the lvis annotations.

### Objects365 v2

download the [Objects365 v2 dataset](https://www.objects365.org/overview.html) and place them in the following way:

```
objects365/
    annotations/
        zhiyuan_objv2_train.json
        zhiyuan_objv2_train0-5.1@2.0.json   
        object365_cat_freq.json   // for multi-data balanced extraction(2 stage)
    images/
        v1/
            patch0/
            ...
            patch5/
```

## Extract and clean the datasets

extract random images for the training subset by:

```
python scripts/extract_random_images.py --seed 1 --percent 3.5 --ann data/objects365/annotations/zhiyuan_objv2_train.json
```

to avoid categories with no images in training, run the following script to statics this:

```
python scripts/get_cat_info.py --seed 1 --ann data/objects365/annotations/zhiyuan_objv2_train.1@3.5.json
```
this will create the file `data/objects365/annotations/zhiyuan_objv2_train.1@3.5_cat_info.json`


[Note]: As I only use the subsets for Objects365, it is not necessary to download the whole dataset. 