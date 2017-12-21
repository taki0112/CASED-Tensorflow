# Preprocessing
```python
 > all_in_one.py = convert_luna_to_npy.py + h5py_patch_py
```

## convert_luna_to_npy.py
* This is the code that reads `annotations.csv`
* `series_uid` is **key** and **value** is the `coordinate value (x, y, z order) and diameter`
```python
def read_csv(filename):
    lines = []
    with open(filename, 'r') as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)

    lines = lines[1:] # remove csv headers
    annotations_dict = {}
    for i in lines:
        series_uid, x, y, z, diameter = i
        value = {'position':[float(x),float(y),float(z)],
                 'diameter':float(diameter)}
        if series_uid in annotations_dict.keys():
            annotations_dict[series_uid].append(value)
        else:
            annotations_dict[series_uid] = [value]

    return annotations_dict
```

* This code converts `mhd file` to a `numpy array`
* The direction axis is set to `[z, y, x]`
```python
def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing    
```

* If you load all the luna16 data and make it into a numpy array... You do `resample`, `normalize` and `zero_centering`
* Each mhd file has different distances between the x, y, and z axes. (You might think this is because the machines that took the pictures are different)
* `resample` is to match the distances between the x, y, and z axes in all mhd files. 
* `OUTPUT_SPACING` is the distance mentioned above. (In now, `1.25mm`)
```python
def resample(image, org_spacing, new_spacing=OUTPUT_SPACING):

    resize_factor = org_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = org_spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing
```

* In `normalize`, Please check the `Hounsfield_Unit` in the table below.
* In LUNA16 nodule detection, uses `-1000 ~ 400`

![Hounsfield_Unit](http://i.imgur.com/4rlyReh.png)

```python
def normalize_planes(npzarray):

    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray
```

* `zero center` makes the average of the images zero. If you do this, your training will be better. (But sometimes it is better not to do it.)
* In LUNA16, use `0.25`
```python
def zero_center(image):
    PIXEL_MEAN = 0.25
    image = image - PIXEL_MEAN

    return image
```
