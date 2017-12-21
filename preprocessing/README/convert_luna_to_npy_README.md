# Preprocessing
```python
 > all_in_one.py = convert_luna_to_npy.py + h5py_patch_py
```

## convert_luna_to_npy.py
### 1. read_csv
* This is the code that reads `annotations.csv`
* `series_uid` is **key**
* **value** is the `coordinate value (x, y, z order) and diameter`
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

### 2. load_itk_image
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

### 3. resampling
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

### 4. normalize
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

### 5. zero centering
* `zero center` makes the average of the images zero. If you do this, your training will be better. (But sometimes it is better not to do it.)
* In LUNA16, use `0.25`
```python
def zero_center(image):
    PIXEL_MEAN = 0.25
    image = image - PIXEL_MEAN

    return image
```

### 6. create label
* This is the code that masks the `nodule`
* *example*
<div align="center">
   <img src="/assests/nodule.png" width="420">
  <img src="/assests/nodule_label.png"  width="420">
</div>

```python
def create_label(arr_shape, nodules, new_spacing, coord=False):
    """
        nodules = list of dict {'position', 'diameter'}
    """
    def _create_mask(arr_shape, position, diameter):

        z_dim, y_dim, x_dim = arr_shape
        z_pos, y_pos, x_pos = position

        z,y,x = np.ogrid[-z_pos:z_dim-z_pos, -y_pos:y_dim-y_pos, -x_pos:x_dim-x_pos]
        mask = z**2 + y**2 + x**2 <= int(diameter//2)**2

        return mask

    if coord:
        label = []
    else:
        label = np.zeros(arr_shape, dtype='bool')

    for nodule in nodules:
        worldCoord = nodule['position']
        worldCoord = np.asarray([worldCoord[2],worldCoord[1],worldCoord[0]])

        # new_spacing came from resample
        voxelCoord = compute_coord(worldCoord, origin, new_spacing)
        voxelCoord = [int(i) for i in voxelCoord]

        diameter = nodule['diameter']
        diameter = diameter / new_spacing[1]

        if coord:
            label.append(voxelCoord + [diameter])
        else:
            mask = _create_mask(arr_shape, voxelCoord, diameter)
            label = np.logical_or(label, mask)

    return label
```

### Next Step
* [h5py_patch](https://github.com/taki0112/CASED-Tensorflow/blob/master/preprocessing/README/h5py_patch_README.md)

## Author
Junho Kim
