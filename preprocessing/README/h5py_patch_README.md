# Preprocessing
```python
 > all_in_one.py = convert_luna_to_npy.py + h5py_patch_py
```

## h5py_patch
* This code is used to train the image by extracting it by patch unit.

### 1. read npy file
* The reason for transpose is to make the coordinate directions `[x, y, z]`
```python
    image = np.transpose(np.load(mhd_to_npy))
    label = np.transpose(np.load(create_label_npy))
```

### 2. padding the image
```python
    offset = patch_size // 2

    non_pad = image
    non_label_pad = label

    non_pad = np.pad(non_pad, offset, 'constant', constant_values=np.min(non_pad))
    non_label_pad = np.pad(non_label_pad, offset, 'constant', constant_values=np.min(non_label_pad))

    image = np.pad(image, offset + (stride * move), 'constant', constant_values=np.min(image))
    label = np.pad(label, offset + (stride * move), 'constant', constant_values=np.min(label))

```

* The patch is ***centered on the nodule*** and then extracted.
* If the coordinates of the nodule are on the edge of the image, it is hard to extract the patch... so do padding
* `patch size` = 68, ie the size of the patch is `68 * 68 * 68` 
* `offset` = padding size

<img src="/assests/patch.png" width="30%">

* `stride` = It means how many spaces to move around the nodule
* It is possible to stride `8 times(4 * 2)` in the ***x, y, z, xy, xz, yz, and xyz*** directions respectively. (*Then, get 56 patch*)
* Therefore, It will be make the patch per single nodule point is 56 + 1 = `57`
* `stride`=8, `move`=4

![stride](/assests/stride.png)

### 3. get the coordinates of the nodule
```python
def world_2_voxel(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord
```

### 4. extract the patch that contains the nodule and any patches that are not included
* the number of non-nodule patch = 3 * the number of nodule patch
* For the label patch, resize the patch using `max-pooling`, since the output size of the network (U-net) is 8 * 8 * 8

### 5. create the patch with h5py
* See [link](https://www.safaribooksonline.com/library/view/python-and-hdf5/9781491944981/ch04.html) for reasons why I chose `lzf type`
```python
    with h5py.File(save_path + 'subset' + str(i) + '.h5', 'w') as hf:
        hf.create_dataset('nodule', data=nodule[:], compression='lzf')
        hf.create_dataset('label_nodule', data=nodule_label[:], compression='lzf')

        hf.create_dataset('non_nodule', data=non_nodule[:], compression='lzf')
        hf.create_dataset('label_non_nodule', data=non_nodule_label[:], compression='lzf')
```

## Author
Junho Kim / [@Lunit](http://lunit.io/)
