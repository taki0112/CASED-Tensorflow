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
* The patch is ***centered on the nodule*** and then extracted.
* `patch size` = 68, ie the size of the patch is `68 * 68 * 68` 
* `offset` = padding size

![patch](/assests/patch.png)

* `stride` = It means how many spaces to move around the nodule
* It is possible to stride `8 times(4 * 2)` in the ***x, y, z, xy, xz, yz, and xyz*** directions respectively. (*Then, get 56 patch*)
* Therefore, It will be make the patch per single nodule point is 56 + 1 = `57`

![stride](/assests/stride.png)

```python
    offset = patch_size // 2

    non_pad = image
    non_label_pad = label

    non_pad = np.pad(non_pad, offset, 'constant', constant_values=0)
    non_label_pad = np.pad(non_label_pad, offset, 'constant', constant_values=0)

    image = np.pad(image, offset + (stride * move), 'constant', constant_values=0)
    label = np.pad(label, offset + (stride * move), 'constant', constant_values=0)
```


