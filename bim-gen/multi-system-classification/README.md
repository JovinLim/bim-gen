The shape_classification script takes in obj(?) inputs and classifies them according to the different labels present.

### Data

TO BE CREATED

### Training from scratch

On each training run, we generate a random train/test split with 10 training inputs per-class.

To train the models on the **original** SHREC meshes, use

```python
python classification_shrec11.py --dataset_type=original --input_features=hks
```
or, with positional coordinates as features
```python
python classification_shrec11.py --dataset_type=original --input_features=xyz
```

And likewise, to train on the simplified meshes

```python
python classification_shrec11.py --dataset_type=simplified --input_features=hks
python classification_shrec11.py --dataset_type=simplified --input_features=xyz
```

There will be variance in the final accuracy, because the networks generally predict just 0-3 test models incorrectly, and the test split is randomized. Perform multiple runs to get a good sample!

**Note:** 

