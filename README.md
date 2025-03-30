# AI Project: Speech Emotion Recognition

## Download and set up dataset

1. Download `dataset.zip` from our [Microsoft Teams](https://sutdapac.sharepoint.com/:u:/s/50.021AIProject/Ec2AnDYuETpLtoOz9oHT0YsBPxKvq2XlyNlyUkuKpfOPug?e=htQvlQ).
2. Extract `dataset.zip` within the `data/` folder. The project structure is as follows:

```
data
├── dataset
│   ├── crema-d
│   ├── esd
│   ├── jl-corpus
│   ├── meld
│   ├── mlend
│   ├── ravdess
│   ├── savee
│   ├── tess
├── Legacy
├── melspectrograms
.
.
.
```

3. Delete the `dataset.zip` if you wish.

## Opensmile static features

To retrieve the static features, go to teams and download the `combined_features.csv` file which is about 3.7GB.

Place it in the `data/features` folder, though you can change it, just make sure to change any path references accordingly.

In order to extract the train, validation and test dataframes, please run the `preprocess_features.py` script at the `data/feature-extraction/preprocess_features.py` path.
