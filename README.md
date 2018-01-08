# Age-Gender-Predication
Predicate age and gender from a single face image

Pytorch implementation of CNN training for age and gender predication from a single face image.

Training Data: [the IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

## Dependencies
- Python 3.6+ (Anaconda)
- Pytorch-0.2.0 +
- scipy, numpy, sklearn etc.
- OpenCV3 (Python)

Tested on Ubuntu 14.04 LTS, Python 3.6 (Anaconda), Pytorch-0.3.0, CUDA 8.0, cuDNN 5.0

## Usage
### Data Preprocessing
det_MTCNN.py	

det_MTCNN_wiki.py	

imdbagesel.py	

wikiagesel.py

### Data loader
dataloaderimdb.py	

dataloaderimdbwiki.py	

dataloaderimdbwiki256.py	

dataloaderimdbwikiTest.py

dataloaderimdbwikiAgeG.py (Age and Gender multi-task Training)

### Model Training
TrainAgePre.py

TrainAgePre256.py	

TrainAgePreResNet18Det256.py	

TrainAgePreResNet256.py

TrainAgePreResNet256Cl.py

TrainAgePreResNet34Det256.py

TrainAgePreResNet34Det256OESM.py

TrainAgePreResNet34_256.py

TrainAgeRegression.py	

TrainAgeRegressionV2.py

TrainAgeGPreResNet34Det256.py	(Age and Gender multi-task Training, Recommended)

### Models
AgePreModel.py

AgePreModel256.py

AgePreModelResNet256.py

AgePreModelResNet256Cl.py

AgePreModelResNet34_256.py

AgePreModelV1.py

AgeGPreModelResNet34_256.py (Age and Gender multi-task Training, Recommended)

### Model evaluation
AgeEva.py

AgeEva256.py

AgeEvaResNet256.py

AgeEvaResNet256Cl.py

AgeEvaResNet34_256.py

AgeEvaResNet34_256_BI.py

AgeEvaV1.py

AgeEvaV2.py

AgeGEvaResNet34_256.py (Age and Gender multi-task training model)

