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

## OESM (online example selection methon)
TrainAgePreResNet34Det256OESM.py

```python
class OESM_CrossEntropy(nn.Module):
    def __init__(self, down_k=0.9, top_k=0.7):
        super(OESM_CrossEntropy, self).__init__()
        self.loss = nn.NLLLoss()
        self.down_k = down_k
        self.top_k = top_k
        self.softmax = nn.LogSoftmax()
        return
    def forward(self, input, target):
        softmax_result = self.softmax(input)
        loss = Variable(torch.Tensor(1).zero_())
        for idx, row in enumerate(softmax_result):
            gt = target[idx]
            pred = torch.unsqueeze(row, 0)
            cost = self.loss(pred, gt)
            loss = torch.cat((loss, cost.cpu()), 0)
        loss = loss[1:]
        loss_m = -loss
        if self.top_k == 1:
            valid_loss = loss
        index = torch.topk(loss_m, int(self.down_k * loss.size()[0]))
        loss = loss[index[1]]
        index = torch.topk(loss, int(self.top_k * loss.size()[0]))
        valid_loss = loss[index[1]]
        return torch.mean(valid_loss)
```
