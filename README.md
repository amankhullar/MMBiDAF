# Multimodal Bidirectional Attention Flow (MMBiDAF)

Code for my Bachelor Thesis Project:

[Multimodal summarization and Beyond](https://amankhullar.github.io/data/Thesis_pdf.pdf).

![Model Architecture](https://github.com/amankhullar/amankhullar.github.io/blob/master/images/model_arch.png)

## Overview

Multimodal summarization is a superset of text summarization and is defined as the task of generating output summary (either abstractive or extractive) taking into account the different multimedia data as input. The output summary may be presented in single modality or multiple modalities. The ongoing research has proven that inclusion of audio and video elements as a part of the dataset may greatly improve the output summary. The output summary will be able to take into account the audio and the visual features along with text as input. 

The MultiModal Bidirectional Attention Flow (MMBiDAF) model (Figure) is the proposed models for carrying out the task of multimodal summarization. This model has been inspired from the various previous state of the art models existing in the literature. This model was chosen since it encompasses all the input modalities, calculates the similarity between them and then uses a multimodal
attention later on top of **image-aware** and **audio-aware** texts to get an output distribution over the source document. The model is used for **extractive summarization** in which at each timestep the most probable sentences are selected and chosen as part of the output summary. The summary terminates when the probability of a special (\<End Of Summary>) token is the greatest. The proposed model is inherently a combination of Bidirectional Attention Flow and Multimodal Attention models. 

Our model follows the high-level structure of :
1. Embedding Layer
2. Encoder Layer
3. Bidirectional Attention Layer
4. Modality Aware Sequence Sequence Modeling Layer
5. Multimodal Attention Layer 
6. Output Layer.

## Datasets

The code has been trained and evaluated on a custom Coursera Dataset comprising of Videos along with transcripts. The dataset comprises of 965 Coursera videos and the dataset has been split into 775 training videos, 95 videos in the validation set and 95 videos in the test set. 

## Usage

### Training
The code can be used on any other dataset by changing the path of the dataset in the train.py file. After fixing the *data path* and the *checkpoint path* in train.py run the command : 

```python
python train.py
```

### Hyper-parameters
* `text_embedding_size`: default = 300
* `audio_embedding_size`: default = 128
* `image_embedding_size`: default = 1000
* `hidden_size`: default = 100
* `drop_prob`: default = 0.2
* `num_epochs`: default = 90
* `batch_size`: default = 3

### Evaluation
For evaluation both Rouge score (Rouge1, Rouge2 and RougeL) are calculated and the F1 score is calculated. To evaluate the model on the validation set, provide the *checkpoint path* in the evaluate.py file. The run the command :

```python
python evaluate.py
```

## Acknowledgement
I would like to thank Crish Chute for open sourcing his [BiDAF](https://github.com/chrischute/squad) starter code which has been used as the base code for this model.

## License
MIT
