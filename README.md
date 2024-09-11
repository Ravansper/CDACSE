# CDACSE

### Title
《Large Language Models as Curriculum Data Generators for Unsupervised Sentence Representation》

### Training
For each existing model to be improved, you can use ChatGPT data from ./data or LLaMA data from ./data_llama.
### Training scripts
We provide training scripts `./run_unsup_example.sh`. Training scripts call `./train.py` for training. The script runs with the following code,
```python
bash run_unsup_example.sh
```
### Evaluation
Before evaluation, please download the evaluation datasets by running,
```python
cd SentEval/data/downstream/
bash download_dataset.sh
```
Our evaluation code for sentence embeddings is based on a modified version of [SentEval](https://github.com/facebookresearch/SentEval). It evaluates sentence embeddings on semantic textual similarity (STS) tasks and downstream transfer tasks. For STS tasks, our evaluation takes the "all" setting and reports Spearman's correlation. 
You can evaluate any transformers-based pre-trained models using our evaluation code. For example,
```python
python evaluation.py --model_name_or_path `Replace with your model or path` --pooler cls_before_pooler --task_set sts --mode test
```
