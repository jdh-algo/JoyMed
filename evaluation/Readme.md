
## ⚙️ Installation

### 1. Prepare nltk_data

To evaluate the report dataset, the nltk_data need to be downloaded.

``` python
import nltk
nltk.download('wordnet','/path/of/anaconda3/envs/Eval/nltk_data')
```

### 2. Prepare third party model

To evaluate the report dataset, the following models need to be downloaded.

``` sh
huggingface-cli download --resume-download --repo-type model Angelakeke/RaTE-NER-Deberta --local-dir third_party/RaTE-NER-Deberta
huggingface-cli download --resume-download --repo-type model FremyCompany/BioLORD-2023-C --local-dir third_party/BioLORD-2023-C
```

``` sh
cd third_party
git clone https://github.com/michelecafagna26/cider.git
cd cider
pip install .
pip install typing_extensions==4.14.0
```

## 📚 Datesets download

You can see the datasets we supported from eval_model.py.
Download the datasets and put them in /path/of/dataset.

### 1. CMMLU
``` sh
huggingface-cli download --resume-download --repo-type dataset lmlmcat/cmmlu --local-dir /path/of/dataset/CMMLU
cd /path/of/dataset/CMMLU && unzip cmmlu_v1_0_1.zip
```
### 2. OCR
``` sh
huggingface-cli download --resume-download --repo-type dataset jdh-algo/MedDocBench --local-dir /path/of/dataset
```
**open VLMEvalKit/.env and modify LMUData="/path/of/dataset"**


## 🚀 Eval

You can see the models we supported from eval_model.py.

**We use gpt-4o and gpt-4.1 as the judge model to evaluate the result. Please update OPENAI_API_URL and OPENAI_API_KEY in api_token.py**
### 1. eval JoyMed
``` sh
# model download
huggingface-cli download --resume-download --repo-type model jdh-algo/JoyMed --local-dir /path/of/model/JoyMed

#eval no_think
python eval_model.py --model_name 'JoyMed' --model_path "/path/of/model/JoyMed" --eval_datasets 'Medbullets_op4'

#eval think
python eval_model.py --model_name 'JoyMed' --model_path "/path/of/model/JoyMed" --eval_datasets 'Medbullets_op4' --prompt_version "./think_v5.json"

#eval auto
python eval_model.py --model_name 'JoyMed' --model_path "/path/of/model/JoyMed" --eval_datasets 'Medbullets_op4' --prompt_version "./auto_think_v1.json"

```
### 2. eval Qwen2.5-VL
```shell
python eval_model.py --model_name 'Qwen2.5-VL' --model_path "/path/of/model/Qwen2.5-VL-7B-Instruct" --eval_datasets 'Medbullets_op4' --model_size 7
python eval_model.py --model_name 'Qwen2.5-VL' --model_path "/path/of/model/Qwen2.5-VL-32B-Instruct" --eval_datasets 'Medbullets_op4' --model_size 32
```

## 🤝 Acknowledgments

We would like to thank the contributors to the [MedEvalKit](https://github.com/alibaba-damo-academy/MedEvalKit), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), [RaTE-NER-Deberta](https://huggingface.co/Angelakeke/RaTE-NER-Deberta), [BioLORD-2023-C](https://huggingface.co/FremyCompany/BioLORD-2023-C), and [cider](https://github.com/michelecafagna26/cider) repositories, for their open research and extraordinary work.
