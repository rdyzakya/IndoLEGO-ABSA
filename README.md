# Indo LEGO-ABSA

Indo LEGO-ABSA is an aspect-based sentiment analysis build on LEGO-ABSA framework for Indonesian language.

## Prepare

You can create a conda environment or use your own python system environment. We use Python 3.9.16 during the experiment.

```bash
conda create --name indo-lego-absa python=3.9.16
pip install -r requirements.txt
```

## Configurations

There are 4 configuration file in json format, located in the configs folder.

* na_config: Path(s) to the Non-ABSA dataset (csv format) used in training. Consist of two columns, "input" and "output".
* td_config: Containing configuration for the training data.
* vd_config: Containing configuration for the validation data.
* train_args: Training arguments, refer to https://huggingface.co/docs/transformers/main_classes/trainer for the details.

## Training

You can use train.py or simple_train.py for simplicity.

For the arguments needed, you can use the help command.

```bash
python train.py --help
```

If you use simple_train.py, wrap the arguments in args.json.

```bash
python simple_train.py args.json
```

## Inference

You can use the inference.ipynb notebook for inferencing.
