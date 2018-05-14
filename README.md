# spellsorter_classification_dev

# Background

- code for `retrain.py` and `label_image.py` is from the TF tutorial https://www.tensorflow.org/tutorials/image_retraining

- the problem appears to be a matter of signature; both https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/inception_saved_model.py and https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py have the correct signatures
- `retrain.py` looks like an old-school way to do it, but it also has some nice features like resizing for input, bottleneck saves, auto-splitting the train/validations.
- i use the module `https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/1` for training
- total classes is around 18K, to be split across 25 or so models
- `train.py` is pulled from the snipptes of https://damienpontifex.com/2018/05/06/using-tensorflow-serving-to-host-our-retrained-image-classifier/ (who, is the author of the code that put in the incorrect signature into `retrain.py`)

# goals

- goal is to get `retrain.py` and `label_image.py` functionality working across gRPC via `tensorflow_model_server` serving
- either update `retrain.py` to have the propper signature, or get `train.py` (or some variant) working (which seems like the "new" way to do things). 
- should be able to consume model with https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/inception_client.py
- release code to OSS to make it easier for others to blackbox tensorflow.
- must be trainable (~700 classes per model) on digitalocean CPU hardware (i use 8GB 4vCPU high capacity droplet); ability to move to cloud would be a nice feature (but from using `retrain.py` has not seemed neccesary)

# future

- JSON REST API to aggregate requests to models from single client REST call, capture card images that failed, and queue requests; return card record from cards database if match found. 
- identify how to best serve ~25 models 
