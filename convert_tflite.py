import tensorflow as tf
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from src.factory import get_model
from src.generator import ImageSequence

@hydra.main(config_path="src/config.yaml")
def main(cfg):
    if cfg.wandb.project:
        import wandb
        from wandb.keras import WandbCallback
        wandb.init(project=cfg.wandb.project)
        callbacks = [WandbCallback()]
    else:
        callbacks = []

    csv_path = Path(to_absolute_path(__file__)).parent.joinpath("meta", f"{cfg.data.db}.csv")
    df = pd.read_csv(str(csv_path))
    train, val = train_test_split(df, random_state=42, test_size=0.1)
    train_gen = ImageSequence(cfg, train, "train")
    val_gen = ImageSequence(cfg, val, "val")

    #import pdb;pdb.set_trace()
    def representative_dataset_gen():
        for i in range(1000):
            # あなたが選択した関数内で、サンプル入力データをnumpy配列として取得する。
            yield [train_gen[i + 1][0][0][np.newaxis, :].astype('float32')]
            #yield [train_gen[0]]

    weight_file = '/home/ubuntu/projects/age-gender-estimation/checkpoint/MobileNet_224_weights.29-3.48.hdf5'
    weignt_file = '/home/ubuntu/projects/age-gender-estimation/checkpoint/MobileNetV2_224_weights.23-3.48.hdf5'
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
    model = get_model(cfg)
    model.load_weights(weight_file)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tfmodel = converter.convert()
    open ("/home/ubuntu/projects/age-gender-estimation/model.tflite" , "wb") .write(tfmodel)


if __name__ == '__main__':
    main()
