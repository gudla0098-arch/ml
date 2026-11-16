from pathlib import Path
import json
import tensorflow as tf
ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"chest_xray"; MODEL_DIR=ROOT/"models"; MODEL_DIR.mkdir(exist_ok=True)
SIZE=(224,224); BATCH=32; SEED=42
def main():
    train_dir,val_dir,test_dir=(DATA/"train",DATA/"val",DATA/"test")
    for folder in (train_dir,val_dir,test_dir):
        if not folder.exists(): raise SystemExit(f"Dataset folder missing: {folder}. See README.md.")
    train=tf.keras.utils.image_dataset_from_directory(train_dir,image_size=SIZE,batch_size=BATCH,seed=SEED,label_mode="binary")
    val=tf.keras.utils.image_dataset_from_directory(val_dir,image_size=SIZE,batch_size=BATCH,seed=SEED,label_mode="binary")
    test=tf.keras.utils.image_dataset_from_directory(test_dir,image_size=SIZE,batch_size=BATCH,shuffle=False,label_mode="binary")
    if train.class_names!=["NORMAL","PNEUMONIA"]: raise SystemExit("Expected class folders NORMAL and PNEUMONIA.")
    autotune=tf.data.AUTOTUNE; train=train.prefetch(autotune); val=val.prefetch(autotune); test=test.prefetch(autotune)
    augmentation=tf.keras.Sequential([tf.keras.layers.RandomRotation(.04),tf.keras.layers.RandomZoom(.08),tf.keras.layers.RandomContrast(.1)])
    base=tf.keras.applications.MobileNetV2(input_shape=(*SIZE,3),include_top=False,weights="imagenet"); base.trainable=False
    inputs=tf.keras.Input(shape=(*SIZE,3)); x=augmentation(inputs); x=tf.keras.applications.mobilenet_v2.preprocess_input(x*255.0)
    x=base(x,training=False); x=tf.keras.layers.GlobalAveragePooling2D()(x); x=tf.keras.layers.Dropout(.25)(x)
    outputs=tf.keras.layers.Dense(1,activation="sigmoid")(x); model=tf.keras.Model(inputs,outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="binary_crossentropy",metrics=["accuracy",tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True,monitor="val_loss"),tf.keras.callbacks.ModelCheckpoint(MODEL_DIR/"pneumonia_classifier.keras",save_best_only=True,monitor="val_loss")]
    model.fit(train,validation_data=val,epochs=12,callbacks=callbacks)
    metrics=model.evaluate(test,return_dict=True); (MODEL_DIR/"evaluation.json").write_text(json.dumps(metrics,indent=2)); print("Held-out test metrics:",metrics)
if __name__=="__main__": main()
