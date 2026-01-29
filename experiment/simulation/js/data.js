// Transfer Learning CNN Simulation Data
// Pre-computed outputs from Jupyter notebooks

const EXPERIMENT_DATA = {
  "mobilenetv2_0": {
    "cells": [
      {
        "code": "# Importing Libraries\nimport os\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport tensorflow as tf\nimport seaborn as sns\nimport json\nfrom sklearn.metrics import classification_report\nfrom tensorflow.keras.applications.mobilenet_v2 import preprocess_input\nfrom tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout\nfrom tensorflow.keras.models import Model\nfrom tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint\nfrom tensorflow.keras.optimizers import Adam\nfrom tensorflow.keras.applications import MobileNetV2\n\nprint(\"All the required libraries are imported.\\n\")",
        "outputs": [
          "All the required libraries are imported.\n\n"
        ]
      },
      {
        "code": "# Defining Paths & Parameters\nbase_path = '/kaggle/input/oxford-102-flower-dataset/102 flower/flowers'\njson_path = '/kaggle/input/oxford-102-flower-dataset/102 flower/cat_to_name.json'\nimg_size = 224\nbatch_size = 32\nepochs = 150\npatience = 15\nlearning_rate = 1e-4\noutput_dir = './mobilenetv2_outputs'\nos.makedirs(output_dir, exist_ok=True)\n\nprint(f\"Parameters for training are set as:\")\nprint(f\"  Image Size: {img_size}\")\nprint(f\"  Batch Size: {batch_size}\")\nprint(f\"  Epochs: {epochs}\")\nprint(f\"  Patience: {patience}\")\nprint(f\"  Learning Rate: {learning_rate}\\n\")",
        "outputs": [
          "Parameters for training are set as:\n  Image Size: 224\n  Batch Size: 32\n  Epochs: 150\n  Patience: 15\n  Learning Rate: 0.0001\n\n"
        ]
      },
      {
        "code": "# Visualzing Dataset\nwith open(json_path, 'r') as f:\n    real_names = json.load(f)\n\nnumeric_classes = sorted(os.listdir(os.path.join(base_path, 'train')))\nnum_classes = len(numeric_classes)\nclass_names = [real_names[c] for c in numeric_classes]\n\nprint(\"\\n================ CLASS MAPPING TABLE ================\")\nprint(f\"Number of classes: {num_classes}\\n\")\nprint(f\"{'Folder':<10} | {'Flower Name'}\")\nprint(\"-\" * 50)\nfor f, cname in zip(numeric_classes, class_names):\n    print(f\"{f:<10} | {cname}\")\nprint(\"======================================================\\n\")\n\ntrain_raw = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"train\"),\n    image_size=(img_size, img_size),\n    batch_size=9,          # small batch for visualization\n    label_mode='int',\n    shuffle=True,\n    verbose=0)\n\n# Showing samples from dataset\ndef show_samples(ds_raw, numeric_class_list, name_map, n=9):\n    plt.figure(figsize=(8, 8))\n\n    for batch_x, batch_y in ds_raw.take(1):\n        imgs = batch_x.numpy().astype(\"uint8\")\n        labels = batch_y.numpy()\n        break\n\n    rows = int(np.ceil(np.sqrt(n)))\n    for i in range(n):\n        ax = plt.subplot(rows, rows, i+1)\n        plt.imshow(imgs[i])\n        folder_id = numeric_class_list[int(labels[i])]\n        cls_name = name_map[folder_id]\n        ax.set_title(cls_name, fontsize=9)\n        ax.axis(\"off\")\n\n    plt.tight_layout()\n    plt.show()\n\nprint(\"\\nShowing random sample images from the dataset...\\n\")\nshow_samples(train_raw, numeric_classes, real_names, n=9)",
        "outputs": [
          "\n================ CLASS MAPPING TABLE ================\nNumber of classes: 102\n\nFolder     | Flower Name\n--------------------------------------------------\n1          | pink primrose\n10         | globe thistle\n100        | blanket flower\n101        | trumpet creeper\n102        | blackberry lily\n11         | snapdragon\n12         | colt's foot\n13         | king protea\n14         | spear thistle\n15         | yellow iris\n16         | globe-flower\n17         | purple coneflower\n18         | peruvian lily\n19         | balloon flower\n2          | hard-leaved pocket orchid\n20         | giant white arum lily\n21         | fire lily\n22         | pincushion flower\n23         | fritillary\n24         | red ginger\n25         | grape hyacinth\n26         | corn poppy\n27         | prince of wales feathers\n28         | stemless gentian\n29         | artichoke\n3          | canterbury bells\n30         | sweet william\n31         | carnation\n32         | garden phlox\n33         | love in the mist\n34         | mexican aster\n35         | alpine sea holly\n36         | ruby-lipped cattleya\n37         | cape flower\n38         | great masterwort\n39         | siam tulip\n4          | sweet pea\n40         | lenten rose\n41         | barbeton daisy\n42         | daffodil\n43         | sword lily\n44         | poinsettia\n45         | bolero deep blue\n46         | wallflower\n47         | marigold\n48         | buttercup\n49         | oxeye daisy\n5          | english marigold\n50         | common dandelion\n51         | petunia\n52         | wild pansy\n53         | primula\n54         | sunflower\n55         | pelargonium\n56         | bishop of llandaff\n57         | gaura\n58         | geranium\n59         | orange dahlia\n6          | tiger lily\n60         | pink-yellow dahlia\n61         | cautleya spicata\n62         | japanese anemone\n63         | black-eyed susan\n64         | silverbush\n65         | californian poppy\n66         | osteospermum\n67         | spring crocus\n68         | bearded iris\n69         | windflower\n7          | moon orchid\n70         | tree poppy\n71         | gazania\n72         | azalea\n73         | water lily\n74         | rose\n75         | thorn apple\n76         | morning glory\n77         | passion flower\n78         | lotus lotus\n79         | toad lily\n8          | bird of paradise\n80         | anthurium\n81         | frangipani\n82         | clematis\n83         | hibiscus\n84         | columbine\n85         | desert-rose\n86         | tree mallow\n87         | magnolia\n88         | cyclamen\n89         | watercress\n9          | monkshood\n90         | canna lily\n91         | hippeastrum\n92         | bee balm\n93         | ball moss\n94         | foxglove\n95         | bougainvillea\n96         | camellia\n97         | mallow\n98         | mexican petunia\n99         | bromelia\n======================================================\n\n",
          "\nShowing random sample images from the dataset...\n\n"
        ]
      },
      {
        "code": "# Model Training\n\n# Loading dataset\ntrain_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"train\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=True)\n\nvalid_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"valid\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=True)\n\ntest_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"test\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=False)\n\n# Preprocess function\ndef preprocess(image, label):\n    image = preprocess_input(tf.cast(image, tf.float32))\n    label = tf.one_hot(label, num_classes)\n    return image, label\n\ntrain_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\nvalid_ds = valid_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\ntest_ds = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\n\n# Function to set trainable layers\ndef set_trainable_layers(base_model, unfreeze_percent):\n    total_layers = len(base_model.layers)\n    num_unfreeze = int(total_layers * unfreeze_percent)\n    \n    for i, layer in enumerate(base_model.layers):\n        if i < total_layers - num_unfreeze:\n            layer.trainable = False\n        else:\n            layer.trainable = True\n    \n    print(f\"Total layers in base model: {total_layers}\")\n    print(f\"Unfreeze percent: {unfreeze_percent*100:.1f}%\")\n    print(f\"No. of layers unfreeze: {num_unfreeze}\")\n    print(f\"Layers freeze: {total_layers - num_unfreeze}\")\n    if unfreeze_percent != 0:\n        print(\"\\nList of unfreeze layers:\")\n        for i, layer in enumerate(base_model.layers[-num_unfreeze:]):\n            print(f\"  {i + (total_layers - num_unfreeze)}: {layer.name}\")\n\n# Build model\nbase_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))\n\n# Unfreeze last 0% of layers\nset_trainable_layers(base_model, 0.00)\n\nx = GlobalAveragePooling2D()(base_model.output)\nx = Dropout(0.3)(x)\nx = Dense(512, activation='relu')(x)\nx = Dropout(0.3)(x)\noutput = Dense(num_classes, activation='softmax')(x)\nmodel = Model(base_model.input, output)\nprint(f\"Total layers in final model: {len(model.layers)}\")\n\ndef compact_model_summary(model, base_model_name=\"MobileNetV2\", custom_layer_start_idx=-5):\n    col1_width = 52\n    col2_width = 25\n    col3_width = 12\n\n    print(f\"{'Layer (type)':<{col1_width}} {'Output Shape':<{col2_width}} {'Param #':>{col3_width}}\")\n    print(\"=\" * (col1_width + col2_width + col3_width))\n\n    base_output = model.layers[custom_layer_start_idx - 1].output.shape\n    total_base_params = sum([l.count_params() for l in model.layers[:custom_layer_start_idx]])\n    print(f\"{base_model_name.title()} (Functional)\".ljust(col1_width), f\"{str(base_output):<{col2_width}}\", f\"{total_base_params:>{col3_width},}\")\n\n    for layer in model.layers[custom_layer_start_idx:]:\n        name = layer.name\n        layer_type = layer.__class__.__name__\n        try:\n            output_shape = str(layer.output.shape)\n        except:\n            output_shape = \"N/A\"\n        params = f\"{layer.count_params():,}\"\n        print(f\"{name + ' (' + layer_type + ')':<{col1_width}} {output_shape:<{col2_width}} {params:>{col3_width}}\")\n\n    print(\"=\" * (col1_width + col2_width + col3_width))\n\n    total_params = model.count_params()\n    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])\n    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])\n\n    print(f\"Total params: {total_params:,}\")\n    print(f\"Trainable params: {trainable_params:,}\")\n    print(f\"Non-trainable params: {non_trainable_params:,}\")\n\nprint(\"\\nPrinting Model Summary:\")\ncompact_model_summary(model, base_model_name=\"MobileNetV2\", custom_layer_start_idx=-5)\n\n# Compile model\nloss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)\n\nckpt_path = os.path.join(output_dir, \"best_model.h5\")\ncallbacks = [\n    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0),\n    ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss', verbose=0)\n]\n\nmodel.compile(optimizer=Adam(learning_rate), loss=loss_fn, metrics=['accuracy'])\n\n# Train model\nprint(\"\\nStarting training...\\n\")\nhistory = model.fit(\n    train_ds,\n    validation_data=valid_ds,\n    epochs=epochs,\n    callbacks=callbacks,\n    verbose=0)\nprint(\"\\nTraining completed.\")",
        "outputs": [
          "Found 6552 files belonging to 102 classes.\nFound 818 files belonging to 102 classes.\nFound 819 files belonging to 102 classes.\nDownloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5\n\u001b[1m9406464/9406464\u001b[0m \u001b[32m\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 0us/step\nTotal layers in base model: 154\nUnfreeze percent: 0.0%\nNo. of layers unfreeze: 0\nLayers freeze: 154\nTotal layers in final model: 159\n\nPrinting Model Summary:\nLayer (type)                                         Output Shape                   Param #\n=========================================================================================\nMobilenetv2 (Functional)                             (None, 7, 7, 1280)           2,257,984\nglobal_average_pooling2d (GlobalAveragePooling2D)    (None, 1280)                         0\ndropout (Dropout)                                    (None, 1280)                         0\ndense (Dense)                                        (None, 512)                    655,872\ndropout_1 (Dropout)                                  (None, 512)                          0\ndense_1 (Dense)                                      (None, 102)                     52,326\n=========================================================================================\nTotal params: 2,966,182\nTrainable params: 708,198\nNon-trainable params: 2,257,984\n\nStarting training...\n\n",
          "\nTraining completed.\n"
        ]
      },
      {
        "code": "# Model Evaluation\nprint(\"\\nLoading best model for evaluation...\")\nbest_model = tf.keras.models.load_model(ckpt_path)\n\nprint(\"\\nEvaluating on Train:\")\ntrain_loss, train_acc = best_model.evaluate(train_ds, verbose=0)\nprint(f\"Train Accuracy: {train_acc*100:.2f}%\")\n\nprint(\"\\nEvaluating on Validation:\")\nval_loss, val_acc = best_model.evaluate(valid_ds, verbose=0)\nprint(f\"Validation Accuracy: {val_acc*100:.2f}%\")\n\nprint(\"\\nEvaluating on Test:\")\ntest_loss, test_acc = best_model.evaluate(test_ds, verbose=0)\nprint(f\"Test Accuracy: {test_acc*100:.2f}%\")",
        "outputs": [
          "\nLoading best model for evaluation...\n",
          "\nEvaluating on Train:\nTrain Accuracy: 100.00%\n\nEvaluating on Validation:\nValidation Accuracy: 91.81%\n\nEvaluating on Test:\nTest Accuracy: 91.09%\n"
        ]
      },
      {
        "code": "# Visualizing Results\n\n# Training curves\ndef plot_training_curves(history):\n    acc = history.history[\"accuracy\"]\n    val_acc = history.history[\"val_accuracy\"]\n    loss = history.history[\"loss\"]\n    val_loss = history.history[\"val_loss\"]\n    epochs_range = range(1, len(acc) + 1)\n\n    plt.figure(figsize=(14, 5))\n\n    # Accuracy\n    plt.subplot(1, 2, 1)\n    plt.plot(epochs_range, acc, label=\"Train Acc\")\n    plt.plot(epochs_range, val_acc, label=\"Val Acc\")\n    plt.xlabel(\"Epoch\")\n    plt.ylabel(\"Accuracy\")\n    plt.title(\"Training vs Validation Accuracy\")\n    plt.legend()\n\n    # Loss\n    plt.subplot(1, 2, 2)\n    plt.plot(epochs_range, loss, label=\"Train Loss\")\n    plt.plot(epochs_range, val_loss, label=\"Val Loss\")\n    plt.xlabel(\"Epoch\")\n    plt.ylabel(\"Loss\")\n    plt.title(\"Training vs Validation Loss\")\n    plt.legend()\n\n    plt.tight_layout()\n    plt.show()\n\nprint(\"\\nPlotting training curves...\")\nplot_training_curves(history)\n\n# Classification report\ny_true = []\ny_pred = []\n\nfor x_batch, y_batch in test_ds:\n    preds = best_model.predict(x_batch, verbose=0)\n    y_pred.extend(np.argmax(preds, axis=1))\n    y_true.extend(np.argmax(y_batch.numpy(), axis=1))\n\ny_true = np.array(y_true)\ny_pred = np.array(y_pred)\n\nprint(\"\\nClassification Report:\")\nprint(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))",
        "outputs": [
          "\nPlotting training curves...\n",
          "\nClassification Report:\n                           precision    recall  f1-score   support\n\n            pink primrose       1.00      0.60      0.75         5\n            globe thistle       1.00      1.00      1.00         3\n           blanket flower       1.00      1.00      1.00         8\n          trumpet creeper       0.67      1.00      0.80         4\n          blackberry lily       1.00      1.00      1.00         6\n               snapdragon       0.75      0.67      0.71         9\n              colt's foot       1.00      0.89      0.94         9\n              king protea       0.83      0.83      0.83         6\n            spear thistle       1.00      1.00      1.00         3\n              yellow iris       1.00      1.00      1.00         4\n             globe-flower       1.00      1.00      1.00         3\n        purple coneflower       0.90      1.00      0.95         9\n            peruvian lily       0.86      1.00      0.92         6\n           balloon flower       1.00      0.71      0.83         7\nhard-leaved pocket orchid       1.00      1.00      1.00         5\n    giant white arum lily       1.00      0.67      0.80         3\n                fire lily       1.00      0.50      0.67         2\n        pincushion flower       0.80      1.00      0.89         4\n               fritillary       1.00      1.00      1.00         7\n               red ginger       1.00      1.00      1.00         2\n           grape hyacinth       1.00      1.00      1.00         5\n               corn poppy       1.00      0.80      0.89         5\n prince of wales feathers       1.00      1.00      1.00         3\n         stemless gentian       1.00      1.00      1.00         6\n                artichoke       0.89      0.89      0.89         9\n         canterbury bells       1.00      0.50      0.67         2\n            sweet william       1.00      1.00      1.00        14\n                carnation       0.67      1.00      0.80         2\n             garden phlox       1.00      0.67      0.80         6\n         love in the mist       1.00      0.88      0.93         8\n            mexican aster       1.00      1.00      1.00         5\n         alpine sea holly       1.00      1.00      1.00         6\n     ruby-lipped cattleya       1.00      0.86      0.92         7\n              cape flower       1.00      0.88      0.93         8\n         great masterwort       1.00      1.00      1.00         8\n               siam tulip       0.75      0.60      0.67         5\n                sweet pea       0.67      0.33      0.44         6\n              lenten rose       0.86      0.75      0.80         8\n           barbeton daisy       1.00      0.86      0.92        14\n                 daffodil       1.00      1.00      1.00         4\n               sword lily       0.82      0.88      0.85        16\n               poinsettia       1.00      1.00      1.00        11\n         bolero deep blue       1.00      0.67      0.80         3\n               wallflower       0.84      1.00      0.91        21\n                 marigold       1.00      1.00      1.00         3\n                buttercup       0.83      1.00      0.91         5\n              oxeye daisy       1.00      0.67      0.80         3\n         english marigold       1.00      1.00      1.00         4\n         common dandelion       0.89      1.00      0.94         8\n                  petunia       0.91      0.88      0.89        24\n               wild pansy       1.00      1.00      1.00         8\n                  primula       1.00      0.79      0.88        14\n                sunflower       1.00      1.00      1.00         4\n              pelargonium       0.88      1.00      0.93         7\n       bishop of llandaff       0.89      1.00      0.94         8\n                    gaura       0.91      0.91      0.91        11\n                 geranium       0.93      1.00      0.97        14\n            orange dahlia       1.00      1.00      1.00         7\n               tiger lily       1.00      0.89      0.94         9\n       pink-yellow dahlia       1.00      1.00      1.00        10\n         cautleya spicata       1.00      1.00      1.00         8\n         japanese anemone       0.50      0.50      0.50         4\n         black-eyed susan       1.00      1.00      1.00         4\n               silverbush       1.00      1.00      1.00         5\n        californian poppy       1.00      0.86      0.92         7\n             osteospermum       1.00      1.00      1.00         4\n            spring crocus       0.75      0.75      0.75         4\n             bearded iris       0.67      0.67      0.67         3\n               windflower       1.00      0.67      0.80         3\n              moon orchid       1.00      0.67      0.80         6\n               tree poppy       0.80      1.00      0.89         4\n                  gazania       1.00      1.00      1.00         9\n                   azalea       1.00      0.91      0.95        11\n               water lily       0.96      0.96      0.96        28\n                     rose       0.82      1.00      0.90        14\n              thorn apple       0.87      1.00      0.93        13\n            morning glory       1.00      1.00      1.00         4\n           passion flower       0.96      1.00      0.98        25\n              lotus lotus       0.93      1.00      0.97        14\n                toad lily       1.00      1.00      1.00         3\n         bird of paradise       0.91      1.00      0.95        10\n                anthurium       0.92      1.00      0.96        11\n               frangipani       0.93      1.00      0.96        13\n                 clematis       0.88      0.88      0.88        17\n                 hibiscus       0.78      1.00      0.88        14\n                columbine       0.80      0.80      0.80        10\n              desert-rose       0.70      0.70      0.70        10\n              tree mallow       1.00      1.00      1.00         5\n                 magnolia       0.86      1.00      0.92         6\n                 cyclamen       0.81      1.00      0.90        13\n               watercress       0.82      0.93      0.87        15\n                monkshood       0.67      1.00      0.80         2\n               canna lily       1.00      0.71      0.83        14\n              hippeastrum       1.00      1.00      1.00         8\n                 bee balm       1.00      0.91      0.95        11\n                ball moss       0.86      1.00      0.92         6\n                 foxglove       1.00      1.00      1.00        16\n            bougainvillea       0.93      0.93      0.93        14\n                 camellia       0.55      0.67      0.60         9\n                   mallow       0.50      0.40      0.44         5\n          mexican petunia       0.75      0.75      0.75         4\n                 bromelia       1.00      0.86      0.92         7\n\n                 accuracy                           0.91       819\n                macro avg       0.92      0.89      0.90       819\n             weighted avg       0.92      0.91      0.91       819\n\n"
        ]
      },
      {
        "code": "# not to be written in code cell (in ppt), for our reference only\n\nfrom sklearn.metrics import confusion_matrix\ncm = confusion_matrix(y_true, y_pred)\n\nplt.figure(figsize=(30, 30))\nsns.heatmap(cm, cmap=\"Blues\", annot=True, fmt='d',\n            xticklabels=class_names, yticklabels=class_names,)\nplt.xlabel(\"Predicted\")\nplt.ylabel(\"True\")\nplt.title(\"Confusion Matrix\")\nplt.tight_layout()\nplt.show()",
        "outputs": []
      }
    ]
  },
  "mobilenetv2_10": {
    "cells": [
      {
        "code": "# Importing Libraries\nimport os\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport tensorflow as tf\nimport seaborn as sns\nimport json\nfrom sklearn.metrics import classification_report\nfrom tensorflow.keras.applications.mobilenet_v2 import preprocess_input\nfrom tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout\nfrom tensorflow.keras.models import Model\nfrom tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint\nfrom tensorflow.keras.optimizers import Adam\nfrom tensorflow.keras.applications import MobileNetV2",
        "outputs": []
      },
      {
        "code": "# Defining Paths & Parameters\nbase_path = '/kaggle/input/oxford-102-flower-dataset/102 flower/flowers'\njson_path = '/kaggle/input/oxford-102-flower-dataset/102 flower/cat_to_name.json' \nimg_size = 224\nbatch_size = 32\nepochs = 150\npatience = 15\nlearning_rate = 1e-5\noutput_dir = './mobilenetv2_outputs'\nos.makedirs(output_dir, exist_ok=True)\n\nprint(f\"Parameters for training are set as:\")\nprint(f\"  Image Size: {img_size}\")\nprint(f\"  Batch Size: {batch_size}\")\nprint(f\"  Epochs: {epochs}\")\nprint(f\"  Patience: {patience}\")\nprint(f\"  Learning Rate: {learning_rate}\\n\")",
        "outputs": [
          "Parameters for training are set as:\n  Image Size: 224\n  Batch Size: 32\n  Epochs: 150\n  Patience: 15\n  Learning Rate: 1e-05\n\n"
        ]
      },
      {
        "code": "# Visualizing Dataset\nwith open(json_path, 'r') as f:\n    real_names = json.load(f)\n\nnumeric_classes = sorted(os.listdir(os.path.join(base_path, 'train')))\nnum_classes = len(numeric_classes)\nclass_names = [real_names[c] for c in numeric_classes]\n\nprint(\"\\n================ CLASS MAPPING TABLE ================\")\nprint(f\"Number of classes: {num_classes}\\n\")\nprint(f\"{'Folder':<10} | {'Flower Name'}\")\nprint(\"-\" * 50)\nfor f, cname in zip(numeric_classes, class_names):\n    print(f\"{f:<10} | {cname}\")\nprint(\"======================================================\\n\")\n\ntrain_raw = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"train\"),\n    image_size=(img_size, img_size),\n    batch_size=9,          # small batch for visualization\n    label_mode='int',\n    shuffle=True,\n    verbose=0)\n\n# Showing samples from dataset\ndef show_samples(ds_raw, numeric_class_list, name_map, n=9):\n    plt.figure(figsize=(8, 8))\n\n    for batch_x, batch_y in ds_raw.take(1):\n        imgs = batch_x.numpy().astype(\"uint8\")\n        labels = batch_y.numpy()\n        break\n\n    rows = int(np.ceil(np.sqrt(n)))\n    for i in range(n):\n        ax = plt.subplot(rows, rows, i+1)\n        plt.imshow(imgs[i])\n        folder_id = numeric_class_list[int(labels[i])]\n        cls_name = name_map[folder_id]\n        ax.set_title(cls_name, fontsize=9)\n        ax.axis(\"off\")\n\n    plt.tight_layout()\n    plt.show()\n\nprint(\"\\nShowing random sample images from the dataset...\\n\")\nshow_samples(train_raw, numeric_classes, real_names, n=9)",
        "outputs": [
          "\n================ CLASS MAPPING TABLE ================\nNumber of classes: 102\n\nFolder     | Flower Name\n--------------------------------------------------\n1          | pink primrose\n10         | globe thistle\n100        | blanket flower\n101        | trumpet creeper\n102        | blackberry lily\n11         | snapdragon\n12         | colt's foot\n13         | king protea\n14         | spear thistle\n15         | yellow iris\n16         | globe-flower\n17         | purple coneflower\n18         | peruvian lily\n19         | balloon flower\n2          | hard-leaved pocket orchid\n20         | giant white arum lily\n21         | fire lily\n22         | pincushion flower\n23         | fritillary\n24         | red ginger\n25         | grape hyacinth\n26         | corn poppy\n27         | prince of wales feathers\n28         | stemless gentian\n29         | artichoke\n3          | canterbury bells\n30         | sweet william\n31         | carnation\n32         | garden phlox\n33         | love in the mist\n34         | mexican aster\n35         | alpine sea holly\n36         | ruby-lipped cattleya\n37         | cape flower\n38         | great masterwort\n39         | siam tulip\n4          | sweet pea\n40         | lenten rose\n41         | barbeton daisy\n42         | daffodil\n43         | sword lily\n44         | poinsettia\n45         | bolero deep blue\n46         | wallflower\n47         | marigold\n48         | buttercup\n49         | oxeye daisy\n5          | english marigold\n50         | common dandelion\n51         | petunia\n52         | wild pansy\n53         | primula\n54         | sunflower\n55         | pelargonium\n56         | bishop of llandaff\n57         | gaura\n58         | geranium\n59         | orange dahlia\n6          | tiger lily\n60         | pink-yellow dahlia\n61         | cautleya spicata\n62         | japanese anemone\n63         | black-eyed susan\n64         | silverbush\n65         | californian poppy\n66         | osteospermum\n67         | spring crocus\n68         | bearded iris\n69         | windflower\n7          | moon orchid\n70         | tree poppy\n71         | gazania\n72         | azalea\n73         | water lily\n74         | rose\n75         | thorn apple\n76         | morning glory\n77         | passion flower\n78         | lotus lotus\n79         | toad lily\n8          | bird of paradise\n80         | anthurium\n81         | frangipani\n82         | clematis\n83         | hibiscus\n84         | columbine\n85         | desert-rose\n86         | tree mallow\n87         | magnolia\n88         | cyclamen\n89         | watercress\n9          | monkshood\n90         | canna lily\n91         | hippeastrum\n92         | bee balm\n93         | ball moss\n94         | foxglove\n95         | bougainvillea\n96         | camellia\n97         | mallow\n98         | mexican petunia\n99         | bromelia\n======================================================\n\n",
          "\nShowing random sample images from the dataset...\n\n"
        ]
      },
      {
        "code": "# Model Training\n\n# Loading dataset\ntrain_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"train\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=True)\n\nvalid_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"valid\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=True)\n\ntest_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"test\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=False)\n\n# Preprocess function\ndef preprocess(image, label):\n    image = preprocess_input(tf.cast(image, tf.float32))\n    label = tf.one_hot(label, num_classes)\n    return image, label\n\ntrain_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\nvalid_ds = valid_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\ntest_ds = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\n\n# Function to set trainable layers\ndef set_trainable_layers(base_model, unfreeze_percent):\n    total_layers = len(base_model.layers)\n    num_unfreeze = int(total_layers * unfreeze_percent)\n    \n    for i, layer in enumerate(base_model.layers):\n        if i < total_layers - num_unfreeze:\n            layer.trainable = False\n        else:\n            layer.trainable = True\n    \n    print(f\"Total layers in base model: {total_layers}\")\n    print(f\"Unfreeze percent: {unfreeze_percent*100:.1f}%\")\n    print(f\"No. of layers unfreeze: {num_unfreeze}\")\n    print(f\"Layers freeze: {total_layers - num_unfreeze}\")\n    if unfreeze_percent != 0:\n        print(\"\\nList of unfreeze layers:\")\n        for i, layer in enumerate(base_model.layers[-num_unfreeze:]):\n            print(f\"  {i + (total_layers - num_unfreeze)}: {layer.name}\")\n\n# Build model\nbase_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))\n\n# Unfreeze last 10% of layers\nset_trainable_layers(base_model, 0.10)\n\nx = GlobalAveragePooling2D()(base_model.output)\nx = Dropout(0.3)(x)\nx = Dense(512, activation='relu')(x)\nx = Dropout(0.3)(x)\noutput = Dense(num_classes, activation='softmax')(x)\nmodel = Model(base_model.input, output)\nprint(f\"Total layers in final model: {len(model.layers)}\")\n\ndef compact_model_summary(model, base_model_name=\"MobileNetV2\", custom_layer_start_idx=-5):\n    col1_width = 52\n    col2_width = 25\n    col3_width = 12\n\n    print(f\"{'Layer (type)':<{col1_width}} {'Output Shape':<{col2_width}} {'Param #':>{col3_width}}\")\n    print(\"=\" * (col1_width + col2_width + col3_width))\n\n    base_output = model.layers[custom_layer_start_idx - 1].output.shape\n    total_base_params = sum([l.count_params() for l in model.layers[:custom_layer_start_idx]])\n    print(f\"{base_model_name.title()} (Functional)\".ljust(col1_width), f\"{str(base_output):<{col2_width}}\", f\"{total_base_params:>{col3_width},}\")\n\n    for layer in model.layers[custom_layer_start_idx:]:\n        name = layer.name\n        layer_type = layer.__class__.__name__\n        try:\n            output_shape = str(layer.output.shape)\n        except:\n            output_shape = \"N/A\"\n        params = f\"{layer.count_params():,}\"\n        print(f\"{name + ' (' + layer_type + ')':<{col1_width}} {output_shape:<{col2_width}} {params:>{col3_width}}\")\n\n    print(\"=\" * (col1_width + col2_width + col3_width))\n\n    total_params = model.count_params()\n    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])\n    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])\n\n    print(f\"Total params: {total_params:,}\")\n    print(f\"Trainable params: {trainable_params:,}\")\n    print(f\"Non-trainable params: {non_trainable_params:,}\")\n\nprint(\"\\nPrinting Model Summary:\")\ncompact_model_summary(model, base_model_name=\"MobileNetV2\", custom_layer_start_idx=-5)\n\n# Compile model\nloss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)\n\nckpt_path = os.path.join(output_dir, \"best_model.h5\")\ncallbacks = [\n    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0),\n    ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss', verbose=0)\n]\n\nmodel.compile(optimizer=Adam(learning_rate), loss=loss_fn, metrics=['accuracy'])\n\n# Train model\nprint(\"\\nStarting training...\\n\")\nhistory = model.fit(\n    train_ds,\n    validation_data=valid_ds,\n    epochs=epochs,\n    callbacks=callbacks,\n    verbose=0)\nprint(\"\\nTraining completed.\")",
        "outputs": [
          "Found 6552 files belonging to 102 classes.\nFound 818 files belonging to 102 classes.\nFound 819 files belonging to 102 classes.\nDownloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5\n\u001b[1m9406464/9406464\u001b[0m \u001b[32m\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 0us/step\nTotal layers in base model: 154\nUnfreeze percent: 10.0%\nNo. of layers unfreeze: 15\nLayers freeze: 139\n\nList of unfreeze layers:\n  139: block_15_depthwise_relu\n  140: block_15_project\n  141: block_15_project_BN\n  142: block_15_add\n  143: block_16_expand\n  144: block_16_expand_BN\n  145: block_16_expand_relu\n  146: block_16_depthwise\n  147: block_16_depthwise_BN\n  148: block_16_depthwise_relu\n  149: block_16_project\n  150: block_16_project_BN\n  151: Conv_1\n  152: Conv_1_bn\n  153: out_relu\nTotal layers in final model: 159\n\nPrinting Model Summary:\nLayer (type)                                         Output Shape                   Param #\n=========================================================================================\nMobilenetv2 (Functional)                             (None, 7, 7, 1280)           2,257,984\nglobal_average_pooling2d (GlobalAveragePooling2D)    (None, 1280)                         0\ndropout (Dropout)                                    (None, 1280)                         0\ndense (Dense)                                        (None, 512)                    655,872\ndropout_1 (Dropout)                                  (None, 512)                          0\ndense_1 (Dense)                                      (None, 102)                     52,326\n=========================================================================================\nTotal params: 2,966,182\nTrainable params: 1,748,198\nNon-trainable params: 1,217,984\n\nStarting training...\n\n",
          "\nTraining completed.\n"
        ]
      },
      {
        "code": "# Model Evaluation\nprint(\"\\nLoading best model for evaluation...\")\nbest_model = tf.keras.models.load_model(ckpt_path)\n\nprint(\"\\nEvaluating on Train:\")\ntrain_loss, train_acc = best_model.evaluate(train_ds, verbose=0)\nprint(f\"Train Accuracy: {train_acc*100:.2f}%\")\n\nprint(\"\\nEvaluating on Validation:\")\nval_loss, val_acc = best_model.evaluate(valid_ds, verbose=0)\nprint(f\"Validation Accuracy: {val_acc*100:.2f}%\")\n\nprint(\"\\nEvaluating on Test:\")\ntest_loss, test_acc = best_model.evaluate(test_ds, verbose=0)\nprint(f\"Test Accuracy: {test_acc*100:.2f}%\")",
        "outputs": [
          "\nLoading best model for evaluation...\n",
          "\nEvaluating on Train:\nTrain Accuracy: 100.00%\n\nEvaluating on Validation:\nValidation Accuracy: 94.99%\n\nEvaluating on Test:\nTest Accuracy: 94.87%\n"
        ]
      },
      {
        "code": "# Visualizing Results\n\n# Training curves\ndef plot_training_curves(history):\n    acc = history.history[\"accuracy\"]\n    val_acc = history.history[\"val_accuracy\"]\n    loss = history.history[\"loss\"]\n    val_loss = history.history[\"val_loss\"]\n    epochs_range = range(1, len(acc) + 1)\n\n    plt.figure(figsize=(14, 5))\n\n    # Accuracy\n    plt.subplot(1, 2, 1)\n    plt.plot(epochs_range, acc, label=\"Train Acc\")\n    plt.plot(epochs_range, val_acc, label=\"Val Acc\")\n    plt.xlabel(\"Epoch\")\n    plt.ylabel(\"Accuracy\")\n    plt.title(\"Training vs Validation Accuracy\")\n    plt.legend()\n\n    # Loss\n    plt.subplot(1, 2, 2)\n    plt.plot(epochs_range, loss, label=\"Train Loss\")\n    plt.plot(epochs_range, val_loss, label=\"Val Loss\")\n    plt.xlabel(\"Epoch\")\n    plt.ylabel(\"Loss\")\n    plt.title(\"Training vs Validation Loss\")\n    plt.legend()\n\n    plt.tight_layout()\n    plt.show()\n\nprint(\"\\nPlotting training curves...\")\nplot_training_curves(history)\n\n# Classification report\ny_true = []\ny_pred = []\n\nfor x_batch, y_batch in test_ds:\n    preds = best_model.predict(x_batch, verbose=0)\n    y_pred.extend(np.argmax(preds, axis=1))\n    y_true.extend(np.argmax(y_batch.numpy(), axis=1))\n\ny_true = np.array(y_true)\ny_pred = np.array(y_pred)\n\nprint(\"\\nClassification Report:\")\nprint(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))",
        "outputs": [
          "\nPlotting training curves...\n",
          "\nClassification Report:\n                           precision    recall  f1-score   support\n\n            pink primrose       1.00      1.00      1.00         5\n            globe thistle       1.00      1.00      1.00         3\n           blanket flower       1.00      1.00      1.00         8\n          trumpet creeper       0.80      1.00      0.89         4\n          blackberry lily       1.00      1.00      1.00         6\n               snapdragon       1.00      0.67      0.80         9\n              colt's foot       1.00      0.89      0.94         9\n              king protea       0.86      1.00      0.92         6\n            spear thistle       1.00      1.00      1.00         3\n              yellow iris       1.00      1.00      1.00         4\n             globe-flower       1.00      0.67      0.80         3\n        purple coneflower       1.00      1.00      1.00         9\n            peruvian lily       1.00      1.00      1.00         6\n           balloon flower       1.00      0.86      0.92         7\nhard-leaved pocket orchid       1.00      1.00      1.00         5\n    giant white arum lily       1.00      0.67      0.80         3\n                fire lily       1.00      1.00      1.00         2\n        pincushion flower       0.80      1.00      0.89         4\n               fritillary       0.88      1.00      0.93         7\n               red ginger       1.00      1.00      1.00         2\n           grape hyacinth       1.00      1.00      1.00         5\n               corn poppy       1.00      0.80      0.89         5\n prince of wales feathers       1.00      1.00      1.00         3\n         stemless gentian       1.00      1.00      1.00         6\n                artichoke       1.00      0.89      0.94         9\n         canterbury bells       1.00      1.00      1.00         2\n            sweet william       1.00      1.00      1.00        14\n                carnation       1.00      1.00      1.00         2\n             garden phlox       1.00      0.67      0.80         6\n         love in the mist       1.00      1.00      1.00         8\n            mexican aster       1.00      1.00      1.00         5\n         alpine sea holly       1.00      1.00      1.00         6\n     ruby-lipped cattleya       1.00      1.00      1.00         7\n              cape flower       1.00      0.88      0.93         8\n         great masterwort       1.00      1.00      1.00         8\n               siam tulip       1.00      0.60      0.75         5\n                sweet pea       0.83      0.83      0.83         6\n              lenten rose       0.88      0.88      0.88         8\n           barbeton daisy       1.00      1.00      1.00        14\n                 daffodil       1.00      1.00      1.00         4\n               sword lily       0.89      1.00      0.94        16\n               poinsettia       1.00      1.00      1.00        11\n         bolero deep blue       1.00      1.00      1.00         3\n               wallflower       0.95      1.00      0.98        21\n                 marigold       1.00      1.00      1.00         3\n                buttercup       1.00      1.00      1.00         5\n              oxeye daisy       1.00      1.00      1.00         3\n         english marigold       1.00      1.00      1.00         4\n         common dandelion       0.89      1.00      0.94         8\n                  petunia       0.96      0.92      0.94        24\n               wild pansy       0.88      0.88      0.88         8\n                  primula       1.00      0.86      0.92        14\n                sunflower       1.00      1.00      1.00         4\n              pelargonium       1.00      1.00      1.00         7\n       bishop of llandaff       1.00      1.00      1.00         8\n                    gaura       1.00      0.91      0.95        11\n                 geranium       0.88      1.00      0.93        14\n            orange dahlia       1.00      1.00      1.00         7\n               tiger lily       1.00      0.89      0.94         9\n       pink-yellow dahlia       1.00      1.00      1.00        10\n         cautleya spicata       1.00      1.00      1.00         8\n         japanese anemone       1.00      1.00      1.00         4\n         black-eyed susan       1.00      1.00      1.00         4\n               silverbush       1.00      1.00      1.00         5\n        californian poppy       1.00      1.00      1.00         7\n             osteospermum       1.00      1.00      1.00         4\n            spring crocus       1.00      1.00      1.00         4\n             bearded iris       1.00      1.00      1.00         3\n               windflower       1.00      0.67      0.80         3\n              moon orchid       1.00      0.67      0.80         6\n               tree poppy       1.00      1.00      1.00         4\n                  gazania       1.00      1.00      1.00         9\n                   azalea       1.00      1.00      1.00        11\n               water lily       1.00      1.00      1.00        28\n                     rose       0.82      1.00      0.90        14\n              thorn apple       0.81      1.00      0.90        13\n            morning glory       1.00      1.00      1.00         4\n           passion flower       1.00      1.00      1.00        25\n              lotus lotus       0.93      1.00      0.97        14\n                toad lily       1.00      1.00      1.00         3\n         bird of paradise       0.83      1.00      0.91        10\n                anthurium       1.00      1.00      1.00        11\n               frangipani       0.93      1.00      0.96        13\n                 clematis       0.94      0.94      0.94        17\n                 hibiscus       0.74      1.00      0.85        14\n                columbine       0.89      0.80      0.84        10\n              desert-rose       0.90      0.90      0.90        10\n              tree mallow       1.00      1.00      1.00         5\n                 magnolia       0.86      1.00      0.92         6\n                 cyclamen       1.00      1.00      1.00        13\n               watercress       0.93      0.87      0.90        15\n                monkshood       0.67      1.00      0.80         2\n               canna lily       1.00      0.79      0.88        14\n              hippeastrum       0.89      1.00      0.94         8\n                 bee balm       1.00      0.91      0.95        11\n                ball moss       0.86      1.00      0.92         6\n                 foxglove       1.00      0.94      0.97        16\n            bougainvillea       0.93      1.00      0.97        14\n                 camellia       0.86      0.67      0.75         9\n                   mallow       0.67      0.80      0.73         5\n          mexican petunia       1.00      0.75      0.86         4\n                 bromelia       0.88      1.00      0.93         7\n\n                 accuracy                           0.95       819\n                macro avg       0.96      0.95      0.95       819\n             weighted avg       0.95      0.95      0.95       819\n\n"
        ]
      },
      {
        "code": "# not to be written in code cell (in ppt), for our reference only\n\nfrom sklearn.metrics import confusion_matrix\ncm = confusion_matrix(y_true, y_pred)\n\nplt.figure(figsize=(30, 30))\nsns.heatmap(cm, cmap=\"Blues\", annot=True, fmt='d',\n            xticklabels=class_names, yticklabels=class_names,)\nplt.xlabel(\"Predicted\")\nplt.ylabel(\"True\")\nplt.title(\"Confusion Matrix\")\nplt.tight_layout()\nplt.show()",
        "outputs": []
      }
    ]
  },
  "mobilenetv2_20": {
    "cells": [
      {
        "code": "# Importing Libraries\nimport os\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport tensorflow as tf\nimport seaborn as sns\nimport json\nfrom sklearn.metrics import classification_report\nfrom tensorflow.keras.applications.mobilenet_v2 import preprocess_input\nfrom tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout\nfrom tensorflow.keras.models import Model\nfrom tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint\nfrom tensorflow.keras.optimizers import Adam\nfrom tensorflow.keras.applications import MobileNetV2",
        "outputs": []
      },
      {
        "code": "# Defining Paths & Parameters\nbase_path = '/kaggle/input/oxford-102-flower-dataset/102 flower/flowers'\njson_path = '/kaggle/input/oxford-102-flower-dataset/102 flower/cat_to_name.json'\nimg_size = 224\nbatch_size = 32\nepochs = 150\npatience = 15\nlearning_rate = 1e-5\noutput_dir = './mobilenetv2_outputs'\nos.makedirs(output_dir, exist_ok=True)\n\nprint(f\"Parameters for training are set as:\")\nprint(f\"  Image Size: {img_size}\")\nprint(f\"  Batch Size: {batch_size}\")\nprint(f\"  Epochs: {epochs}\")\nprint(f\"  Patience: {patience}\")\nprint(f\"  Learning Rate: {learning_rate}\\n\")",
        "outputs": [
          "Parameters for training are set as:\n  Image Size: 224\n  Batch Size: 32\n  Epochs: 150\n  Patience: 15\n  Learning Rate: 1e-05\n\n"
        ]
      },
      {
        "code": "# Visualizing Dataset\nwith open(json_path, 'r') as f:\n    real_names = json.load(f)\n\nnumeric_classes = sorted(os.listdir(os.path.join(base_path, 'train')))\nnum_classes = len(numeric_classes)\nclass_names = [real_names[c] for c in numeric_classes]\n\nprint(\"\\n================ CLASS MAPPING TABLE ================\")\nprint(f\"Number of classes: {num_classes}\\n\")\nprint(f\"{'Folder':<10} | {'Flower Name'}\")\nprint(\"-\" * 50)\nfor f, cname in zip(numeric_classes, class_names):\n    print(f\"{f:<10} | {cname}\")\nprint(\"======================================================\\n\")\n\ntrain_raw = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"train\"),\n    image_size=(img_size, img_size),\n    batch_size=9,          # small batch for visualization\n    label_mode='int',\n    shuffle=True,\n    verbose=0)\n\n# Showing samples from dataset\ndef show_samples(ds_raw, numeric_class_list, name_map, n=9):\n    plt.figure(figsize=(8, 8))\n\n    for batch_x, batch_y in ds_raw.take(1):\n        imgs = batch_x.numpy().astype(\"uint8\")\n        labels = batch_y.numpy()\n        break\n\n    rows = int(np.ceil(np.sqrt(n)))\n    for i in range(n):\n        ax = plt.subplot(rows, rows, i+1)\n        plt.imshow(imgs[i])\n        folder_id = numeric_class_list[int(labels[i])]\n        cls_name = name_map[folder_id]\n        ax.set_title(cls_name, fontsize=9)\n        ax.axis(\"off\")\n\n    plt.tight_layout()\n    plt.show()\n\nprint(\"\\nShowing random sample images from the dataset...\\n\")\nshow_samples(train_raw, numeric_classes, real_names, n=9)",
        "outputs": [
          "\n================ CLASS MAPPING TABLE ================\nNumber of classes: 102\n\nFolder     | Flower Name\n--------------------------------------------------\n1          | pink primrose\n10         | globe thistle\n100        | blanket flower\n101        | trumpet creeper\n102        | blackberry lily\n11         | snapdragon\n12         | colt's foot\n13         | king protea\n14         | spear thistle\n15         | yellow iris\n16         | globe-flower\n17         | purple coneflower\n18         | peruvian lily\n19         | balloon flower\n2          | hard-leaved pocket orchid\n20         | giant white arum lily\n21         | fire lily\n22         | pincushion flower\n23         | fritillary\n24         | red ginger\n25         | grape hyacinth\n26         | corn poppy\n27         | prince of wales feathers\n28         | stemless gentian\n29         | artichoke\n3          | canterbury bells\n30         | sweet william\n31         | carnation\n32         | garden phlox\n33         | love in the mist\n34         | mexican aster\n35         | alpine sea holly\n36         | ruby-lipped cattleya\n37         | cape flower\n38         | great masterwort\n39         | siam tulip\n4          | sweet pea\n40         | lenten rose\n41         | barbeton daisy\n42         | daffodil\n43         | sword lily\n44         | poinsettia\n45         | bolero deep blue\n46         | wallflower\n47         | marigold\n48         | buttercup\n49         | oxeye daisy\n5          | english marigold\n50         | common dandelion\n51         | petunia\n52         | wild pansy\n53         | primula\n54         | sunflower\n55         | pelargonium\n56         | bishop of llandaff\n57         | gaura\n58         | geranium\n59         | orange dahlia\n6          | tiger lily\n60         | pink-yellow dahlia\n61         | cautleya spicata\n62         | japanese anemone\n63         | black-eyed susan\n64         | silverbush\n65         | californian poppy\n66         | osteospermum\n67         | spring crocus\n68         | bearded iris\n69         | windflower\n7          | moon orchid\n70         | tree poppy\n71         | gazania\n72         | azalea\n73         | water lily\n74         | rose\n75         | thorn apple\n76         | morning glory\n77         | passion flower\n78         | lotus lotus\n79         | toad lily\n8          | bird of paradise\n80         | anthurium\n81         | frangipani\n82         | clematis\n83         | hibiscus\n84         | columbine\n85         | desert-rose\n86         | tree mallow\n87         | magnolia\n88         | cyclamen\n89         | watercress\n9          | monkshood\n90         | canna lily\n91         | hippeastrum\n92         | bee balm\n93         | ball moss\n94         | foxglove\n95         | bougainvillea\n96         | camellia\n97         | mallow\n98         | mexican petunia\n99         | bromelia\n======================================================\n\n",
          "\nShowing random sample images from the dataset...\n\n"
        ]
      },
      {
        "code": "# Model Training\n\n# Loading dataset\ntrain_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"train\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=True)\n\nvalid_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"valid\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=True)\n\ntest_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"test\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=False)\n\n# Preprocess function\ndef preprocess(image, label):\n    image = preprocess_input(tf.cast(image, tf.float32))\n    label = tf.one_hot(label, num_classes)\n    return image, label\n\ntrain_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\nvalid_ds = valid_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\ntest_ds = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\n\n# Function to set trainable layers\ndef set_trainable_layers(base_model, unfreeze_percent):\n    total_layers = len(base_model.layers)\n    num_unfreeze = int(total_layers * unfreeze_percent)\n    \n    for i, layer in enumerate(base_model.layers):\n        if i < total_layers - num_unfreeze:\n            layer.trainable = False\n        else:\n            layer.trainable = True\n    \n    print(f\"Total layers in base model: {total_layers}\")\n    print(f\"Unfreeze percent: {unfreeze_percent*100:.1f}%\")\n    print(f\"No. of layers unfreeze: {num_unfreeze}\")\n    print(f\"Layers freeze: {total_layers - num_unfreeze}\")\n    if unfreeze_percent != 0:\n        print(\"\\nList of unfreeze layers:\")\n        for i, layer in enumerate(base_model.layers[-num_unfreeze:]):\n            print(f\"  {i + (total_layers - num_unfreeze)}: {layer.name}\")\n\n# Build model\nbase_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))\n\n# Unfreeze last 20% of layers\nset_trainable_layers(base_model, 0.20)\n\nx = GlobalAveragePooling2D()(base_model.output)\nx = Dropout(0.3)(x)\nx = Dense(512, activation='relu')(x)\nx = Dropout(0.3)(x)\noutput = Dense(num_classes, activation='softmax')(x)\nmodel = Model(base_model.input, output)\nprint(f\"Total layers in final model: {len(model.layers)}\")\n\ndef compact_model_summary(model, base_model_name=\"MobileNetV2\", custom_layer_start_idx=-5):\n    col1_width = 52\n    col2_width = 25\n    col3_width = 12\n\n    print(f\"{'Layer (type)':<{col1_width}} {'Output Shape':<{col2_width}} {'Param #':>{col3_width}}\")\n    print(\"=\" * (col1_width + col2_width + col3_width))\n\n    base_output = model.layers[custom_layer_start_idx - 1].output.shape\n    total_base_params = sum([l.count_params() for l in model.layers[:custom_layer_start_idx]])\n    print(f\"{base_model_name.title()} (Functional)\".ljust(col1_width), f\"{str(base_output):<{col2_width}}\", f\"{total_base_params:>{col3_width},}\")\n\n    for layer in model.layers[custom_layer_start_idx:]:\n        name = layer.name\n        layer_type = layer.__class__.__name__\n        try:\n            output_shape = str(layer.output.shape)\n        except:\n            output_shape = \"N/A\"\n        params = f\"{layer.count_params():,}\"\n        print(f\"{name + ' (' + layer_type + ')':<{col1_width}} {output_shape:<{col2_width}} {params:>{col3_width}}\")\n\n    print(\"=\" * (col1_width + col2_width + col3_width))\n\n    total_params = model.count_params()\n    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])\n    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])\n\n    print(f\"Total params: {total_params:,}\")\n    print(f\"Trainable params: {trainable_params:,}\")\n    print(f\"Non-trainable params: {non_trainable_params:,}\")\n\nprint(\"\\nPrinting Model Summary:\")\ncompact_model_summary(model, base_model_name=\"MobileNetV2\", custom_layer_start_idx=-5)\n\n# Compile model\nloss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)\n\nckpt_path = os.path.join(output_dir, \"best_model.h5\")\ncallbacks = [\n    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0),\n    ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss', verbose=0)\n]\n\nmodel.compile(optimizer=Adam(learning_rate), loss=loss_fn, metrics=['accuracy'])\n\n# Train model\nprint(\"\\nStarting training...\\n\")\nhistory = model.fit(\n    train_ds,\n    validation_data=valid_ds,\n    epochs=epochs,\n    callbacks=callbacks,\n    verbose=0)\nprint(\"\\nTraining completed.\")",
        "outputs": [
          "Found 6552 files belonging to 102 classes.\nFound 818 files belonging to 102 classes.\nFound 819 files belonging to 102 classes.\nDownloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5\n\u001b[1m9406464/9406464\u001b[0m \u001b[32m\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 0us/step\nTotal layers in base model: 154\nUnfreeze percent: 20.0%\nNo. of layers unfreeze: 30\nLayers freeze: 124\n\nList of unfreeze layers:\n  124: block_13_project_BN\n  125: block_14_expand\n  126: block_14_expand_BN\n  127: block_14_expand_relu\n  128: block_14_depthwise\n  129: block_14_depthwise_BN\n  130: block_14_depthwise_relu\n  131: block_14_project\n  132: block_14_project_BN\n  133: block_14_add\n  134: block_15_expand\n  135: block_15_expand_BN\n  136: block_15_expand_relu\n  137: block_15_depthwise\n  138: block_15_depthwise_BN\n  139: block_15_depthwise_relu\n  140: block_15_project\n  141: block_15_project_BN\n  142: block_15_add\n  143: block_16_expand\n  144: block_16_expand_BN\n  145: block_16_expand_relu\n  146: block_16_depthwise\n  147: block_16_depthwise_BN\n  148: block_16_depthwise_relu\n  149: block_16_project\n  150: block_16_project_BN\n  151: Conv_1\n  152: Conv_1_bn\n  153: out_relu\nTotal layers in final model: 159\n\nPrinting Model Summary:\nLayer (type)                                         Output Shape                   Param #\n=========================================================================================\nMobilenetv2 (Functional)                             (None, 7, 7, 1280)           2,257,984\nglobal_average_pooling2d (GlobalAveragePooling2D)    (None, 1280)                         0\ndropout (Dropout)                                    (None, 1280)                         0\ndense (Dense)                                        (None, 512)                    655,872\ndropout_1 (Dropout)                                  (None, 512)                          0\ndense_1 (Dense)                                      (None, 102)                     52,326\n=========================================================================================\nTotal params: 2,966,182\nTrainable params: 2,234,598\nNon-trainable params: 731,584\n\nStarting training...\n\n",
          "\nTraining completed.\n"
        ]
      },
      {
        "code": "# Model Evaluation\nprint(\"\\nLoading best model for evaluation...\")\nbest_model = tf.keras.models.load_model(ckpt_path)\n\nprint(\"\\nEvaluating on Train:\")\ntrain_loss, train_acc = best_model.evaluate(train_ds, verbose=0)\nprint(f\"Train Accuracy: {train_acc*100:.2f}%\")\n\nprint(\"\\nEvaluating on Validation:\")\nval_loss, val_acc = best_model.evaluate(valid_ds, verbose=0)\nprint(f\"Validation Accuracy: {val_acc*100:.2f}%\")\n\nprint(\"\\nEvaluating on Test:\")\ntest_loss, test_acc = best_model.evaluate(test_ds, verbose=0)\nprint(f\"Test Accuracy: {test_acc*100:.2f}%\")",
        "outputs": [
          "\nLoading best model for evaluation...\n",
          "\nEvaluating on Train:\nTrain Accuracy: 100.00%\n\nEvaluating on Validation:\nValidation Accuracy: 96.33%\n\nEvaluating on Test:\nTest Accuracy: 95.48%\n"
        ]
      },
      {
        "code": "# Visualizing Results\n\n# Training curves\ndef plot_training_curves(history):\n    acc = history.history[\"accuracy\"]\n    val_acc = history.history[\"val_accuracy\"]\n    loss = history.history[\"loss\"]\n    val_loss = history.history[\"val_loss\"]\n    epochs_range = range(1, len(acc) + 1)\n\n    plt.figure(figsize=(14, 5))\n\n    # Accuracy\n    plt.subplot(1, 2, 1)\n    plt.plot(epochs_range, acc, label=\"Train Acc\")\n    plt.plot(epochs_range, val_acc, label=\"Val Acc\")\n    plt.xlabel(\"Epoch\")\n    plt.ylabel(\"Accuracy\")\n    plt.title(\"Training vs Validation Accuracy\")\n    plt.legend()\n\n    # Loss\n    plt.subplot(1, 2, 2)\n    plt.plot(epochs_range, loss, label=\"Train Loss\")\n    plt.plot(epochs_range, val_loss, label=\"Val Loss\")\n    plt.xlabel(\"Epoch\")\n    plt.ylabel(\"Loss\")\n    plt.title(\"Training vs Validation Loss\")\n    plt.legend()\n\n    plt.tight_layout()\n    plt.show()\n\nprint(\"\\nPlotting training curves...\")\nplot_training_curves(history)\n\n# Classification report\ny_true = []\ny_pred = []\n\nfor x_batch, y_batch in test_ds:\n    preds = best_model.predict(x_batch, verbose=0)\n    y_pred.extend(np.argmax(preds, axis=1))\n    y_true.extend(np.argmax(y_batch.numpy(), axis=1))\n\ny_true = np.array(y_true)\ny_pred = np.array(y_pred)\n\nprint(\"\\nClassification Report:\")\nprint(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))",
        "outputs": [
          "\nPlotting training curves...\n",
          "\nClassification Report:\n                           precision    recall  f1-score   support\n\n            pink primrose       1.00      1.00      1.00         5\n            globe thistle       1.00      1.00      1.00         3\n           blanket flower       1.00      1.00      1.00         8\n          trumpet creeper       1.00      0.75      0.86         4\n          blackberry lily       1.00      1.00      1.00         6\n               snapdragon       1.00      0.89      0.94         9\n              colt's foot       1.00      0.89      0.94         9\n              king protea       1.00      1.00      1.00         6\n            spear thistle       1.00      1.00      1.00         3\n              yellow iris       1.00      1.00      1.00         4\n             globe-flower       0.75      1.00      0.86         3\n        purple coneflower       1.00      1.00      1.00         9\n            peruvian lily       1.00      1.00      1.00         6\n           balloon flower       1.00      0.86      0.92         7\nhard-leaved pocket orchid       1.00      1.00      1.00         5\n    giant white arum lily       1.00      0.67      0.80         3\n                fire lily       1.00      1.00      1.00         2\n        pincushion flower       0.80      1.00      0.89         4\n               fritillary       1.00      1.00      1.00         7\n               red ginger       1.00      1.00      1.00         2\n           grape hyacinth       1.00      1.00      1.00         5\n               corn poppy       1.00      0.80      0.89         5\n prince of wales feathers       1.00      1.00      1.00         3\n         stemless gentian       1.00      1.00      1.00         6\n                artichoke       0.90      1.00      0.95         9\n         canterbury bells       1.00      1.00      1.00         2\n            sweet william       1.00      1.00      1.00        14\n                carnation       1.00      1.00      1.00         2\n             garden phlox       1.00      0.83      0.91         6\n         love in the mist       1.00      1.00      1.00         8\n            mexican aster       1.00      1.00      1.00         5\n         alpine sea holly       1.00      1.00      1.00         6\n     ruby-lipped cattleya       1.00      1.00      1.00         7\n              cape flower       1.00      1.00      1.00         8\n         great masterwort       1.00      1.00      1.00         8\n               siam tulip       1.00      0.80      0.89         5\n                sweet pea       0.67      0.67      0.67         6\n              lenten rose       0.86      0.75      0.80         8\n           barbeton daisy       1.00      1.00      1.00        14\n                 daffodil       1.00      1.00      1.00         4\n               sword lily       0.94      0.94      0.94        16\n               poinsettia       1.00      1.00      1.00        11\n         bolero deep blue       1.00      1.00      1.00         3\n               wallflower       0.95      1.00      0.98        21\n                 marigold       1.00      1.00      1.00         3\n                buttercup       1.00      1.00      1.00         5\n              oxeye daisy       1.00      1.00      1.00         3\n         english marigold       1.00      0.75      0.86         4\n         common dandelion       0.80      1.00      0.89         8\n                  petunia       0.95      0.88      0.91        24\n               wild pansy       1.00      1.00      1.00         8\n                  primula       1.00      0.86      0.92        14\n                sunflower       1.00      1.00      1.00         4\n              pelargonium       0.88      1.00      0.93         7\n       bishop of llandaff       1.00      1.00      1.00         8\n                    gaura       1.00      0.91      0.95        11\n                 geranium       0.93      1.00      0.97        14\n            orange dahlia       1.00      1.00      1.00         7\n               tiger lily       1.00      0.89      0.94         9\n       pink-yellow dahlia       1.00      1.00      1.00        10\n         cautleya spicata       1.00      1.00      1.00         8\n         japanese anemone       0.80      1.00      0.89         4\n         black-eyed susan       1.00      1.00      1.00         4\n               silverbush       1.00      1.00      1.00         5\n        californian poppy       1.00      1.00      1.00         7\n             osteospermum       1.00      1.00      1.00         4\n            spring crocus       1.00      1.00      1.00         4\n             bearded iris       1.00      1.00      1.00         3\n               windflower       0.75      1.00      0.86         3\n              moon orchid       1.00      0.83      0.91         6\n               tree poppy       1.00      1.00      1.00         4\n                  gazania       1.00      1.00      1.00         9\n                   azalea       1.00      1.00      1.00        11\n               water lily       1.00      1.00      1.00        28\n                     rose       0.93      0.93      0.93        14\n              thorn apple       0.87      1.00      0.93        13\n            morning glory       1.00      1.00      1.00         4\n           passion flower       1.00      1.00      1.00        25\n              lotus lotus       0.93      1.00      0.97        14\n                toad lily       1.00      1.00      1.00         3\n         bird of paradise       0.91      1.00      0.95        10\n                anthurium       0.92      1.00      0.96        11\n               frangipani       1.00      1.00      1.00        13\n                 clematis       0.89      1.00      0.94        17\n                 hibiscus       0.87      0.93      0.90        14\n                columbine       0.89      0.80      0.84        10\n              desert-rose       0.75      0.90      0.82        10\n              tree mallow       1.00      1.00      1.00         5\n                 magnolia       0.86      1.00      0.92         6\n                 cyclamen       0.93      1.00      0.96        13\n               watercress       0.94      1.00      0.97        15\n                monkshood       0.67      1.00      0.80         2\n               canna lily       0.91      0.71      0.80        14\n              hippeastrum       1.00      1.00      1.00         8\n                 bee balm       1.00      0.91      0.95        11\n                ball moss       1.00      1.00      1.00         6\n                 foxglove       1.00      0.94      0.97        16\n            bougainvillea       0.93      1.00      0.97        14\n                 camellia       1.00      0.67      0.80         9\n                   mallow       0.71      1.00      0.83         5\n          mexican petunia       1.00      0.75      0.86         4\n                 bromelia       1.00      1.00      1.00         7\n\n                 accuracy                           0.95       819\n                macro avg       0.96      0.96      0.95       819\n             weighted avg       0.96      0.95      0.95       819\n\n"
        ]
      },
      {
        "code": "# not to be written in code cell (in ppt), for our reference only\n\nfrom sklearn.metrics import confusion_matrix\ncm = confusion_matrix(y_true, y_pred)\n\nplt.figure(figsize=(30, 30))\nsns.heatmap(cm, cmap=\"Blues\", annot=True, fmt='d',\n            xticklabels=class_names, yticklabels=class_names,)\nplt.xlabel(\"Predicted\")\nplt.ylabel(\"True\")\nplt.title(\"Confusion Matrix\")\nplt.tight_layout()\nplt.show()",
        "outputs": []
      }
    ]
  },
  "vgg19_0": {
    "cells": [
      {
        "code": "# Importing Libraries\nimport os\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport tensorflow as tf\nimport seaborn as sns\nimport json\nfrom sklearn.metrics import classification_report\nfrom tensorflow.keras.applications.vgg19 import preprocess_input\nfrom tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout\nfrom tensorflow.keras.models import Model\nfrom tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint\nfrom tensorflow.keras.optimizers import Adam\nfrom tensorflow.keras.applications import VGG19\n\nprint(\"All the required libraries are imported.\\n\")",
        "outputs": [
          "All the required libraries are imported.\n\n"
        ]
      },
      {
        "code": "# Defining Paths & Parameters\nbase_path = '/kaggle/input/102 flower/flowers'\njson_path = '/kaggle/input/102 flower/cat_to_name.json'\nimg_size = 224\nbatch_size = 32\nepochs = 150\npatience = 15\nlearning_rate = 1e-4\noutput_dir = './vgg19_outputs'\nos.makedirs(output_dir, exist_ok=True)\n\nprint(f\"Parameters for training are set as:\")\nprint(f\"  Image Size: {img_size}\")\nprint(f\"  Batch Size: {batch_size}\")\nprint(f\"  Epochs: {epochs}\")\nprint(f\"  Patience: {patience}\")\nprint(f\"  Learning Rate: {learning_rate}\\n\")",
        "outputs": [
          "Parameters for training are set as:\n  Image Size: 224\n  Batch Size: 32\n  Epochs: 150\n  Patience: 15\n  Learning Rate: 0.0001\n\n"
        ]
      },
      {
        "code": "# Visualzing Dataset\nwith open(json_path, 'r') as f:\n    real_names = json.load(f)\n\nnumeric_classes = sorted(os.listdir(os.path.join(base_path, 'train')))\nnum_classes = len(numeric_classes)\nclass_names = [real_names[c] for c in numeric_classes]\n\nprint(\"\\n================ CLASS MAPPING TABLE ================\")\nprint(f\"Number of classes: {num_classes}\\n\")\nprint(f\"{'Folder':<10} | {'Flower Name'}\")\nprint(\"-\" * 50)\nfor f, cname in zip(numeric_classes, class_names):\n    print(f\"{f:<10} | {cname}\")\nprint(\"======================================================\\n\")\n\ntrain_raw = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"train\"),\n    image_size=(img_size, img_size),\n    batch_size=9,          # small batch for visualization\n    label_mode='int',\n    shuffle=True,\n    verbose=0)\n\n# Showing samples from dataset\ndef show_samples(ds_raw, numeric_class_list, name_map, n=9):\n    plt.figure(figsize=(8, 8))\n\n    for batch_x, batch_y in ds_raw.take(1):\n        imgs = batch_x.numpy().astype(\"uint8\")\n        labels = batch_y.numpy()\n        break\n\n    rows = int(np.ceil(np.sqrt(n)))\n    for i in range(n):\n        ax = plt.subplot(rows, rows, i+1)\n        plt.imshow(imgs[i])\n        folder_id = numeric_class_list[int(labels[i])]\n        cls_name = name_map[folder_id]\n        ax.set_title(cls_name, fontsize=9)\n        ax.axis(\"off\")\n\n    plt.tight_layout()\n    plt.show()\n\nprint(\"\\nShowing random sample images from the dataset...\\n\")\nshow_samples(train_raw, numeric_classes, real_names, n=9)",
        "outputs": [
          "\n================ CLASS MAPPING TABLE ================\nNumber of classes: 102\n\nFolder     | Flower Name\n--------------------------------------------------\n1          | pink primrose\n10         | globe thistle\n100        | blanket flower\n101        | trumpet creeper\n102        | blackberry lily\n11         | snapdragon\n12         | colt's foot\n13         | king protea\n14         | spear thistle\n15         | yellow iris\n16         | globe-flower\n17         | purple coneflower\n18         | peruvian lily\n19         | balloon flower\n2          | hard-leaved pocket orchid\n20         | giant white arum lily\n21         | fire lily\n22         | pincushion flower\n23         | fritillary\n24         | red ginger\n25         | grape hyacinth\n26         | corn poppy\n27         | prince of wales feathers\n28         | stemless gentian\n29         | artichoke\n3          | canterbury bells\n30         | sweet william\n31         | carnation\n32         | garden phlox\n33         | love in the mist\n34         | mexican aster\n35         | alpine sea holly\n36         | ruby-lipped cattleya\n37         | cape flower\n38         | great masterwort\n39         | siam tulip\n4          | sweet pea\n40         | lenten rose\n41         | barbeton daisy\n42         | daffodil\n43         | sword lily\n44         | poinsettia\n45         | bolero deep blue\n46         | wallflower\n47         | marigold\n48         | buttercup\n49         | oxeye daisy\n5          | english marigold\n50         | common dandelion\n51         | petunia\n52         | wild pansy\n53         | primula\n54         | sunflower\n55         | pelargonium\n56         | bishop of llandaff\n57         | gaura\n58         | geranium\n59         | orange dahlia\n6          | tiger lily\n60         | pink-yellow dahlia\n61         | cautleya spicata\n62         | japanese anemone\n63         | black-eyed susan\n64         | silverbush\n65         | californian poppy\n66         | osteospermum\n67         | spring crocus\n68         | bearded iris\n69         | windflower\n7          | moon orchid\n70         | tree poppy\n71         | gazania\n72         | azalea\n73         | water lily\n74         | rose\n75         | thorn apple\n76         | morning glory\n77         | passion flower\n78         | lotus lotus\n79         | toad lily\n8          | bird of paradise\n80         | anthurium\n81         | frangipani\n82         | clematis\n83         | hibiscus\n84         | columbine\n85         | desert-rose\n86         | tree mallow\n87         | magnolia\n88         | cyclamen\n89         | watercress\n9          | monkshood\n90         | canna lily\n91         | hippeastrum\n92         | bee balm\n93         | ball moss\n94         | foxglove\n95         | bougainvillea\n96         | camellia\n97         | mallow\n98         | mexican petunia\n99         | bromelia\n======================================================\n\n",
          "\nShowing random sample images from the dataset...\n\n"
        ]
      },
      {
        "code": "# Model Training\n\n# Loading dataset\ntrain_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"train\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=True)\n\nvalid_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"valid\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=True)\n\ntest_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"test\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=False)\n\n# Preprocess function\ndef preprocess(image, label):\n    image = preprocess_input(tf.cast(image, tf.float32))\n    label = tf.one_hot(label, num_classes)\n    return image, label\n\ntrain_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\nvalid_ds = valid_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\ntest_ds = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\n\n# Function to set trainable layers\ndef set_trainable_layers(base_model, unfreeze_percent):\n    total_layers = len(base_model.layers)\n    num_unfreeze = int(total_layers * unfreeze_percent)\n    \n    for i, layer in enumerate(base_model.layers):\n        if i < total_layers - num_unfreeze:\n            layer.trainable = False\n        else:\n            layer.trainable = True\n    \n    print(f\"Total layers in base model: {total_layers}\")\n    print(f\"Unfreeze percent: {unfreeze_percent*100:.1f}%\")\n    print(f\"No. of layers unfreeze: {num_unfreeze}\")\n    print(f\"Layers freeze: {total_layers - num_unfreeze}\")\n    if unfreeze_percent != 0:\n        print(\"\\nList of unfreeze layers:\")\n        for i, layer in enumerate(base_model.layers[-num_unfreeze:]):\n            print(f\"  {i + (total_layers - num_unfreeze)}: {layer.name}\")\n\n# Build model\nbase_model = VGG19(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))\n\n# Unfreeze last 0% of layers\nset_trainable_layers(base_model, 0.00)\n\nx = GlobalAveragePooling2D()(base_model.output)\nx = Dropout(0.3)(x)\nx = Dense(256, activation='relu')(x)\nx = Dropout(0.3)(x)\noutput = Dense(num_classes, activation='softmax')(x)\nmodel = Model(base_model.input, output)\nprint(f\"Total layers in final model: {len(model.layers)}\")\n\ndef compact_model_summary(model, base_model_name=\"VGG19\", custom_layer_start_idx=-5):\n    col1_width = 52\n    col2_width = 25\n    col3_width = 12\n\n    print(f\"{'Layer (type)':<{col1_width}} {'Output Shape':<{col2_width}} {'Param #':>{col3_width}}\")\n    print(\"=\" * (col1_width + col2_width + col3_width))\n\n    base_output = model.layers[custom_layer_start_idx - 1].output.shape\n    total_base_params = sum([l.count_params() for l in model.layers[:custom_layer_start_idx]])\n    print(f\"{base_model_name.title()} (Functional)\".ljust(col1_width), f\"{str(base_output):<{col2_width}}\", f\"{total_base_params:>{col3_width},}\")\n\n    for layer in model.layers[custom_layer_start_idx:]:\n        name = layer.name\n        layer_type = layer.__class__.__name__\n        try:\n            output_shape = str(layer.output.shape)\n        except:\n            output_shape = \"N/A\"\n        params = f\"{layer.count_params():,}\"\n        print(f\"{name + ' (' + layer_type + ')':<{col1_width}} {output_shape:<{col2_width}} {params:>{col3_width}}\")\n\n    print(\"=\" * (col1_width + col2_width + col3_width))\n\n    total_params = model.count_params()\n    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])\n    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])\n\n    print(f\"Total params: {total_params:,}\")\n    print(f\"Trainable params: {trainable_params:,}\")\n    print(f\"Non-trainable params: {non_trainable_params:,}\")\n\nprint(\"\\nPrinting Model Summary:\")\ncompact_model_summary(model, base_model_name=\"VGG19\", custom_layer_start_idx=-5)\n\n# Compile model\nloss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)\n\nckpt_path = os.path.join(output_dir, \"best_model.h5\")\ncallbacks = [\n    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0),\n    ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss', verbose=0)\n]\n\nmodel.compile(optimizer=Adam(learning_rate), loss=loss_fn, metrics=['accuracy'])\n\n# Train model\nprint(\"\\nStarting training...\\n\")\nhistory = model.fit(\n    train_ds,\n    validation_data=valid_ds,\n    epochs=epochs,\n    callbacks=callbacks,\n    verbose=0)\nprint(\"\\nTraining completed.\")",
        "outputs": [
          "Found 6552 files belonging to 102 classes.\nFound 818 files belonging to 102 classes.\nFound 819 files belonging to 102 classes.\nDownloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5\n\u001b[1m80134624/80134624\u001b[0m \u001b[32m\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 0us/step\nTotal layers in base model: 22\nUnfreeze percent: 0.0%\nNo. of layers unfreeze: 0\nLayers freeze: 22\nTotal layers in final model: 27\n\nPrinting Model Summary:\nLayer (type)                                         Output Shape                   Param #\n=========================================================================================\nVgg19 (Functional)                                   (None, 7, 7, 512)           20,024,384\nglobal_average_pooling2d (GlobalAveragePooling2D)    (None, 512)                          0\ndropout (Dropout)                                    (None, 512)                          0\ndense (Dense)                                        (None, 256)                    131,328\ndropout_1 (Dropout)                                  (None, 256)                          0\ndense_1 (Dense)                                      (None, 102)                     26,214\n=========================================================================================\nTotal params: 20,181,926\nTrainable params: 157,542\nNon-trainable params: 20,024,384\n\nStarting training...\n\n",
          "\nTraining completed.\n"
        ]
      },
      {
        "code": "# Model Evaluation\nprint(\"\\nLoading best model for evaluation...\")\nbest_model = tf.keras.models.load_model(ckpt_path)\n\nprint(\"\\nEvaluating on Train:\")\ntrain_loss, train_acc = best_model.evaluate(train_ds, verbose=0)\nprint(f\"Train Accuracy: {train_acc*100:.2f}%\")\n\nprint(\"\\nEvaluating on Validation:\")\nval_loss, val_acc = best_model.evaluate(valid_ds, verbose=0)\nprint(f\"Validation Accuracy: {val_acc*100:.2f}%\")\n\nprint(\"\\nEvaluating on Test:\")\ntest_loss, test_acc = best_model.evaluate(test_ds, verbose=0)\nprint(f\"Test Accuracy: {test_acc*100:.2f}%\")",
        "outputs": [
          "\nLoading best model for evaluation...\n\nEvaluating on Train:\nTrain Accuracy: 99.88%\n\nEvaluating on Validation:\nValidation Accuracy: 88.51%\n\nEvaluating on Test:\nTest Accuracy: 88.28%\n"
        ]
      },
      {
        "code": "# Visualizing Results\n\n# Training curves\ndef plot_training_curves(history):\n    acc = history.history[\"accuracy\"]\n    val_acc = history.history[\"val_accuracy\"]\n    loss = history.history[\"loss\"]\n    val_loss = history.history[\"val_loss\"]\n    epochs_range = range(1, len(acc) + 1)\n\n    plt.figure(figsize=(14, 5))\n\n    # Accuracy\n    plt.subplot(1, 2, 1)\n    plt.plot(epochs_range, acc, label=\"Train Acc\")\n    plt.plot(epochs_range, val_acc, label=\"Val Acc\")\n    plt.xlabel(\"Epoch\")\n    plt.ylabel(\"Accuracy\")\n    plt.title(\"Training vs Validation Accuracy\")\n    plt.legend()\n\n    # Loss\n    plt.subplot(1, 2, 2)\n    plt.plot(epochs_range, loss, label=\"Train Loss\")\n    plt.plot(epochs_range, val_loss, label=\"Val Loss\")\n    plt.xlabel(\"Epoch\")\n    plt.ylabel(\"Loss\")\n    plt.title(\"Training vs Validation Loss\")\n    plt.legend()\n\n    plt.tight_layout()\n    plt.show()\n\nprint(\"\\nPlotting training curves...\")\nplot_training_curves(history)\n\n# Classification report\ny_true = []\ny_pred = []\n\nfor x_batch, y_batch in test_ds:\n    preds = best_model.predict(x_batch, verbose=0)\n    y_pred.extend(np.argmax(preds, axis=1))\n    y_true.extend(np.argmax(y_batch.numpy(), axis=1))\n\ny_true = np.array(y_true)\ny_pred = np.array(y_pred)\n\nprint(\"\\nClassification Report:\")\nprint(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))",
        "outputs": [
          "\nPlotting training curves...\n",
          "\nClassification Report:\n                           precision    recall  f1-score   support\n\n            pink primrose       1.00      0.60      0.75         5\n            globe thistle       1.00      1.00      1.00         3\n           blanket flower       1.00      1.00      1.00         8\n          trumpet creeper       0.80      1.00      0.89         4\n          blackberry lily       0.75      1.00      0.86         6\n               snapdragon       1.00      0.89      0.94         9\n              colt's foot       0.86      0.67      0.75         9\n              king protea       1.00      0.67      0.80         6\n            spear thistle       1.00      1.00      1.00         3\n              yellow iris       1.00      1.00      1.00         4\n             globe-flower       1.00      0.67      0.80         3\n        purple coneflower       1.00      1.00      1.00         9\n            peruvian lily       0.75      1.00      0.86         6\n           balloon flower       0.88      1.00      0.93         7\nhard-leaved pocket orchid       1.00      1.00      1.00         5\n    giant white arum lily       0.50      0.67      0.57         3\n                fire lily       1.00      1.00      1.00         2\n        pincushion flower       1.00      0.75      0.86         4\n               fritillary       1.00      1.00      1.00         7\n               red ginger       1.00      1.00      1.00         2\n           grape hyacinth       1.00      1.00      1.00         5\n               corn poppy       1.00      0.80      0.89         5\n prince of wales feathers       1.00      1.00      1.00         3\n         stemless gentian       1.00      1.00      1.00         6\n                artichoke       1.00      1.00      1.00         9\n         canterbury bells       0.67      1.00      0.80         2\n            sweet william       1.00      0.86      0.92        14\n                carnation       1.00      1.00      1.00         2\n             garden phlox       1.00      0.67      0.80         6\n         love in the mist       1.00      0.88      0.93         8\n            mexican aster       1.00      0.60      0.75         5\n         alpine sea holly       1.00      1.00      1.00         6\n     ruby-lipped cattleya       1.00      1.00      1.00         7\n              cape flower       0.89      1.00      0.94         8\n         great masterwort       0.88      0.88      0.88         8\n               siam tulip       0.80      0.80      0.80         5\n                sweet pea       0.75      0.50      0.60         6\n              lenten rose       0.62      0.62      0.62         8\n           barbeton daisy       0.93      0.93      0.93        14\n                 daffodil       0.67      1.00      0.80         4\n               sword lily       0.75      0.94      0.83        16\n               poinsettia       0.90      0.82      0.86        11\n         bolero deep blue       1.00      1.00      1.00         3\n               wallflower       0.91      1.00      0.95        21\n                 marigold       0.75      1.00      0.86         3\n                buttercup       1.00      1.00      1.00         5\n              oxeye daisy       1.00      1.00      1.00         3\n         english marigold       0.67      0.50      0.57         4\n         common dandelion       0.88      0.88      0.88         8\n                  petunia       0.71      0.83      0.77        24\n               wild pansy       0.70      0.88      0.78         8\n                  primula       0.89      0.57      0.70        14\n                sunflower       1.00      1.00      1.00         4\n              pelargonium       1.00      1.00      1.00         7\n       bishop of llandaff       1.00      1.00      1.00         8\n                    gaura       0.91      0.91      0.91        11\n                 geranium       1.00      1.00      1.00        14\n            orange dahlia       1.00      1.00      1.00         7\n               tiger lily       1.00      0.67      0.80         9\n       pink-yellow dahlia       1.00      1.00      1.00        10\n         cautleya spicata       1.00      1.00      1.00         8\n         japanese anemone       0.75      0.75      0.75         4\n         black-eyed susan       1.00      1.00      1.00         4\n               silverbush       1.00      1.00      1.00         5\n        californian poppy       0.58      1.00      0.74         7\n             osteospermum       0.67      1.00      0.80         4\n            spring crocus       1.00      0.50      0.67         4\n             bearded iris       0.75      1.00      0.86         3\n               windflower       1.00      1.00      1.00         3\n              moon orchid       1.00      0.83      0.91         6\n               tree poppy       1.00      1.00      1.00         4\n                  gazania       1.00      1.00      1.00         9\n                   azalea       0.78      0.64      0.70        11\n               water lily       0.90      1.00      0.95        28\n                     rose       0.87      0.93      0.90        14\n              thorn apple       0.93      1.00      0.96        13\n            morning glory       1.00      1.00      1.00         4\n           passion flower       0.89      1.00      0.94        25\n              lotus lotus       0.87      0.93      0.90        14\n                toad lily       1.00      1.00      1.00         3\n         bird of paradise       0.91      1.00      0.95        10\n                anthurium       1.00      0.82      0.90        11\n               frangipani       0.93      1.00      0.96        13\n                 clematis       0.92      0.71      0.80        17\n                 hibiscus       0.80      0.86      0.83        14\n                columbine       0.78      0.70      0.74        10\n              desert-rose       0.75      0.60      0.67        10\n              tree mallow       1.00      0.80      0.89         5\n                 magnolia       0.75      1.00      0.86         6\n                 cyclamen       0.79      0.85      0.81        13\n               watercress       0.88      1.00      0.94        15\n                monkshood       0.67      1.00      0.80         2\n               canna lily       0.90      0.64      0.75        14\n              hippeastrum       0.88      0.88      0.88         8\n                 bee balm       0.85      1.00      0.92        11\n                ball moss       0.86      1.00      0.92         6\n                 foxglove       1.00      0.94      0.97        16\n            bougainvillea       0.91      0.71      0.80        14\n                 camellia       0.71      0.56      0.63         9\n                   mallow       0.50      0.60      0.55         5\n          mexican petunia       1.00      0.75      0.86         4\n                 bromelia       0.88      1.00      0.93         7\n\n                 accuracy                           0.88       819\n                macro avg       0.90      0.89      0.88       819\n             weighted avg       0.89      0.88      0.88       819\n\n"
        ]
      },
      {
        "code": "# not to be written in code cell (in ppt), for our reference only\n\nfrom sklearn.metrics import confusion_matrix\ncm = confusion_matrix(y_true, y_pred)\n\nplt.figure(figsize=(30, 30))\nsns.heatmap(cm, cmap=\"Blues\", annot=True, fmt='d',\n            xticklabels=class_names, yticklabels=class_names,)\nplt.xlabel(\"Predicted\")\nplt.ylabel(\"True\")\nplt.title(\"Confusion Matrix\")\nplt.tight_layout()\nplt.show()",
        "outputs": []
      }
    ]
  },
  "vgg19_20": {
    "cells": [
      {
        "code": "# Importing Libraries\nimport os\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport tensorflow as tf\nimport seaborn as sns\nimport json\nfrom sklearn.metrics import classification_report\nfrom tensorflow.keras.applications.vgg19 import preprocess_input\nfrom tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout\nfrom tensorflow.keras.models import Model\nfrom tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint\nfrom tensorflow.keras.optimizers import Adam\nfrom tensorflow.keras.applications import VGG19\nprint(\"All the required libraries are imported.\\n\")\n",
        "outputs": [
          "All the required libraries are imported.\n\n"
        ]
      },
      {
        "code": "# Defining Paths & Parameters\nbase_path = '/kaggle/input/oxford-102-flower-dataset/102 flower/flowers'\njson_path = '/kaggle/input/oxford-102-flower-dataset/102 flower/cat_to_name.json'\nimg_size = 224\nbatch_size = 32\nepochs = 150\npatience = 15\nlearning_rate = 1e-5\noutput_dir = './vgg19_outputs'\nos.makedirs(output_dir, exist_ok=True)\n\nprint(f\"Parameters for training are set as:\")\nprint(f\"  Image Size: {img_size}\")\nprint(f\"  Batch Size: {batch_size}\")\nprint(f\"  Epochs: {epochs}\")\nprint(f\"  Patience: {patience}\")\nprint(f\"  Learning Rate: {learning_rate}\\n\")",
        "outputs": [
          "Parameters for training are set as:\n  Image Size: 224\n  Batch Size: 32\n  Epochs: 150\n  Patience: 15\n  Learning Rate: 1e-05\n\n"
        ]
      },
      {
        "code": "# Visualizing Dataset\nwith open(json_path, 'r') as f:\n    real_names = json.load(f)\n\nnumeric_classes = sorted(os.listdir(os.path.join(base_path, 'train')))\nnum_classes = len(numeric_classes)\nclass_names = [real_names[c] for c in numeric_classes]\n\nprint(\"\\n================ CLASS MAPPING TABLE ================\")\nprint(f\"Number of classes: {num_classes}\\n\")\nprint(f\"{'Folder':<10} | {'Flower Name'}\")\nprint(\"-\" * 50)\nfor f, cname in zip(numeric_classes, class_names):\n    print(f\"{f:<10} | {cname}\")\nprint(\"======================================================\\n\")\n\ntrain_raw = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"train\"),\n    image_size=(img_size, img_size),\n    batch_size=9,          # small batch for visualization\n    label_mode='int',\n    shuffle=True,\n    verbose=0)\n\n# Showing samples from dataset\ndef show_samples(ds_raw, numeric_class_list, name_map, n=9):\n    plt.figure(figsize=(8, 8))\n\n    for batch_x, batch_y in ds_raw.take(1):\n        imgs = batch_x.numpy().astype(\"uint8\")\n        labels = batch_y.numpy()\n        break\n\n    rows = int(np.ceil(np.sqrt(n)))\n    for i in range(n):\n        ax = plt.subplot(rows, rows, i+1)\n        plt.imshow(imgs[i])\n        folder_id = numeric_class_list[int(labels[i])]\n        cls_name = name_map[folder_id]\n        ax.set_title(cls_name, fontsize=9)\n        ax.axis(\"off\")\n\n    plt.tight_layout()\n    plt.show()\n\nprint(\"\\nShowing random sample images from the dataset...\\n\")\nshow_samples(train_raw, numeric_classes, real_names, n=9)",
        "outputs": [
          "\n================ CLASS MAPPING TABLE ================\nNumber of classes: 102\n\nFolder     | Flower Name\n--------------------------------------------------\n1          | pink primrose\n10         | globe thistle\n100        | blanket flower\n101        | trumpet creeper\n102        | blackberry lily\n11         | snapdragon\n12         | colt's foot\n13         | king protea\n14         | spear thistle\n15         | yellow iris\n16         | globe-flower\n17         | purple coneflower\n18         | peruvian lily\n19         | balloon flower\n2          | hard-leaved pocket orchid\n20         | giant white arum lily\n21         | fire lily\n22         | pincushion flower\n23         | fritillary\n24         | red ginger\n25         | grape hyacinth\n26         | corn poppy\n27         | prince of wales feathers\n28         | stemless gentian\n29         | artichoke\n3          | canterbury bells\n30         | sweet william\n31         | carnation\n32         | garden phlox\n33         | love in the mist\n34         | mexican aster\n35         | alpine sea holly\n36         | ruby-lipped cattleya\n37         | cape flower\n38         | great masterwort\n39         | siam tulip\n4          | sweet pea\n40         | lenten rose\n41         | barbeton daisy\n42         | daffodil\n43         | sword lily\n44         | poinsettia\n45         | bolero deep blue\n46         | wallflower\n47         | marigold\n48         | buttercup\n49         | oxeye daisy\n5          | english marigold\n50         | common dandelion\n51         | petunia\n52         | wild pansy\n53         | primula\n54         | sunflower\n55         | pelargonium\n56         | bishop of llandaff\n57         | gaura\n58         | geranium\n59         | orange dahlia\n6          | tiger lily\n60         | pink-yellow dahlia\n61         | cautleya spicata\n62         | japanese anemone\n63         | black-eyed susan\n64         | silverbush\n65         | californian poppy\n66         | osteospermum\n67         | spring crocus\n68         | bearded iris\n69         | windflower\n7          | moon orchid\n70         | tree poppy\n71         | gazania\n72         | azalea\n73         | water lily\n74         | rose\n75         | thorn apple\n76         | morning glory\n77         | passion flower\n78         | lotus lotus\n79         | toad lily\n8          | bird of paradise\n80         | anthurium\n81         | frangipani\n82         | clematis\n83         | hibiscus\n84         | columbine\n85         | desert-rose\n86         | tree mallow\n87         | magnolia\n88         | cyclamen\n89         | watercress\n9          | monkshood\n90         | canna lily\n91         | hippeastrum\n92         | bee balm\n93         | ball moss\n94         | foxglove\n95         | bougainvillea\n96         | camellia\n97         | mallow\n98         | mexican petunia\n99         | bromelia\n======================================================\n\n",
          "\nShowing random sample images from the dataset...\n\n"
        ]
      },
      {
        "code": "# Model Training\n\n# Loading dataset\ntrain_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"train\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=True)\n\nvalid_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"valid\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=True)\n\ntest_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"test\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=False)\n\n# Preprocess function\ndef preprocess(image, label):\n    image = preprocess_input(tf.cast(image, tf.float32))\n    label = tf.one_hot(label, num_classes)\n    return image, label\n\ntrain_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\nvalid_ds = valid_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\ntest_ds = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\n\n# Function to set trainable layers\ndef set_trainable_layers(base_model, unfreeze_percent):\n    total_layers = len(base_model.layers)\n    num_unfreeze = int(total_layers * unfreeze_percent)\n    \n    for i, layer in enumerate(base_model.layers):\n        if i < total_layers - num_unfreeze:\n            layer.trainable = False\n        else:\n            layer.trainable = True\n    \n    print(f\"Total layers in base model: {total_layers}\")\n    print(f\"Unfreeze percent: {unfreeze_percent*100:.1f}%\")\n    print(f\"No. of layers unfreeze: {num_unfreeze}\")\n    print(f\"Layers freeze: {total_layers - num_unfreeze}\")\n    if unfreeze_percent != 0:\n        print(\"\\nList of unfreeze layers:\")\n        for i, layer in enumerate(base_model.layers[-num_unfreeze:]):\n            print(f\"  {i + (total_layers - num_unfreeze)}: {layer.name}\")\n\n# Build model\nbase_model = VGG19(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))\n\n# Unfreeze last 20% of layers\nset_trainable_layers(base_model, 0.20)\n\nx = GlobalAveragePooling2D()(base_model.output)\nx = Dropout(0.3)(x)\nx = Dense(256, activation='relu')(x)\nx = Dropout(0.3)(x)\noutput = Dense(num_classes, activation='softmax')(x)\nmodel = Model(base_model.input, output)\nprint(f\"Total layers in final model: {len(model.layers)}\")\n\ndef compact_model_summary(model, base_model_name=\"VGG19\", custom_layer_start_idx=-5):\n    col1_width = 52\n    col2_width = 25\n    col3_width = 12\n\n    print(f\"{'Layer (type)':<{col1_width}} {'Output Shape':<{col2_width}} {'Param #':>{col3_width}}\")\n    print(\"=\" * (col1_width + col2_width + col3_width))\n\n    base_output = model.layers[custom_layer_start_idx - 1].output.shape\n    total_base_params = sum([l.count_params() for l in model.layers[:custom_layer_start_idx]])\n    print(f\"{base_model_name.title()} (Functional)\".ljust(col1_width), f\"{str(base_output):<{col2_width}}\", f\"{total_base_params:>{col3_width},}\")\n\n    for layer in model.layers[custom_layer_start_idx:]:\n        name = layer.name\n        layer_type = layer.__class__.__name__\n        try:\n            output_shape = str(layer.output.shape)\n        except:\n            output_shape = \"N/A\"\n        params = f\"{layer.count_params():,}\"\n        print(f\"{name + ' (' + layer_type + ')':<{col1_width}} {output_shape:<{col2_width}} {params:>{col3_width}}\")\n\n    print(\"=\" * (col1_width + col2_width + col3_width))\n\n    total_params = model.count_params()\n    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])\n    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])\n\n    print(f\"Total params: {total_params:,}\")\n    print(f\"Trainable params: {trainable_params:,}\")\n    print(f\"Non-trainable params: {non_trainable_params:,}\")\n\nprint(\"\\nPrinting Model Summary:\")\ncompact_model_summary(model, base_model_name=\"VGG19\", custom_layer_start_idx=-5)\n\n# Compile model\nloss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)\n\nckpt_path = os.path.join(output_dir, \"best_model.h5\")\ncallbacks = [\n    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0),\n    ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss', verbose=0)\n]\n\nmodel.compile(optimizer=Adam(learning_rate), loss=loss_fn, metrics=['accuracy'])\n\n# Train model\nprint(\"\\nStarting training...\\n\")\nhistory = model.fit(\n    train_ds,\n    validation_data=valid_ds,\n    epochs=epochs,\n    callbacks=callbacks,\n    verbose=0)\nprint(\"\\nTraining completed.\")",
        "outputs": [
          "Found 6552 files belonging to 102 classes.\nFound 818 files belonging to 102 classes.\nFound 819 files belonging to 102 classes.\nDownloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5\n\u001b[1m80134624/80134624\u001b[0m \u001b[32m\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 0us/step\nTotal layers in base model: 22\nUnfreeze percent: 20.0%\nNo. of layers unfreeze: 4\nLayers freeze: 18\n\nList of unfreeze layers:\n  18: block5_conv2\n  19: block5_conv3\n  20: block5_conv4\n  21: block5_pool\nTotal layers in final model: 27\n\nPrinting Model Summary:\nLayer (type)                                         Output Shape                   Param #\n=========================================================================================\nVgg19 (Functional)                                   (None, 7, 7, 512)           20,024,384\nglobal_average_pooling2d (GlobalAveragePooling2D)    (None, 512)                          0\ndropout (Dropout)                                    (None, 512)                          0\ndense (Dense)                                        (None, 256)                    131,328\ndropout_1 (Dropout)                                  (None, 256)                          0\ndense_1 (Dense)                                      (None, 102)                     26,214\n=========================================================================================\nTotal params: 20,181,926\nTrainable params: 7,236,966\nNon-trainable params: 12,944,960\n\nStarting training...\n\n",
          "\nTraining completed.\n"
        ]
      },
      {
        "code": "# Model Evaluation\nprint(\"\\nLoading best model for evaluation...\")\nbest_model = tf.keras.models.load_model(ckpt_path)\n\nprint(\"\\nEvaluating on Train:\")\ntrain_loss, train_acc = best_model.evaluate(train_ds, verbose=0)\nprint(f\"Train Accuracy: {train_acc*100:.2f}%\")\n\nprint(\"\\nEvaluating on Validation:\")\nval_loss, val_acc = best_model.evaluate(valid_ds, verbose=0)\nprint(f\"Validation Accuracy: {val_acc*100:.2f}%\")\n\nprint(\"\\nEvaluating on Test:\")\ntest_loss, test_acc = best_model.evaluate(test_ds, verbose=0)\nprint(f\"Test Accuracy: {test_acc*100:.2f}%\")",
        "outputs": [
          "\nLoading best model for evaluation...\n\nEvaluating on Train:\nTrain Accuracy: 100.00%\n\nEvaluating on Validation:\nValidation Accuracy: 93.52%\n\nEvaluating on Test:\nTest Accuracy: 92.92%\n"
        ]
      },
      {
        "code": "# Visualizing Results\n\n# Training curves\ndef plot_training_curves(history):\n    acc = history.history[\"accuracy\"]\n    val_acc = history.history[\"val_accuracy\"]\n    loss = history.history[\"loss\"]\n    val_loss = history.history[\"val_loss\"]\n    epochs_range = range(1, len(acc) + 1)\n\n    plt.figure(figsize=(14, 5))\n\n    # Accuracy\n    plt.subplot(1, 2, 1)\n    plt.plot(epochs_range, acc, label=\"Train Acc\")\n    plt.plot(epochs_range, val_acc, label=\"Val Acc\")\n    plt.xlabel(\"Epoch\")\n    plt.ylabel(\"Accuracy\")\n    plt.title(\"Training vs Validation Accuracy\")\n    plt.legend()\n\n    # Loss\n    plt.subplot(1, 2, 2)\n    plt.plot(epochs_range, loss, label=\"Train Loss\")\n    plt.plot(epochs_range, val_loss, label=\"Val Loss\")\n    plt.xlabel(\"Epoch\")\n    plt.ylabel(\"Loss\")\n    plt.title(\"Training vs Validation Loss\")\n    plt.legend()\n\n    plt.tight_layout()\n    plt.show()\n\nprint(\"\\nPlotting training curves...\")\nplot_training_curves(history)\n\n# Classification report\ny_true = []\ny_pred = []\n\nfor x_batch, y_batch in test_ds:\n    preds = best_model.predict(x_batch, verbose=0)\n    y_pred.extend(np.argmax(preds, axis=1))\n    y_true.extend(np.argmax(y_batch.numpy(), axis=1))\n\ny_true = np.array(y_true)\ny_pred = np.array(y_pred)\n\nprint(\"\\nClassification Report:\")\nprint(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))",
        "outputs": [
          "\nPlotting training curves...\n",
          "\nClassification Report:\n                           precision    recall  f1-score   support\n\n            pink primrose       1.00      0.80      0.89         5\n            globe thistle       1.00      1.00      1.00         3\n           blanket flower       1.00      0.88      0.93         8\n          trumpet creeper       1.00      0.75      0.86         4\n          blackberry lily       1.00      1.00      1.00         6\n               snapdragon       0.78      0.78      0.78         9\n              colt's foot       1.00      1.00      1.00         9\n              king protea       1.00      0.83      0.91         6\n            spear thistle       1.00      1.00      1.00         3\n              yellow iris       1.00      1.00      1.00         4\n             globe-flower       1.00      1.00      1.00         3\n        purple coneflower       0.82      1.00      0.90         9\n            peruvian lily       0.83      0.83      0.83         6\n           balloon flower       1.00      1.00      1.00         7\nhard-leaved pocket orchid       1.00      1.00      1.00         5\n    giant white arum lily       1.00      0.67      0.80         3\n                fire lily       1.00      1.00      1.00         2\n        pincushion flower       1.00      1.00      1.00         4\n               fritillary       1.00      1.00      1.00         7\n               red ginger       1.00      1.00      1.00         2\n           grape hyacinth       1.00      1.00      1.00         5\n               corn poppy       1.00      0.80      0.89         5\n prince of wales feathers       1.00      1.00      1.00         3\n         stemless gentian       1.00      0.83      0.91         6\n                artichoke       1.00      1.00      1.00         9\n         canterbury bells       0.50      1.00      0.67         2\n            sweet william       0.88      1.00      0.93        14\n                carnation       1.00      1.00      1.00         2\n             garden phlox       1.00      0.33      0.50         6\n         love in the mist       1.00      0.88      0.93         8\n            mexican aster       1.00      0.80      0.89         5\n         alpine sea holly       1.00      1.00      1.00         6\n     ruby-lipped cattleya       1.00      1.00      1.00         7\n              cape flower       0.89      1.00      0.94         8\n         great masterwort       0.89      1.00      0.94         8\n               siam tulip       1.00      0.60      0.75         5\n                sweet pea       1.00      0.33      0.50         6\n              lenten rose       0.56      0.62      0.59         8\n           barbeton daisy       1.00      0.93      0.96        14\n                 daffodil       1.00      0.75      0.86         4\n               sword lily       0.89      1.00      0.94        16\n               poinsettia       1.00      0.91      0.95        11\n         bolero deep blue       1.00      0.67      0.80         3\n               wallflower       0.91      1.00      0.95        21\n                 marigold       1.00      1.00      1.00         3\n                buttercup       1.00      1.00      1.00         5\n              oxeye daisy       0.75      1.00      0.86         3\n         english marigold       1.00      0.75      0.86         4\n         common dandelion       0.89      1.00      0.94         8\n                  petunia       0.92      0.96      0.94        24\n               wild pansy       0.88      0.88      0.88         8\n                  primula       1.00      0.79      0.88        14\n                sunflower       1.00      1.00      1.00         4\n              pelargonium       1.00      1.00      1.00         7\n       bishop of llandaff       1.00      1.00      1.00         8\n                    gaura       0.91      0.91      0.91        11\n                 geranium       1.00      1.00      1.00        14\n            orange dahlia       1.00      1.00      1.00         7\n               tiger lily       1.00      0.89      0.94         9\n       pink-yellow dahlia       1.00      1.00      1.00        10\n         cautleya spicata       0.89      1.00      0.94         8\n         japanese anemone       1.00      0.75      0.86         4\n         black-eyed susan       1.00      1.00      1.00         4\n               silverbush       1.00      1.00      1.00         5\n        californian poppy       1.00      1.00      1.00         7\n             osteospermum       0.80      1.00      0.89         4\n            spring crocus       1.00      1.00      1.00         4\n             bearded iris       1.00      0.67      0.80         3\n               windflower       0.75      1.00      0.86         3\n              moon orchid       1.00      0.67      0.80         6\n               tree poppy       1.00      1.00      1.00         4\n                  gazania       1.00      1.00      1.00         9\n                   azalea       0.92      1.00      0.96        11\n               water lily       1.00      1.00      1.00        28\n                     rose       0.93      1.00      0.97        14\n              thorn apple       0.81      1.00      0.90        13\n            morning glory       1.00      1.00      1.00         4\n           passion flower       0.96      1.00      0.98        25\n              lotus lotus       0.93      0.93      0.93        14\n                toad lily       1.00      1.00      1.00         3\n         bird of paradise       1.00      1.00      1.00        10\n                anthurium       1.00      1.00      1.00        11\n               frangipani       0.93      1.00      0.96        13\n                 clematis       0.77      1.00      0.87        17\n                 hibiscus       1.00      0.93      0.96        14\n                columbine       0.90      0.90      0.90        10\n              desert-rose       0.90      0.90      0.90        10\n              tree mallow       1.00      1.00      1.00         5\n                 magnolia       0.50      0.83      0.62         6\n                 cyclamen       0.87      1.00      0.93        13\n               watercress       0.94      1.00      0.97        15\n                monkshood       0.67      1.00      0.80         2\n               canna lily       0.92      0.79      0.85        14\n              hippeastrum       1.00      0.88      0.93         8\n                 bee balm       1.00      1.00      1.00        11\n                ball moss       1.00      1.00      1.00         6\n                 foxglove       1.00      0.94      0.97        16\n            bougainvillea       0.93      0.93      0.93        14\n                 camellia       0.75      0.67      0.71         9\n                   mallow       0.60      0.60      0.60         5\n          mexican petunia       1.00      1.00      1.00         4\n                 bromelia       1.00      0.86      0.92         7\n\n                 accuracy                           0.93       819\n                macro avg       0.94      0.92      0.92       819\n             weighted avg       0.94      0.93      0.93       819\n\n"
        ]
      },
      {
        "code": "# not to be written in code cell (in ppt), for our reference only\n\nfrom sklearn.metrics import confusion_matrix\ncm = confusion_matrix(y_true, y_pred)\n\nplt.figure(figsize=(30, 30))\nsns.heatmap(cm, cmap=\"Blues\", annot=True, fmt='d',\n            xticklabels=class_names, yticklabels=class_names,)\nplt.xlabel(\"Predicted\")\nplt.ylabel(\"True\")\nplt.title(\"Confusion Matrix\")\nplt.tight_layout()\nplt.show()",
        "outputs": []
      }
    ]
  },
  "vgg19_30": {
    "cells": [
      {
        "code": "# Importing Libraries\nimport os\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport tensorflow as tf\nimport seaborn as sns\nimport json\nfrom sklearn.metrics import classification_report\nfrom tensorflow.keras.applications.vgg19 import preprocess_input\nfrom tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout\nfrom tensorflow.keras.models import Model\nfrom tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint\nfrom tensorflow.keras.optimizers import Adam\nfrom tensorflow.keras.applications import VGG19\nprint(\"All the required libraries are imported.\\n\")",
        "outputs": [
          "All the required libraries are imported.\n\n"
        ]
      },
      {
        "code": "# Defining Paths & Parameters\nbase_path = '/kaggle/input/oxford-102-flower-dataset/102 flower/flowers'\njson_path = '/kaggle/input/oxford-102-flower-dataset/102 flower/cat_to_name.json'\nimg_size = 224\nbatch_size = 32\nepochs = 150\npatience = 15\nlearning_rate = 1e-5\noutput_dir = './vgg19_outputs'\nos.makedirs(output_dir, exist_ok=True)\n\nprint(f\"Parameters for training are set as:\")\nprint(f\"  Image Size: {img_size}\")\nprint(f\"  Batch Size: {batch_size}\")\nprint(f\"  Epochs: {epochs}\")\nprint(f\"  Patience: {patience}\")\nprint(f\"  Learning Rate: {learning_rate}\\n\")",
        "outputs": [
          "Parameters for training are set as:\n  Image Size: 224\n  Batch Size: 32\n  Epochs: 150\n  Patience: 15\n  Learning Rate: 1e-05\n\n"
        ]
      },
      {
        "code": "# Visualizing Dataset\nwith open(json_path, 'r') as f:\n    real_names = json.load(f)\n\nnumeric_classes = sorted(os.listdir(os.path.join(base_path, 'train')))\nnum_classes = len(numeric_classes)\nclass_names = [real_names[c] for c in numeric_classes]\n\nprint(\"\\n================ CLASS MAPPING TABLE ================\")\nprint(f\"Number of classes: {num_classes}\\n\")\nprint(f\"{'Folder':<10} | {'Flower Name'}\")\nprint(\"-\" * 50)\nfor f, cname in zip(numeric_classes, class_names):\n    print(f\"{f:<10} | {cname}\")\nprint(\"======================================================\\n\")\n\ntrain_raw = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"train\"),\n    image_size=(img_size, img_size),\n    batch_size=9,          # small batch for visualization\n    label_mode='int',\n    shuffle=True,\n    verbose=0)\n\n# Showing samples from dataset\ndef show_samples(ds_raw, numeric_class_list, name_map, n=9):\n    plt.figure(figsize=(8, 8))\n\n    for batch_x, batch_y in ds_raw.take(1):\n        imgs = batch_x.numpy().astype(\"uint8\")\n        labels = batch_y.numpy()\n        break\n\n    rows = int(np.ceil(np.sqrt(n)))\n    for i in range(n):\n        ax = plt.subplot(rows, rows, i+1)\n        plt.imshow(imgs[i])\n        folder_id = numeric_class_list[int(labels[i])]\n        cls_name = name_map[folder_id]\n        ax.set_title(cls_name, fontsize=9)\n        ax.axis(\"off\")\n\n    plt.tight_layout()\n    plt.show()\n\nprint(\"\\nShowing random sample images from the dataset...\\n\")\nshow_samples(train_raw, numeric_classes, real_names, n=9)",
        "outputs": [
          "\n================ CLASS MAPPING TABLE ================\nNumber of classes: 102\n\nFolder     | Flower Name\n--------------------------------------------------\n1          | pink primrose\n10         | globe thistle\n100        | blanket flower\n101        | trumpet creeper\n102        | blackberry lily\n11         | snapdragon\n12         | colt's foot\n13         | king protea\n14         | spear thistle\n15         | yellow iris\n16         | globe-flower\n17         | purple coneflower\n18         | peruvian lily\n19         | balloon flower\n2          | hard-leaved pocket orchid\n20         | giant white arum lily\n21         | fire lily\n22         | pincushion flower\n23         | fritillary\n24         | red ginger\n25         | grape hyacinth\n26         | corn poppy\n27         | prince of wales feathers\n28         | stemless gentian\n29         | artichoke\n3          | canterbury bells\n30         | sweet william\n31         | carnation\n32         | garden phlox\n33         | love in the mist\n34         | mexican aster\n35         | alpine sea holly\n36         | ruby-lipped cattleya\n37         | cape flower\n38         | great masterwort\n39         | siam tulip\n4          | sweet pea\n40         | lenten rose\n41         | barbeton daisy\n42         | daffodil\n43         | sword lily\n44         | poinsettia\n45         | bolero deep blue\n46         | wallflower\n47         | marigold\n48         | buttercup\n49         | oxeye daisy\n5          | english marigold\n50         | common dandelion\n51         | petunia\n52         | wild pansy\n53         | primula\n54         | sunflower\n55         | pelargonium\n56         | bishop of llandaff\n57         | gaura\n58         | geranium\n59         | orange dahlia\n6          | tiger lily\n60         | pink-yellow dahlia\n61         | cautleya spicata\n62         | japanese anemone\n63         | black-eyed susan\n64         | silverbush\n65         | californian poppy\n66         | osteospermum\n67         | spring crocus\n68         | bearded iris\n69         | windflower\n7          | moon orchid\n70         | tree poppy\n71         | gazania\n72         | azalea\n73         | water lily\n74         | rose\n75         | thorn apple\n76         | morning glory\n77         | passion flower\n78         | lotus lotus\n79         | toad lily\n8          | bird of paradise\n80         | anthurium\n81         | frangipani\n82         | clematis\n83         | hibiscus\n84         | columbine\n85         | desert-rose\n86         | tree mallow\n87         | magnolia\n88         | cyclamen\n89         | watercress\n9          | monkshood\n90         | canna lily\n91         | hippeastrum\n92         | bee balm\n93         | ball moss\n94         | foxglove\n95         | bougainvillea\n96         | camellia\n97         | mallow\n98         | mexican petunia\n99         | bromelia\n======================================================\n\n",
          "\nShowing random sample images from the dataset...\n\n"
        ]
      },
      {
        "code": "# Model Training\n\n# Loading dataset\ntrain_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"train\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=True)\n\nvalid_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"valid\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=True)\n\ntest_ds = tf.keras.utils.image_dataset_from_directory(\n    os.path.join(base_path, \"test\"),\n    image_size=(img_size, img_size),\n    batch_size=batch_size,\n    label_mode='int',\n    shuffle=False)\n\n# Preprocess function\ndef preprocess(image, label):\n    image = preprocess_input(tf.cast(image, tf.float32))\n    label = tf.one_hot(label, num_classes)\n    return image, label\n\ntrain_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\nvalid_ds = valid_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\ntest_ds = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)\n\n# Function to set trainable layers\ndef set_trainable_layers(base_model, unfreeze_percent):\n    total_layers = len(base_model.layers)\n    num_unfreeze = int(total_layers * unfreeze_percent)\n    \n    for i, layer in enumerate(base_model.layers):\n        if i < total_layers - num_unfreeze:\n            layer.trainable = False\n        else:\n            layer.trainable = True\n    \n    print(f\"Total layers in base model: {total_layers}\")\n    print(f\"Unfreeze percent: {unfreeze_percent*100:.1f}%\")\n    print(f\"No. of layers unfreeze: {num_unfreeze}\")\n    print(f\"Layers freeze: {total_layers - num_unfreeze}\")\n    if unfreeze_percent != 0:\n        print(\"\\nList of unfreeze layers:\")\n        for i, layer in enumerate(base_model.layers[-num_unfreeze:]):\n            print(f\"  {i + (total_layers - num_unfreeze)}: {layer.name}\")\n\n# Build model\nbase_model = VGG19(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))\n\n# Unfreeze last 30% of layers\nset_trainable_layers(base_model, 0.30)\n\nx = GlobalAveragePooling2D()(base_model.output)\nx = Dropout(0.3)(x)\nx = Dense(256, activation='relu')(x)\nx = Dropout(0.3)(x)\noutput = Dense(num_classes, activation='softmax')(x)\nmodel = Model(base_model.input, output)\nprint(f\"Total layers in final model: {len(model.layers)}\")\n\ndef compact_model_summary(model, base_model_name=\"VGG19\", custom_layer_start_idx=-5):\n    col1_width = 52\n    col2_width = 25\n    col3_width = 12\n\n    print(f\"{'Layer (type)':<{col1_width}} {'Output Shape':<{col2_width}} {'Param #':>{col3_width}}\")\n    print(\"=\" * (col1_width + col2_width + col3_width))\n\n    base_output = model.layers[custom_layer_start_idx - 1].output.shape\n    total_base_params = sum([l.count_params() for l in model.layers[:custom_layer_start_idx]])\n    print(f\"{base_model_name.title()} (Functional)\".ljust(col1_width), f\"{str(base_output):<{col2_width}}\", f\"{total_base_params:>{col3_width},}\")\n\n    for layer in model.layers[custom_layer_start_idx:]:\n        name = layer.name\n        layer_type = layer.__class__.__name__\n        try:\n            output_shape = str(layer.output.shape)\n        except:\n            output_shape = \"N/A\"\n        params = f\"{layer.count_params():,}\"\n        print(f\"{name + ' (' + layer_type + ')':<{col1_width}} {output_shape:<{col2_width}} {params:>{col3_width}}\")\n\n    print(\"=\" * (col1_width + col2_width + col3_width))\n\n    total_params = model.count_params()\n    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])\n    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])\n\n    print(f\"Total params: {total_params:,}\")\n    print(f\"Trainable params: {trainable_params:,}\")\n    print(f\"Non-trainable params: {non_trainable_params:,}\")\n\nprint(\"\\nPrinting Model Summary:\")\ncompact_model_summary(model, base_model_name=\"VGG19\", custom_layer_start_idx=-5)\n\n# Compile model\nloss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)\n\nckpt_path = os.path.join(output_dir, \"best_model.h5\")\ncallbacks = [\n    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0),\n    ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss', verbose=0)\n]\n\nmodel.compile(optimizer=Adam(learning_rate), loss=loss_fn, metrics=['accuracy'])\n\n# Train model\nprint(\"\\nStarting training...\\n\")\nhistory = model.fit(\n    train_ds,\n    validation_data=valid_ds,\n    epochs=epochs,\n    callbacks=callbacks,\n    verbose=0)\nprint(\"\\nTraining completed.\")",
        "outputs": [
          "Found 6552 files belonging to 102 classes.\nFound 818 files belonging to 102 classes.\nFound 819 files belonging to 102 classes.\nDownloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5\n\u001b[1m80134624/80134624\u001b[0m \u001b[32m\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 0us/step\nTotal layers in base model: 22\nUnfreeze percent: 30.0%\nNo. of layers unfreeze: 6\nLayers freeze: 16\n\nList of unfreeze layers:\n  16: block4_pool\n  17: block5_conv1\n  18: block5_conv2\n  19: block5_conv3\n  20: block5_conv4\n  21: block5_pool\nTotal layers in final model: 27\n\nPrinting Model Summary:\nLayer (type)                                         Output Shape                   Param #\n=========================================================================================\nVgg19 (Functional)                                   (None, 7, 7, 512)           20,024,384\nglobal_average_pooling2d (GlobalAveragePooling2D)    (None, 512)                          0\ndropout (Dropout)                                    (None, 512)                          0\ndense (Dense)                                        (None, 256)                    131,328\ndropout_1 (Dropout)                                  (None, 256)                          0\ndense_1 (Dense)                                      (None, 102)                     26,214\n=========================================================================================\nTotal params: 20,181,926\nTrainable params: 9,596,774\nNon-trainable params: 10,585,152\n\nStarting training...\n\n",
          "\nTraining completed.\n"
        ]
      },
      {
        "code": "# Model Evaluation\nprint(\"\\nLoading best model for evaluation...\")\nbest_model = tf.keras.models.load_model(ckpt_path)\n\nprint(\"\\nEvaluating on Train:\")\ntrain_loss, train_acc = best_model.evaluate(train_ds, verbose=0)\nprint(f\"Train Accuracy: {train_acc*100:.2f}%\")\n\nprint(\"\\nEvaluating on Validation:\")\nval_loss, val_acc = best_model.evaluate(valid_ds, verbose=0)\nprint(f\"Validation Accuracy: {val_acc*100:.2f}%\")\n\nprint(\"\\nEvaluating on Test:\")\ntest_loss, test_acc = best_model.evaluate(test_ds, verbose=0)\nprint(f\"Test Accuracy: {test_acc*100:.2f}%\")",
        "outputs": [
          "\nLoading best model for evaluation...\n\nEvaluating on Train:\nTrain Accuracy: 100.00%\n\nEvaluating on Validation:\nValidation Accuracy: 94.50%\n\nEvaluating on Test:\nTest Accuracy: 93.28%\n"
        ]
      },
      {
        "code": "# Visualizing Results\n\n# Training curves\ndef plot_training_curves(history):\n    acc = history.history[\"accuracy\"]\n    val_acc = history.history[\"val_accuracy\"]\n    loss = history.history[\"loss\"]\n    val_loss = history.history[\"val_loss\"]\n    epochs_range = range(1, len(acc) + 1)\n\n    plt.figure(figsize=(14, 5))\n\n    # Accuracy\n    plt.subplot(1, 2, 1)\n    plt.plot(epochs_range, acc, label=\"Train Acc\")\n    plt.plot(epochs_range, val_acc, label=\"Val Acc\")\n    plt.xlabel(\"Epoch\")\n    plt.ylabel(\"Accuracy\")\n    plt.title(\"Training vs Validation Accuracy\")\n    plt.legend()\n\n    # Loss\n    plt.subplot(1, 2, 2)\n    plt.plot(epochs_range, loss, label=\"Train Loss\")\n    plt.plot(epochs_range, val_loss, label=\"Val Loss\")\n    plt.xlabel(\"Epoch\")\n    plt.ylabel(\"Loss\")\n    plt.title(\"Training vs Validation Loss\")\n    plt.legend()\n\n    plt.tight_layout()\n    plt.show()\n\nprint(\"\\nPlotting training curves...\")\nplot_training_curves(history)\n\n# Classification report\ny_true = []\ny_pred = []\n\nfor x_batch, y_batch in test_ds:\n    preds = best_model.predict(x_batch, verbose=0)\n    y_pred.extend(np.argmax(preds, axis=1))\n    y_true.extend(np.argmax(y_batch.numpy(), axis=1))\n\ny_true = np.array(y_true)\ny_pred = np.array(y_pred)\n\nprint(\"\\nClassification Report:\")\nprint(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))",
        "outputs": [
          "\nPlotting training curves...\n",
          "\nClassification Report:\n                           precision    recall  f1-score   support\n\n            pink primrose       1.00      1.00      1.00         5\n            globe thistle       1.00      1.00      1.00         3\n           blanket flower       1.00      1.00      1.00         8\n          trumpet creeper       1.00      0.75      0.86         4\n          blackberry lily       1.00      1.00      1.00         6\n               snapdragon       0.88      0.78      0.82         9\n              colt's foot       1.00      0.89      0.94         9\n              king protea       1.00      0.83      0.91         6\n            spear thistle       0.75      1.00      0.86         3\n              yellow iris       0.80      1.00      0.89         4\n             globe-flower       1.00      1.00      1.00         3\n        purple coneflower       1.00      1.00      1.00         9\n            peruvian lily       0.83      0.83      0.83         6\n           balloon flower       1.00      1.00      1.00         7\nhard-leaved pocket orchid       1.00      1.00      1.00         5\n    giant white arum lily       0.40      0.67      0.50         3\n                fire lily       1.00      1.00      1.00         2\n        pincushion flower       1.00      1.00      1.00         4\n               fritillary       1.00      1.00      1.00         7\n               red ginger       0.67      1.00      0.80         2\n           grape hyacinth       1.00      1.00      1.00         5\n               corn poppy       0.75      0.60      0.67         5\n prince of wales feathers       1.00      1.00      1.00         3\n         stemless gentian       1.00      1.00      1.00         6\n                artichoke       1.00      1.00      1.00         9\n         canterbury bells       0.67      1.00      0.80         2\n            sweet william       1.00      0.93      0.96        14\n                carnation       1.00      1.00      1.00         2\n             garden phlox       1.00      0.83      0.91         6\n         love in the mist       1.00      0.88      0.93         8\n            mexican aster       1.00      1.00      1.00         5\n         alpine sea holly       1.00      0.83      0.91         6\n     ruby-lipped cattleya       1.00      1.00      1.00         7\n              cape flower       0.89      1.00      0.94         8\n         great masterwort       0.80      1.00      0.89         8\n               siam tulip       1.00      0.60      0.75         5\n                sweet pea       0.57      0.67      0.62         6\n              lenten rose       0.86      0.75      0.80         8\n           barbeton daisy       0.93      0.93      0.93        14\n                 daffodil       0.80      1.00      0.89         4\n               sword lily       0.93      0.88      0.90        16\n               poinsettia       0.91      0.91      0.91        11\n         bolero deep blue       1.00      0.67      0.80         3\n               wallflower       1.00      1.00      1.00        21\n                 marigold       1.00      1.00      1.00         3\n                buttercup       1.00      1.00      1.00         5\n              oxeye daisy       1.00      0.67      0.80         3\n         english marigold       1.00      0.75      0.86         4\n         common dandelion       1.00      1.00      1.00         8\n                  petunia       0.96      0.96      0.96        24\n               wild pansy       1.00      1.00      1.00         8\n                  primula       1.00      0.93      0.96        14\n                sunflower       1.00      1.00      1.00         4\n              pelargonium       1.00      1.00      1.00         7\n       bishop of llandaff       1.00      1.00      1.00         8\n                    gaura       1.00      0.91      0.95        11\n                 geranium       0.93      1.00      0.97        14\n            orange dahlia       1.00      1.00      1.00         7\n               tiger lily       1.00      0.89      0.94         9\n       pink-yellow dahlia       1.00      1.00      1.00        10\n         cautleya spicata       1.00      1.00      1.00         8\n         japanese anemone       1.00      0.75      0.86         4\n         black-eyed susan       1.00      1.00      1.00         4\n               silverbush       1.00      1.00      1.00         5\n        californian poppy       1.00      1.00      1.00         7\n             osteospermum       1.00      1.00      1.00         4\n            spring crocus       1.00      1.00      1.00         4\n             bearded iris       1.00      1.00      1.00         3\n               windflower       0.75      1.00      0.86         3\n              moon orchid       1.00      0.83      0.91         6\n               tree poppy       0.80      1.00      0.89         4\n                  gazania       1.00      1.00      1.00         9\n                   azalea       1.00      1.00      1.00        11\n               water lily       0.93      1.00      0.97        28\n                     rose       0.88      1.00      0.93        14\n              thorn apple       0.87      1.00      0.93        13\n            morning glory       1.00      0.75      0.86         4\n           passion flower       0.93      1.00      0.96        25\n              lotus lotus       0.93      1.00      0.97        14\n                toad lily       1.00      1.00      1.00         3\n         bird of paradise       0.91      1.00      0.95        10\n                anthurium       0.71      0.91      0.80        11\n               frangipani       0.93      1.00      0.96        13\n                 clematis       0.89      1.00      0.94        17\n                 hibiscus       0.93      0.93      0.93        14\n                columbine       0.89      0.80      0.84        10\n              desert-rose       0.90      0.90      0.90        10\n              tree mallow       1.00      1.00      1.00         5\n                 magnolia       0.86      1.00      0.92         6\n                 cyclamen       1.00      1.00      1.00        13\n               watercress       0.94      1.00      0.97        15\n                monkshood       0.67      1.00      0.80         2\n               canna lily       0.90      0.64      0.75        14\n              hippeastrum       0.88      0.88      0.88         8\n                 bee balm       0.83      0.91      0.87        11\n                ball moss       1.00      0.83      0.91         6\n                 foxglove       1.00      0.94      0.97        16\n            bougainvillea       1.00      0.79      0.88        14\n                 camellia       0.83      0.56      0.67         9\n                   mallow       0.67      0.80      0.73         5\n          mexican petunia       1.00      1.00      1.00         4\n                 bromelia       1.00      1.00      1.00         7\n\n                 accuracy                           0.93       819\n                macro avg       0.93      0.93      0.92       819\n             weighted avg       0.94      0.93      0.93       819\n\n"
        ]
      },
      {
        "code": "# not to be written in code cell (in ppt), for our reference only\n\nfrom sklearn.metrics import confusion_matrix\ncm = confusion_matrix(y_true, y_pred)\n\nplt.figure(figsize=(30, 30))\nsns.heatmap(cm, cmap=\"Blues\", annot=True, fmt='d',\n            xticklabels=class_names, yticklabels=class_names,)\nplt.xlabel(\"Predicted\")\nplt.ylabel(\"True\")\nplt.title(\"Confusion Matrix\")\nplt.tight_layout()\nplt.show()",
        "outputs": []
      }
    ]
  }
};


// Code templates for dynamic code display
const CODE_TEMPLATES = {
    mobilenetv2: {
        cell0: `# Importing Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import json
from sklearn.metrics import classification_report
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

print("All the required libraries are imported.\\n")`,
        cell1: `# Defining Paths & Parameters
base_path = '/kaggle/input/oxford-102-flower-dataset/102 flower/flowers'
json_path = '/kaggle/input/oxford-102-flower-dataset/102 flower/cat_to_name.json'
img_size = 224
batch_size = 32
epochs = 150
patience = 15
learning_rate = 1e-4
output_dir = './mobilenetv2_outputs'
os.makedirs(output_dir, exist_ok=True)

print(f"Parameters for training are set as:")
print(f"  Image Size: {img_size}")
print(f"  Batch Size: {batch_size}")
print(f"  Epochs: {epochs}")
print(f"  Patience: {patience}")
print(f"  Learning Rate: {learning_rate}\\n")`,
        cell2: `# Visualzing Dataset
with open(json_path, 'r') as f:
    real_names = json.load(f)

numeric_classes = sorted(os.listdir(os.path.join(base_path, 'train')))
num_classes = len(numeric_classes)
class_names = [real_names[c] for c in numeric_classes]

print("\\n================ CLASS MAPPING TABLE ================")
print(f"Number of classes: {num_classes}\\n")
print(f"{'Folder':<10} | {'Flower Name'}")
print("-" * 50)
for f, cname in zip(numeric_classes, class_names):
    print(f"{f:<10} | {cname}")
print("======================================================\\n")

train_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_path, "train"),
    image_size=(img_size, img_size),
    batch_size=9,          # small batch for visualization
    label_mode='int',
    shuffle=True,
    verbose=0)

# Showing samples from dataset
def show_samples(ds_raw, numeric_class_list, name_map, n=9):
    plt.figure(figsize=(8, 8))

    for batch_x, batch_y in ds_raw.take(1):
        imgs = batch_x.numpy().astype("uint8")
        labels = batch_y.numpy()
        break

    rows = int(np.ceil(np.sqrt(n)))
    for i in range(n):
        ax = plt.subplot(rows, rows, i+1)
        plt.imshow(imgs[i])
        folder_id = numeric_class_list[int(labels[i])]
        cls_name = name_map[folder_id]
        ax.set_title(cls_name, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

print("\\nShowing random sample images from the dataset...\\n")
show_samples(train_raw, numeric_classes, real_names, n=9)`,
        cell3: `# Model Training

# Loading dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_path, "train"),
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode='int',
    shuffle=True)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_path, "valid"),
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode='int',
    shuffle=True)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_path, "test"),
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode='int',
    shuffle=False)

# Preprocess function
def preprocess(image, label):
    image = preprocess_input(tf.cast(image, tf.float32))
    label = tf.one_hot(label, num_classes)
    return image, label

train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

# Function to set trainable layers
def set_trainable_layers(base_model, unfreeze_percent):
    total_layers = len(base_model.layers)
    num_unfreeze = int(total_layers * unfreeze_percent)
    
    for i, layer in enumerate(base_model.layers):
        if i < total_layers - num_unfreeze:
            layer.trainable = False
        else:
            layer.trainable = True
    
    print(f"Total layers in base model: {total_layers}")
    print(f"Unfreeze percent: {unfreeze_percent*100:.1f}%")
    print(f"No. of layers unfreeze: {num_unfreeze}")
    print(f"Layers freeze: {total_layers - num_unfreeze}")
    if unfreeze_percent != 0:
        print("\\nList of unfreeze layers:")
        for i, layer in enumerate(base_model.layers[-num_unfreeze:]):
            print(f"  {i + (total_layers - num_unfreeze)}: {layer.name}")

# Build model
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))

# Unfreeze last 0% of layers
set_trainable_layers(base_model, 0.00)

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(base_model.input, output)
print(f"Total layers in final model: {len(model.layers)}")

def compact_model_summary(model, base_model_name="MobileNetV2", custom_layer_start_idx=-5):
    col1_width = 52
    col2_width = 25
    col3_width = 12

    print(f"{'Layer (type)':<{col1_width}} {'Output Shape':<{col2_width}} {'Param #':>{col3_width}}")
    print("=" * (col1_width + col2_width + col3_width))

    base_output = model.layers[custom_layer_start_idx - 1].output.shape
    total_base_params = sum([l.count_params() for l in model.layers[:custom_layer_start_idx]])
    print(f"{base_model_name.title()} (Functional)".ljust(col1_width), f"{str(base_output):<{col2_width}}", f"{total_base_params:>{col3_width},}")

    for layer in model.layers[custom_layer_start_idx:]:
        name = layer.name
        layer_type = layer.__class__.__name__
        try:
            output_shape = str(layer.output.shape)
        except:
            output_shape = "N/A"
        params = f"{layer.count_params():,}"
        print(f"{name + ' (' + layer_type + ')':<{col1_width}} {output_shape:<{col2_width}} {params:>{col3_width}}")

    print("=" * (col1_width + col2_width + col3_width))

    total_params = model.count_params()
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")

print("\\nPrinting Model Summary:")
compact_model_summary(model, base_model_name="MobileNetV2", custom_layer_start_idx=-5)

# Compile model
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

ckpt_path = os.path.join(output_dir, "best_model.h5")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0),
    ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss', verbose=0)
]

model.compile(optimizer=Adam(learning_rate), loss=loss_fn, metrics=['accuracy'])

# Train model
print("\\nStarting training...\\n")
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=epochs,
    callbacks=callbacks,
    verbose=0)
print("\\nTraining completed.")`,
        cell4: `# Model Evaluation
print("\\nLoading best model for evaluation...")
best_model = tf.keras.models.load_model(ckpt_path)

print("\\nEvaluating on Train:")
train_loss, train_acc = best_model.evaluate(train_ds, verbose=0)
print(f"Train Accuracy: {train_acc*100:.2f}%")

print("\\nEvaluating on Validation:")
val_loss, val_acc = best_model.evaluate(valid_ds, verbose=0)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

print("\\nEvaluating on Test:")
test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")`,
        cell5: `# Visualizing Results

# Training curves
def plot_training_curves(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Acc")
    plt.plot(epochs_range, val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

print("\\nPlotting training curves...")
plot_training_curves(history)

# Classification report
y_true = []
y_pred = []

for x_batch, y_batch in test_ds:
    preds = best_model.predict(x_batch, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(y_batch.numpy(), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))`,
        cell6: `# not to be written in code cell (in ppt), for our reference only

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(30, 30))
sns.heatmap(cm, cmap="Blues", annot=True, fmt='d',
            xticklabels=class_names, yticklabels=class_names,)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()`,
    },
    vgg19: {
        cell0: `# Importing Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import json
from sklearn.metrics import classification_report
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19

print("All the required libraries are imported.\\n")`,
        cell1: `# Defining Paths & Parameters
base_path = '/kaggle/input/102 flower/flowers'
json_path = '/kaggle/input/102 flower/cat_to_name.json'
img_size = 224
batch_size = 32
epochs = 150
patience = 15
learning_rate = 1e-4
output_dir = './vgg19_outputs'
os.makedirs(output_dir, exist_ok=True)

print(f"Parameters for training are set as:")
print(f"  Image Size: {img_size}")
print(f"  Batch Size: {batch_size}")
print(f"  Epochs: {epochs}")
print(f"  Patience: {patience}")
print(f"  Learning Rate: {learning_rate}\\n")`,
        cell2: `# Visualzing Dataset
with open(json_path, 'r') as f:
    real_names = json.load(f)

numeric_classes = sorted(os.listdir(os.path.join(base_path, 'train')))
num_classes = len(numeric_classes)
class_names = [real_names[c] for c in numeric_classes]

print("\\n================ CLASS MAPPING TABLE ================")
print(f"Number of classes: {num_classes}\\n")
print(f"{'Folder':<10} | {'Flower Name'}")
print("-" * 50)
for f, cname in zip(numeric_classes, class_names):
    print(f"{f:<10} | {cname}")
print("======================================================\\n")

train_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_path, "train"),
    image_size=(img_size, img_size),
    batch_size=9,          # small batch for visualization
    label_mode='int',
    shuffle=True,
    verbose=0)

# Showing samples from dataset
def show_samples(ds_raw, numeric_class_list, name_map, n=9):
    plt.figure(figsize=(8, 8))

    for batch_x, batch_y in ds_raw.take(1):
        imgs = batch_x.numpy().astype("uint8")
        labels = batch_y.numpy()
        break

    rows = int(np.ceil(np.sqrt(n)))
    for i in range(n):
        ax = plt.subplot(rows, rows, i+1)
        plt.imshow(imgs[i])
        folder_id = numeric_class_list[int(labels[i])]
        cls_name = name_map[folder_id]
        ax.set_title(cls_name, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

print("\\nShowing random sample images from the dataset...\\n")
show_samples(train_raw, numeric_classes, real_names, n=9)`,
        cell3: `# Model Training

# Loading dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_path, "train"),
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode='int',
    shuffle=True)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_path, "valid"),
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode='int',
    shuffle=True)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_path, "test"),
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode='int',
    shuffle=False)

# Preprocess function
def preprocess(image, label):
    image = preprocess_input(tf.cast(image, tf.float32))
    label = tf.one_hot(label, num_classes)
    return image, label

train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

# Function to set trainable layers
def set_trainable_layers(base_model, unfreeze_percent):
    total_layers = len(base_model.layers)
    num_unfreeze = int(total_layers * unfreeze_percent)
    
    for i, layer in enumerate(base_model.layers):
        if i < total_layers - num_unfreeze:
            layer.trainable = False
        else:
            layer.trainable = True
    
    print(f"Total layers in base model: {total_layers}")
    print(f"Unfreeze percent: {unfreeze_percent*100:.1f}%")
    print(f"No. of layers unfreeze: {num_unfreeze}")
    print(f"Layers freeze: {total_layers - num_unfreeze}")
    if unfreeze_percent != 0:
        print("\\nList of unfreeze layers:")
        for i, layer in enumerate(base_model.layers[-num_unfreeze:]):
            print(f"  {i + (total_layers - num_unfreeze)}: {layer.name}")

# Build model
base_model = VGG19(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))

# Unfreeze last 0% of layers
set_trainable_layers(base_model, 0.00)

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(base_model.input, output)
print(f"Total layers in final model: {len(model.layers)}")

def compact_model_summary(model, base_model_name="VGG19", custom_layer_start_idx=-5):
    col1_width = 52
    col2_width = 25
    col3_width = 12

    print(f"{'Layer (type)':<{col1_width}} {'Output Shape':<{col2_width}} {'Param #':>{col3_width}}")
    print("=" * (col1_width + col2_width + col3_width))

    base_output = model.layers[custom_layer_start_idx - 1].output.shape
    total_base_params = sum([l.count_params() for l in model.layers[:custom_layer_start_idx]])
    print(f"{base_model_name.title()} (Functional)".ljust(col1_width), f"{str(base_output):<{col2_width}}", f"{total_base_params:>{col3_width},}")

    for layer in model.layers[custom_layer_start_idx:]:
        name = layer.name
        layer_type = layer.__class__.__name__
        try:
            output_shape = str(layer.output.shape)
        except:
            output_shape = "N/A"
        params = f"{layer.count_params():,}"
        print(f"{name + ' (' + layer_type + ')':<{col1_width}} {output_shape:<{col2_width}} {params:>{col3_width}}")

    print("=" * (col1_width + col2_width + col3_width))

    total_params = model.count_params()
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")

print("\\nPrinting Model Summary:")
compact_model_summary(model, base_model_name="VGG19", custom_layer_start_idx=-5)

# Compile model
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

ckpt_path = os.path.join(output_dir, "best_model.h5")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0),
    ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss', verbose=0)
]

model.compile(optimizer=Adam(learning_rate), loss=loss_fn, metrics=['accuracy'])

# Train model
print("\\nStarting training...\\n")
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=epochs,
    callbacks=callbacks,
    verbose=0)
print("\\nTraining completed.")`,
        cell4: `# Model Evaluation
print("\\nLoading best model for evaluation...")
best_model = tf.keras.models.load_model(ckpt_path)

print("\\nEvaluating on Train:")
train_loss, train_acc = best_model.evaluate(train_ds, verbose=0)
print(f"Train Accuracy: {train_acc*100:.2f}%")

print("\\nEvaluating on Validation:")
val_loss, val_acc = best_model.evaluate(valid_ds, verbose=0)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

print("\\nEvaluating on Test:")
test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")`,
        cell5: `# Visualizing Results

# Training curves
def plot_training_curves(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Acc")
    plt.plot(epochs_range, val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

print("\\nPlotting training curves...")
plot_training_curves(history)

# Classification report
y_true = []
y_pred = []

for x_batch, y_batch in test_ds:
    preds = best_model.predict(x_batch, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(y_batch.numpy(), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))`,
        cell6: `# not to be written in code cell (in ppt), for our reference only

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(30, 30))
sns.heatmap(cm, cmap="Blues", annot=True, fmt='d',
            xticklabels=class_names, yticklabels=class_names,)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()`,
    },
};
