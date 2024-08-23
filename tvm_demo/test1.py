# 1. Import Necessary Libraries
# The script starts by importing the necessary libraries for the task:
import time
import tvm
from tvm import relay
import numpy as np
from tvm.contrib.download import download_testdata
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from tvm import autotvm

# 2. Load a Pretrained PyTorch Model
# The script loads a pretrained ResNet-18 model from the torchvision library and sets it to evaluation mode:

model_name = "resnet18"
model = getattr(torchvision.models, model_name)(weights=ResNet18_Weights.IMAGENET1K_V1)
model = model.eval()

# 3. Trace the Model with TorchScript
# The model is traced using TorchScript to convert it into a format that can be used by TVM:
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

# 4. Download and Preprocess an Image
# An image is downloaded, resized, and preprocessed to match the input requirements of the model:

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)

# 5. Convert the PyTorch Model to a Relay Graph
# The TorchScript model is converted to a Relay graph, which is TVM's intermediate representation:
input_name = "input0"
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# 6. Compile the Relay Graph
# The Relay graph is compiled to an LLVM target:
target = tvm.target.Target("llvm", host="llvm")
ctx = tvm.cpu(0)
# Define the tuning options
tuning_option = {
    'log_filename': 'tuning.log',
    'tuner': 'xgb',
    'n_trial': 1000,
    'early_stopping': 600,
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=10, repeat=1, min_repeat_ms=1000),
    ),
}

# Extract tasks from the Relay program
tasks = autotvm.task.extract_from_program(mod["main"], target="llvm", params=params)

# Tune the extracted tasks
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i+1, len(tasks))
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=min(tuning_option['n_trial'], len(task.config_space)),
               early_stopping=tuning_option['early_stopping'],
               measure_option=tuning_option['measure_option'],
               callbacks=[
                   autotvm.callback.progress_bar(tuning_option['n_trial'], prefix=prefix),
                   autotvm.callback.log_to_file(tuning_option['log_filename'])
               ])

# Apply the best schedules
with autotvm.apply_history_best(tuning_option['log_filename']):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="llvm", params=params)


# 7. Execute the Compiled Model on TVM
# The compiled model is executed on the TVM runtime, and the output is obtained:

from tvm import runtime
from tvm.contrib import graph_executor

tvm_t0 = time.process_time()
for i in range(10):
    dtype = "float32"
    m = graph_executor.GraphModule(lib["default"](ctx))
    m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
    m.run()
    tvm_output = m.get_output(0)
tvm_t1 = time.process_time()

# 8. Look Up Synset Names
# The script downloads the synset names and class IDs to map the model's output to human-readable class names:

synset_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_synsets.txt",
    ]
)
synset_name = "imagenet_synsets.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synsets = f.readlines()

synsets = [x.strip() for x in synsets]
splits = [line.split(" ") for line in synsets]
key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

class_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_classes.txt",
    ]
)
class_name = "imagenet_classes.txt"
class_path = download_testdata(class_url, class_name, module="data")
with open(class_path) as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

# 9. Get Top-1 Prediction from TVM
# The top-1 prediction from the TVM output is obtained and mapped to a class name:

top1_tvm = np.argmax(tvm_output.asnumpy()[0])
tvm_class_key = class_id_to_key[top1_tvm]

# 10. Compare with PyTorch
# The same image is passed through the original PyTorch model to get the top-1 prediction for comparison:

torch_t0 = time.process_time()
for i in range(10):
    with torch.no_grad():
        torch_img = torch.from_numpy(img)
        output = model(torch_img)
        top1_torch = np.argmax(output.numpy())
        torch_class_key = class_id_to_key[top1_torch]
torch_t1 = time.process_time()

# 11. Measure and Print Execution Time
# The execution time for both TVM and PyTorch is measured and printed:

tvm_time = tvm_t1 - tvm_t0
torch_time = torch_t1 - torch_t0

print("Relay top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))
print("Torch top-1 id: {}, class name: {}".format(top1_torch, key_to_classname[torch_class_key]))
print('Relay time: ', tvm_time / 10.0, 'seconds')
print('Torch time: ', torch_time / 10.0, 'seconds')