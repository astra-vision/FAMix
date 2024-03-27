import os

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False
    layer.eval()

def freeze_1(model):
    freeze_layer(model.backbone.conv1)
    freeze_layer(model.backbone.bn1)
    freeze_layer(model.backbone.relu1)
    freeze_layer(model.backbone.conv2)
    freeze_layer(model.backbone.bn2)
    freeze_layer(model.backbone.relu2)
    freeze_layer(model.backbone.conv3)
    freeze_layer(model.backbone.bn3)
    freeze_layer(model.backbone.relu3)
    freeze_layer(model.backbone.avgpool)
    freeze_layer(model.backbone.layer1)

def freeze_1_2(model):
    freeze_1(model)
    freeze_layer(model.backbone.layer2)

def freeze_1_2_3(model):
    freeze_1_2(model)
    freeze_layer(model.backbone.layer3)

def freeze_1_2_3_p4(model):
    freeze_1_2_3(model)
    freeze_layer(model.backbone.layer4[0])
    freeze_layer(model.backbone.layer4[1])

def freeze_all(model):
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()