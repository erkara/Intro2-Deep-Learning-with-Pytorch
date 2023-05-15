
import gradio as gr
import timm
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn

#default is cpu on HuggingFace unless you pay for it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#properly enter this as we discuss in the class
idx_to_class = {0: 'adidas', 1: 'converse', 2: 'new-balance', 3: 'nike', 4: 'reebok', 5: 'vans'}
num_classes = len(idx_to_class)

#uploaded images will be transformed before the prediction
mean = [0.485, 0.456, 0.406]
std =  [0.229, 0.224, 0.225]
test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                     transforms.Normalize(mean,std)
                                     ])

#get the model you tranined, make sure to use exactly 
#the same version of whatever model trained
def GetModel(model_name = 'efficientnet_b0',freeze = False):
    model = timm.create_model(model_name = model_name,pretrained=True)
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
    
    in_features = model.classifier.in_features 
    
    model.classifier = nn.Sequential(
                          nn.Linear(in_features, 100), 
                          nn.BatchNorm1d(num_features=100),
                          nn.ReLU(),
                          nn.Dropout(),
                          nn.Linear(100, num_classes),
                                    )
    
    return model


#load the model trained
def LoadModel(model, model_path):
    checkpoint = torch.load(model_path,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.best_scores = checkpoint['best_stats']
    return model


model = LoadModel(GetModel(),"snicker_model.pth")

#this returns a dictory of classes with its confidance scores
def GetClassProbs(img):
    with torch.no_grad():
        model.eval()
        model.to(device)
        #img = Image.open(img).convert("RGB")
        img = test_transforms(img)
        img = img.unsqueeze(0).to(device)
        output = model(img)
        # remember softmax
        probs = F.softmax(output,dim=1)
        probs, indices = probs.topk(k=num_classes)
        probs = probs[0].tolist()
        indices = indices[0].tolist()
        classes = [idx_to_class[index] for index in indices]
        confidences = {classes[i]: round(probs[i],3) for i in range(num_classes)}  

    return confidences


examples = ["samples/a.jpeg","samples/c.jpeg","samples/r.jpeg"]
gr.Interface(fn=GetClassProbs, 
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=3),
             examples=examples).launch(share=False)





