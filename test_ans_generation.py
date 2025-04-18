'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify other code:
1. Replace the random classifier with your trained model.(line 69-72)
2. modify the get_label function to get the predicted label.(line 23-29)(just like Leetcode solutions, the args of the function can't be changed)

REQUIREMENTS:
- You should save your model to the path 'models/conditional_pixelcnn.pth'
- You should Print the accuracy of the model on validation set, when we evaluate your code, we will use test set to evaluate the accuracy
'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
NUM_CLASSES = len(my_bidict)

#TODO: Begin of your code
def get_label(model, model_input, device):
    # Write your code here, replace the random classifier with your trained model
    # and return the predicted label, which is a tensor of shape (batch_size,)

  

    batch_size = model_input.size(0)
    results = torch.zeros(batch_size, 4, device=device) 

    # answer = model(model_input, device)
    model.eval()
    with torch.no_grad():

        for i in range(4): #num classes
            class_labels = [i for _ in range(batch_size)]
            class_labels = torch.tensor(class_labels, device=device)
            out = model(model_input, class_labels=class_labels)
            neg_log_likelihood = discretized_mix_logistic_loss(model_input, out, batched=True)
            log_likelihood = -neg_log_likelihood
            # print(log_likelihood)
            results[:, i] = log_likelihood
    answer = torch.argmax(results, dim=1)

    return answer
# End of your code

def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        original_label = [my_bidict[item] for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])
    
    return acc_tracker.get_ratio()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='validation', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             **kwargs)

    #TODO:Begin of your code
    #You should replace the random classifier with your trained model
    model = PixelCNN(nr_resnet=5, nr_filters=108, nr_logistic_mix=10,
                    resnet_nonlinearity='concat_elu', input_channels=3)
    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to './models/conditional_pixelcnn.pth'
    #You should save your model to this path
    model_path = os.path.join(os.path.dirname(__file__), 'models/kag_1_cont_249.pth')
    print(209)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('model parameters loaded')
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()
    



  
    csv_path = os.path.join(os.path.dirname(__file__), "submission.csv")
    print(f"Writing predictions to {csv_path}")

    model.eval()
    rows = []
    all_samples = dataloader.dataset.samples  # List of (full_path, label)
    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(tqdm(dataloader)):
            imgs = imgs.to(device)
            preds = get_label(model, imgs, device).cpu().tolist()

            batch_paths = all_samples[batch_idx * args.batch_size : batch_idx * args.batch_size + len(preds)]
            rel_paths = [os.path.relpath(p, args.data_dir).replace('\\', '/') for p, _ in batch_paths]

            for path, pred in zip(rel_paths, preds):
                rows.append([path, pred])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("CSV saved ✔")

    # acc = classifier(model = model, data_loader = dataloader, device = device)
    # print(f"Accuracy: {acc}")
        
        