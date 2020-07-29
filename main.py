from src.constants import *
from src.utils import *
from src.models import *
from src.gen import *

from sys import argv
from time import time
from tqdm import tqdm

def backprop(trainloader, model, optimizer):
	total = 0
	for inputs, labels in tqdm(list(trainloader)[:100], ncols=80):
		optimizer.zero_grad()
		outputs = model(inputs)
		label_one_hot = get_one_hot(labels[0], 10)
		loss = torch.sum((outputs - label_one_hot) ** 2)
		loss.backward()
		optimizer.step()
		total += loss.item()
	return total/len(trainloader)

def accuracy(valloader, model):
	correct_count, all_count = 0, 0
	for inputs, labels in valloader:
	    with torch.no_grad():
	        probs = model(inputs).tolist()
	    pred_label = probs.index(max(probs))
	    true_label = labels.numpy()[0]
	    if(true_label == pred_label):
	      correct_count += 1
	    all_count += 1
	return correct_count/all_count

def save_model(model, optimizer, epoch, accuracy_list):
	file_path = MODEL_SAVE_PATH + "/" + model.name + "_" + str(epoch) + ".ckpt"
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(filename, model, data_type):
	lr = 0.001
	optimizer = torch.optim.Adam(model.parameters() , lr=lr, weight_decay=1e-5)
	file_path = MODEL_SAVE_PATH + "/" + filename + "_Trained.ckpt"
	if os.path.exists(file_path):
		print(color.GREEN+"Loading pre-trained model: "+filename+color.ENDC)
		checkpoint = torch.load(file_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		epoch = -1; accuracy_list = []
		print(color.GREEN+"Creating new model: "+model.name+color.ENDC)
	return model, optimizer, epoch, accuracy_list

if __name__ == '__main__':
	data_type = argv[1]
	exec_type = argv[2]

	model = eval(data_type+"()")
	model, optimizer, start_epoch, accuracy_list = load_model(data_type, model, data_type)
	trainset, valset = eval("load_"+data_type+"_data()")

	if exec_type == "train":
		for epoch in range(start_epoch+1, start_epoch+EPOCHS+1):
			print('EPOCH', epoch)
			trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
			valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True)
			loss = backprop(trainloader, model, optimizer)
			trainLoss, testAcc = float(loss), float(accuracy(valloader, model))
			accuracy_list.append((testAcc, trainLoss))
			print("Loss on train, Accuracy on test =", trainLoss, testAcc)
			if epoch % 10 == 0:
				save_model(model, optimizer, epoch, accuracy_list)
		print ("The minimum loss on test set is ", str(min(accuracy_list)), " at epoch ", accuracy_list.index(min(accuracy_list)))

		plot_accuracies(accuracy_list)
	else:
		print(model.find); start = time()
		for param in model.parameters(): param.requires_grad = False
		bounds = np.array([[0,9.5], [0,90], [1,40], [-16,16]])

		if exec_type == "ga":
			ga(dataset, model, bounds, data_type)
		elif exec_type == "opt":
			opt(dataset, model, bounds, data_type)
		elif exec_type == "data":
			least_dataset(dataset, data_type)
		print("Time", time()-start)
