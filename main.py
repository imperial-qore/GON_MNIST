from src.constants import *
from src.utils import *
from src.models import *
from src.gen import *

from sys import argv
from tqdm import tqdm


def augment(trainloader):
	trainlist = list(trainloader)
	for i in tqdm(list(range(N_CLASSES)), ncols=80, desc='Augmenting data'):
		data, labels, _ = gen(model, data_type, trainset, num_examples=BATCH_SIZE//N_CLASSES, label=i)
		trainlist.extend(list(zip(data, labels)))
	random.shuffle(trainlist)
	return trainlist

def backprop(trainloader, model, optimizer):
	total = 0
	l = nn.CrossEntropyLoss()
	for inputs, labels in tqdm(trainloader, ncols=80, desc='Training'):
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = l(outputs.view(1,-1), labels)
		loss.backward()
		optimizer.step()
		total += loss.item()
	return total/len(trainloader)

def accuracy(valloader, model):
	correct_count, all_count = 0, 0
	for inputs, labels in tqdm(list(valloader, ncols=80, desc='Evaluating')):
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
	lr = 0.00001
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
		valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True)
		for epoch in range(start_epoch+1, start_epoch+EPOCHS+1):
			print('EPOCH', epoch)
			indices = np.random.randint(0, len(trainset), BATCH_SIZE)   
			sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
			trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, sampler=sampler)
			loss = backprop(augment(trainloader), model, optimizer)
			trainLoss, testAcc = float(loss), float(accuracy(valloader, model))
			accuracy_list.append((testAcc, trainLoss))
			print("Loss on train, Accuracy on test =", trainLoss, testAcc)
			if epoch % 10 == 0:
				save_model(model, optimizer, epoch, accuracy_list)

		plot_accuracies(accuracy_list)
	else:
		for param in model.parameters(): param.requires_grad = False

		if "mnist" in data_type:
			label = 2; print(label)
			data, _, diffs = gen(model, data_type, trainset, num_examples=5, label=label, epsilon=1e-5)
			best = data[diffs.index(max(diffs))].data.view(1,28,28).numpy().squeeze()
			plot_image(best, label)
