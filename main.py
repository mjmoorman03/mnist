from mnist import MNIST
import math
import torch
from PIL import Image

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.con = torch.nn.Conv2d(1, 6, 5)
        self.fc1 = torch.nn.Linear(864, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.con(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(x.shape[0], -1)

        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        return x


def formatData(images, labels):
    data = []
    for image in images:
        subData = []
        for i in range(int(math.sqrt(len(image)))):
            subData.append(image[i*int(math.sqrt(len(image))):(i+1)*int(math.sqrt(len(image)))])
        data.append(subData)
    data = torch.FloatTensor(data)
    data = data / 255
    # batch data into groups of 60 images
    data = data.view(-1, 50, 1, 28, 28)
    labels = torch.LongTensor(labels)
    labels = labels.view(-1, 50)

    return data, labels


def train(model, data, labels):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lossFunc = torch.nn.CrossEntropyLoss()
    for epoch in range(100):
        for i, (image, label) in enumerate(zip(data, labels)):
            optimizer.zero_grad()
            output = model(image)
            loss = lossFunc(output, label)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('Epoch: {} | Batch: {} | Loss: {}'.format(epoch, i, loss.item()))


def test(model, data, labels):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (image, label) in enumerate(zip(data, labels)):
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            if i % 100 == 0:
                print('Batch: {} | Accuracy: {}'.format(i, 100 * correct / total))
    print('Accuracy: {}'.format(100 * correct / total))


def main():
    mndata = MNIST('samples')

    trainImages, trainLabels = mndata.load_training()
    trainImages, trainLabels = formatData(trainImages, trainLabels)
    # or
    testImages, testLabels = mndata.load_testing()
    testImages, testLabels = formatData(testImages, testLabels)

    model = Net()
    train(model, trainImages, trainLabels)
    model.eval()
    test(model, testImages, testLabels)
    # save model
    torch.save(model.state_dict(), 'model.pt')


def outPutToInt(output):
    return output.argmax(dim=1, keepdim=True)


# load model and test on images in 'images' folder
def singleImage():
    model = Net()
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    im = Image.open('num.png')
    pix_val = list(im.getdata())
    pix_val = [255 - ((x[0] + x[1] + x[2]) / 3) for x in pix_val]
    pix_val = torch.FloatTensor(pix_val)
    pix_val = pix_val / 255
    pix_val = pix_val.view(1, 1, 28, 28)
    output = model(pix_val)
    print(outPutToInt(output).item())


def multiImage():
    model = Net()
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    im = Image.open('num.png')
    pix_val = list(im.getdata())
    imNum = int(len(pix_val) / (28 * 28))
    pix_val = [255 - ((x[0] + x[1] + x[2]) / 3) for x in pix_val]
    newPix = []
    for i in range(imNum):
        image = []
        for j in range(28):
            for k in range(28):
                image.append(pix_val[i*28 + j*28*imNum + k])
        newPix.append(image)
    pix_val = torch.FloatTensor(newPix)
    pix_val = pix_val.view(-1, 1, 28, 28)
    pix_val = pix_val / 255
    for x in pix_val:
        print(outPutToInt(model(x.unsqueeze(0))).item(), end='')
    print('')



if __name__ == '__main__':
    multiImage()
