import torch
import torch.nn.functional as F

def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

# regression
def loss_sample(output, target):
    with torch.no_grad():
        loss = F.mse_loss(output,target.float())
    return loss / len(target)

# hand mask loss
def loss_hand_sample(output, target):
    with torch.no_grad():
        
        loss = F.cross_entropy(output, target)
    return loss / 16

def loss_sigmoid(output, target):
    with torch.no_grad():
        loss = F.binary_cross_entropy_with_logits(output, target.float())
    return loss / 16

def softmax_cross_entropy_with_logits(output, target):
    with torch.no_grad():
        loss = torch.sum(- target * F.log_softmax(output, -1), -1)
        mean_loss = loss.mean()
    return mean_loss


# huawei dataset
def acc(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)
