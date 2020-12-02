'''Train CIFAR10 with PyTorch.'''
#Code modified from https://github.com/kuangliu/pytorch-cifar Resnet18
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnet import *
from utils import progress_bar
class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size, use_grad=True,p=1):
        '''
        Initialize the attacker step with a given perturbation magnitude.
        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad
        self.p=p 

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set
        Args:
            ch.tensor x : the input to project back into the feasible set.
        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).
        Parameters:
            g (ch.tensor): the raw gradient
        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x

class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=self.p, dim=0, maxnorm=self.eps)
        return torch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x, g):
        """
        """
        l = len(x.shape) - 1
        g_norm = torch.norm(g.view(g.shape[0], -1),self.p,dim=1).view(-1, *([1]*l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        l = len(x.shape) - 1
        rp = torch.randn_like(x)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
        return torch.clamp(x + self.eps * rp / (rp_norm + 1e-10), 0, 1)


def getsmooth(model,data):
        n=10
        k=0
        smoothed_grad=0
        for j in range(n):
            data2=data+0.1*torch.randn(data.shape).cuda()
            data2=data2.clone().requires_grad_()
            output = model(data2)

            Mc,_=output.max(dim=1,keepdim=True)
            grad,=torch.autograd.grad(Mc.sum(),[data2])
            grad=torch.abs(grad)
            grad=grad.permute(1,2,3,0)
            grad=grad[0]+grad[1]+grad[2]
            grad=grad.unsqueeze(0)
            grad=grad.permute(3,0,1,2)
            grad1d=grad.view(-1,32*32)
            a,b=torch.topk(grad1d,20,1)
            sp=a.permute(1, 0)
            for i in range(len(grad)):
                md=grad[i].max()*0.5
                grad[i]=torch.clamp(grad[i],0,md)      #clip too high
                grad[i]/=md
            smoothed_grad+=grad          
        smoothed_grad/=n                                    #calculate smooth grad      
        return smoothed_grad
        
def get_adv_examples(data, target, model, lossfunc, eps, step_size, iterations):
    global cnt
    m = 1  # for untargeted 1
    iterator = range(iterations)
    x = data.clone()
    step_count=0


    step = L2Step(x, eps, step_size)
    for _ in iterator:
        x = x.clone().detach().requires_grad_(True)
        step_count+=1
        output = model(x)
       
        losses = lossfunc(output, target.cuda())

        loss = torch.mean(losses)

        if step.use_grad:
            grad, = torch.autograd.grad(m * loss, [x])
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)

    return x
def getw(output):                  #calculate CR
    out = output.permute(0, 2, 3, 1)
    values, indices = torch.topk(out, 1)      #for this,only need 1
    vl = values.permute(3, 0, 1, 2)
    vl = torch.clamp(vl, 1e-8, 0.999998)      #clip to avoid nan

    from scipy.stats import norm
    vl = 2*norm.ppf(vl[0].detach().cpu())    #inverse_phi(x)-inverse_phi(1-x)
    vl = torch.from_numpy(vl).float()
    return vl.unsqueeze(1)
def get_adv_examples_S(data, target, model, lossfunc, eps, step_size, iterations):
    m = 1  # for untargeted 1
    iterator = range(iterations)
    x = data.clone()
    step_count=0
    lastx=0
    
    array=[]
    step = L2Step(x, eps, step_size)
    for _ in iterator:
        x = x.clone().detach().requires_grad_(True)
        step_count+=1

        output = model(x)
        CR = getw( getsmooth(model,x))

        array.append(CR.clone().cpu())
        
        losses = lossfunc(output, target.cuda())

        loss = torch.mean(losses)
        
        if step.use_grad:
            grad, = torch.autograd.grad(m * loss, [x])
        if(step_count!=1):
            mp=array[step_count-2]-array[step_count-1]
            grad=grad.mul(torch.abs(mp).cuda())
            
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
    return x

from torchvision import transforms

def save_(ts,str):
    unloader = transforms.ToPILImage()
    image = ts.detach().cpu().clone()  # clone the tensor
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save(str)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(model,device,testloader):
    global best_acc
    model.eval()
    correct = 0
    correct2=0
    bat=0
    loss=nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(testloader):
            data,target=data.to(device), target.to(device)
            
            adv=get_adv_examples(data,target,model,loss,2.5,0.5,7)        #PGD
            adv2=get_adv_examples_S(data,target,model,loss,2.5,0.5,7)      #CR Guided
            output = model(adv)
            output2=model(adv2)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred2=output2.argmax(dim=1,keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct2 += pred2.eq(target.view_as(pred2)).sum().item()
           
            total+=len(data)
            bat+=1
    print(correct/total,correct2/total) 
                        
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=500, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
net.set_mean_std([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)



state = torch.load("resnet18_cifar.pth")
checkpoint=state['net']
net.load_state_dict(checkpoint)
test(net, device, testloader)  

#for epoch in range(start_epoch, start_epoch+200):
#    train(epoch)
#    test(epoch)
#    scheduler.step()
