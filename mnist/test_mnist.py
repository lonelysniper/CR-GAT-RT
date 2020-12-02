'''
Code modified from pytorch MNIST example, and Madry's adversarial attack framework.
'''
from __future__ import print_function
import argparse
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
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
            grad1d=grad.view(-1,784)
            a,b=torch.topk(grad1d,20,1)
            sp=a.permute(1, 0)
            for i in range(1000):
                md=grad[i].max()*0.55
                grad[i]=torch.clamp(grad[i],0,md)      #clip too high
                grad[i]/=md
            smoothed_grad+=grad          
        smoothed_grad/=n                                    #calculate smooth grad      
        return smoothed_grad
        
def get_adv_examples(data, target, model, lossfunc, eps, step_size, iterations):
    global cnt
    cnt+=1
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
    global cnt
    cnt+=1
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
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x=(x-0.1307)/0.3081          #normalise to (x-mean)/std
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.CrossEntropyLoss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

from torchvision import transforms

def save_(ts,str):
    unloader = transforms.ToPILImage()
    image = ts.detach().cpu().clone()  # clone the tensor
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save(str)


def test(model, device, test_loader):
    model.eval()
    correct = 0
    correct2=0
    bat=0
    loss=nn.CrossEntropyLoss()
    for data, target in test_loader:
        data,target=data.to(device), target.to(device)
        data=data*0.3081+0.1307          #renormalise to [0,1]
    
        adv=get_adv_examples(data,target,model,loss,10,1,15)        #PGD
        adv2=get_adv_examples_S(data,target,model,loss,10,1,15)      #CR Guided
        for chose in range(30):
          save_((data[chose]-data[chose].min())/(data[chose].max()-data[chose].min()),'figs/data{}{}.jpg'.format(bat,chose))
          save_((adv[chose]-adv[chose].min())/(adv[chose].max()-adv[chose].min()),'figs/adv{}{}.jpg'.format(bat,chose))
          save_((adv2[chose]-adv2[chose].min())/(adv2[chose].max()-adv2[chose].min()),'figs/adS{}{}.jpg'.format(bat,chose))
        output = model(adv)
        output2=model(adv2)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        pred2=output2.argmax(dim=1,keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        correct2 += pred2.eq(target.view_as(pred2)).sum().item()
        bat+=1
        
    print('PGD Accuracy: {}/{} ({:.0f}%)\n'.format(
       correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Our Accuracy: {}/{} ({:.0f}%)\n'.format(
       correct2, len(test_loader.dataset),
        100. * correct2 / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    '''
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    '''
    checkpoint = torch.load("mnist_cnn.pth")
    model.load_state_dict(checkpoint)
    test(model, device, test_loader)  

if __name__ == '__main__':
    main()