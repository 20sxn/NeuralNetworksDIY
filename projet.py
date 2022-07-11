import numpy as np

#CORE
class Module:
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input):
        raise NotImplementedError
        
    def backward_delta(self,input,delta): # (2)
        raise NotImplementedError
        
    def backward_update_gradient(self,input,delta): # (1)
        raise NotImplementedError
        
    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self.w -= gradient_step*self.gradient
        
    def zero_grad(self):
        self.gradient = np.zeros(self.gradient.shape)



#MODULES
class Linear(Module):
    def __init__(self, input_size, output_size):
        self.w = np.random.rand(input_size, output_size) - 0.5
        self.gradient = np.zeros((input_size,output_size))
    
    def forward(self, input_data):
        self.input = input_data
        self.output = self.input @ self.w
        return self.output
    
    def backward_delta(self,delta):
        new_delta = delta @ self.w.T
        return new_delta
    
    def backward_update_gradient(self,delta):
        self.gradient = self.input.T @ delta

class Conv1D(Module):
    def __init__(self,k_size,chan_in,chan_out,stride=1):
        self.w = np.random.rand(chan_out,k_size, chan_in) - 0.5
        self.size = k_size
        self.stride = stride
        self.chan_out = chan_out
        self.gradient = np.zeros(self.w.shape)
        
    def forward(self,data):
        N,L,_ = data.shape
        self.input = data
        outputdim = (L-self.size) // self.stride + 1
        self.output = np.zeros((N,outputdim,self.chan_out))
        
        cpt = 0       
        for i in range(0,L,self.stride):
            if cpt == outputdim:
                break
            for j in range(N):
                window = data[j,i:i+self.size,:]
                for c in range(self.chan_out):
                    self.output[j,cpt,c] = np.sum(self.w[c]*window)
            cpt+=1
        return self.output
    
    def backward_delta(self,delta):
        outputdim = delta.shape[1]
        N,L,_ = self.input.shape
        new_delta = np.zeros(self.input.shape)
        
        cpt = 0
        for i in range(0,L,self.stride):
            if cpt == outputdim:
                break
            for c in range(self.chan_out):
                for j in range(N):
                    new_delta[j,i:i+self.size,:] += self.w[c] * delta[j,cpt,c]
            cpt += 1
        return new_delta
    
    def backward_update_gradient(self,delta):
        outputdim = delta.shape[1]
        N,L,_ = self.input.shape
        
        cpt = 0
        for i in range(0,L,self.stride):
            if cpt == outputdim:
                break
            for c in range(self.chan_out):
                for j in range(N):
                    self.gradient[c] += self.input[j,i:i+self.size,:] * delta[j,cpt,c]/N
            cpt += 1
            
class Flatten(Module):
    def forward(self, data):
        self.input = data
        N,L,C = data.shape
        self.output = data.reshape((N,L*C))
        return self.output
    
    def backward_delta(self,delta):
        N,L,C = self.input.shape
        new_delta = delta.reshape((N,L,C))
        return new_delta
    
    def backward_update_gradient(self,delta):
        return None

    def update_parameters(self, gradient_step=1e-3):
        return None
        
    def zero_grad(self):
        return None

class MaxPool1D(Module):
    def __init__(self,k_size,stride):
        self.size = k_size
        self.stride = stride
        self.argmax = None
    
    def backward_update_gradient(self,delta):
        return None

    def update_parameters(self, gradient_step=1e-3):
        return None
        
    def zero_grad(self):
        return None
    
    def forward(self,data):
        N,L,C = data.shape
        outputdim = (L-self.size) // self.stride + 1
        
        self.input = data
        self.output = np.zeros((N,outputdim,C))
        self.argmax = np.zeros((N,outputdim,C))
        
        cpt = 0
        for i in range(0,L,self.stride):
            if cpt == outputdim:
                break
            for j in range(N):
                for c in range(C):
                    window = data[j,i:i+self.size,c]
                    self.argmax[j,cpt,c] = np.argmax(window)
                    self.output[j,cpt,c] = np.max(window)
            cpt += 1
        return self.output
    
    def backward_delta(self,delta):
        N,L,C = self.input.shape
        new_delta = np.zeros((N,L,C))
        outputdim = delta.shape[1]
        
        for i in range(outputdim):
            for j in range(N):
                for c in range(C):
                    new_delta[j,int(i*self.stride + self.argmax[j,i,c]),c] = delta[j,i,c]
        
        return new_delta

#ACTIVATIONS
class Tanh(Module):
    def forward(self,data):
        self.input = data
        self.output = np.tanh(data)
        return self.output
    
    def backward_update_gradient(self,delta):
        return None

    def update_parameters(self, gradient_step=1e-3):
        return None
        
    def zero_grad(self):
        return None
        
    def backward_delta(self,delta): # (2)
        grad = 1-np.tanh(self.input)**2
        return grad * delta


class Sigmoide(Module):
    def forward(self,data):
        self.input = data
        self.output = 1/(1+np.exp(-data))
        return self.output

    def backward_update_gradient(self,delta):
        return None

    def update_parameters(self, gradient_step=1e-3):
        return None
        
    def zero_grad(self):
        return None
    
    def backward_delta(self,delta): # (2)
        grad = 1/(1+np.exp(-self.input))*(1-1/(1+np.exp(-self.input)))
        return grad * delta


class Identity(Module):
    def forward(self,data):
        self.input = data
        self.output = self.input
        return self.output

    def backward_update_gradient(self,delta):
        return None

    def update_parameters(self, gradient_step=1e-3):
        return None
        
    def zero_grad(self):
        return None
    
    def backward_delta(self,delta):
        grad = 1
        return grad * delta



class ReLU(Module):
    def forward(self,data):
        self.input = data
        self.output = np.maximum(self.input,0)
        return self.output

    def backward_update_gradient(self,delta):
        return None

    def update_parameters(self, gradient_step=1e-3):
        return None
        
    def zero_grad(self):
        return None
    
    def backward_delta(self,delta):
        grad = np.where(self.input>0,1,0)
        return grad * delta

class SoftMax(Module):
    def forward(self, data):
        self.input = data
        exp = np.exp(data)
        self.output = exp/np.sum(exp,axis = 1).reshape(-1,1)
        return self.output

    def backward_update_gradient(self,delta):
        return None

    def update_parameters(self, gradient_step=1e-3):
        return None
        
    def zero_grad(self):
        return None

    def backward_delta(self, delta):
        soft = self.output
        new_delta = delta * (soft * (1-soft))
        return new_delta

class LogSoftMax(Module):
    def forward(self, data):
        self.input = data
        #self.input = data - np.max(data, axis=1, keepdims=True)
        exp = np.exp(self.input)
        self.output = self.input - np.log(np.sum(exp,axis = 1).reshape(-1,1))
        return self.output

    def backward_update_gradient(self,delta):
        return None

    def update_parameters(self, gradient_step=1e-3):
        return None
        
    def zero_grad(self):
        return None

    def backward_delta(self, delta):
        exp = np.exp(self.input)
        soft = exp/np.sum(exp,axis = 1).reshape(-1,1)
        new_delta = delta * (1-soft)
        return new_delta

#LOSS
class Loss(object):
    def forward(self,y,yhat):
        raise NotImplementedError
    def backward_loss(self,y,yhat):
        raise NotImplementedError


class MSELoss(Loss):
    def forward(self,y_true,y_pred):
        return np.mean((y_true-y_pred)**2)
    
    def backward_loss(self,y_true,y_pred):
        return 2*(y_pred-y_true)/y_true.size


class CELoss(Loss):
    def forward(self,y_true,y_pred):
        return -y_true* y_pred + np.log(np.sum(np.exp(y_pred)))
    
    def backward_loss(self,y_true,y_pred):
        exp = np.exp(y_pred)
        softm = exp / np.sum(exp,axis=1).reshape(-1,1)
        
        return (softm - y_true)/y_true.size

class OGCELoss(Loss):
    def forward(self,y_true,y_pred):
        return 1 - np.sum(y_true * y_pred, axis = 1)

    def backward_loss(self,y_true,y_pred):
        return y_pred - y_true


class BCELoss(Loss):
    def forward(self,y_true,y_pred):
        eps = 1e-300
        yhat = np.where(y_pred < eps, eps, y_pred)
        yhat = np.where(yhat > 1-eps, 1-eps, yhat)
        return -((1-y_true) * np.log(1-yhat) + y_true * np.log(yhat))
    
    def backward_loss(self,y_true,y_pred):
        eps = 1e-300
        yhat = np.where(y_pred < eps, eps, y_pred)
        yhat = np.where(yhat > 1-eps, 1-eps, yhat)
        
        return  ((1-y_true)/(1-yhat))-(y_true/(yhat))



#ENCAPSULATION
class Sequentiel(object):
    def __init__(self,Loss):
        self.modules = []
        self.loss = Loss
    
    def add(self, mod):
        self.modules.append(mod)
    
    def build_network(self,layer_sizes,activations):
        S = len(layer_sizes) - 1
        assert len(activations) == S
        for i in range(S):
            self.add(Linear(layer_sizes[i],layer_sizes[i+1]))
            self.add(activations[i])

    def forward(self,data,verbose = False):
        N = len(data)
        pred = []
        for i in range(N):
            if verbose: print(str(i+1)+"/"+str(N))
            x = data[i]
            for mod in self.modules:
                x = mod.forward(x)
            pred.append(x)
        return pred
    def backward(self, datay,eps):
        pred = self.modules[-1].output
        delta = (self.loss).backward_loss(datay,pred)
        for mod in reversed(self.modules):
            new_delta = mod.backward_delta(delta)
            mod.backward_update_gradient(delta)
            
            mod.update_parameters(gradient_step = eps)
            delta = new_delta
            mod.zero_grad()

class Optim(object):
    def __init__(self,net,eps,loss = None):
        self.net = net
        if loss == None:
            self.loss = net.loss
        else:
            self.loss = loss
        self.eps = eps
    
    def step(self,batch_x,batch_y,lr = None): #liste de batchs
        if lr is not None:
            self.eps = lr
        N = len(batch_x)
        err = 0
        
        for i in range(N): #for each batch
            z = batch_x[i]
            for mod in self.net.modules:
                z = mod.forward(z)

            #delta = self.loss.backward_delta(batch_y[i],z)
            self.net.backward(batch_y[i],self.eps)
            
            err += np.mean(self.loss.forward(batch_y[i],z))
        return err/N


#MINI BATCH GRADIENT DESCENT
def SGD(opt,batch_x,batch_y,nb_batch,n_epochs,lr = None,verbose = True):
    n = batch_y.shape[0]
    ind = np.arange(n)
    
    for i in range(n_epochs):
        if verbose : print("epoch : "+str(i+1)+"/"+str(n_epochs))
        np.random.shuffle(ind)
        splitind = np.array_split(ind,nb_batch)
        
        X_split = [batch_x[j] for j in splitind]
        Y_split = [batch_y[j] for j in splitind]
        err = opt.step(X_split,Y_split,lr)

        if verbose : print("erreur : ",err)


#ENCODER
class AutoEncoder(Sequentiel):
    def __init__(self,Loss):
        self.modules = []
        self.loss = Loss
        self.encoder = []
        self.decoder = []
        
    def build(self,encoder_sizes,decoder_sizes,encoder_act,decoder__act):
        ES = len(encoder_sizes)-1
        assert len(encoder_act) == ES
        DS = len(decoder_sizes)-1
        assert len(decoder__act) == DS
        assert encoder_sizes[-1] == decoder_sizes[0] #sortie encoder = entree decoder
        assert decoder_sizes[-1] == encoder_sizes[0] #sortie decoder ~ entree encoder
        
        for i in range(ES):
            self.add(Linear(encoder_sizes[i],encoder_sizes[i+1]))
            self.encoder.append(self.modules[-1])
            self.add(encoder_act[i])
            self.encoder.append(self.modules[-1])
            
        for i in range(DS):
            self.add(Linear(decoder_sizes[i],decoder_sizes[i+1]))
            self.decoder.append(self.modules[-1])
            self.add(decoder__act[i])
            self.decoder.append(self.modules[-1])
    
    def encode(self,data):
        N = len(data)
        pred = []
        for i in range(N):
            x = data[i]
            for mod in self.encoder:
                x = mod.forward(x)
            pred.append(x)
        return pred
    
    def decode(self,data):
        N = len(data)
        pred = []
        for i in range(N):
            x = data[i]
            for mod in self.decoder:
                x = mod.forward(x)
            pred.append(x)
        return pred