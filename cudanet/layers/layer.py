'''
Created on Apr 9, 2015

@author: tim
'''
import cudanet as gpu
import numpy as np
import ctypes as ct
import logging
import os
import cPickle as pickle
import multiprocessing as mp
from time import sleep

'''
strategy pattern for torch7-like API
'''
class ActivationFunc(object):
    def __init__(self, dropout, gpu_func, gpu_func_grad):    
        self.dropout = dropout   
        self.gpu_func = gpu_func
        self.gpu_func_grad = gpu_func_grad
    def activation(self, previous_output, my_activation, my_output, useDropout): 
        self.gpu_func(previous_output, my_activation);   
        if useDropout and self.dropout > 0.0: gpu.dropout(my_activation, self.dropout, my_output)
        else: my_activation.mult(1.0-self.dropout, my_output)
        
    def grad(self, my_activation, my_output): return self.gpu_func_grad(my_activation, my_output)  
    
class Logistic(ActivationFunc): 
    def __init__(self):  
        super(Logistic, self).__init__(0.5, gpu.logistic, gpu.logistic_grad)
            
class ReLU(ActivationFunc):      
    def __init__(self):  
        super(ReLU, self).__init__(0.5, gpu.rectified_linear, gpu.rectified_linear_grad)      
        
class Input(ActivationFunc): 
    def __init__(self):
        super(Input, self).__init__(0.2, gpu.identity, gpu.identity)
        
class Linear(ActivationFunc): 
    def __init__(self):  
        super(Linear, self).__init__(0.0, gpu.identity, gpu.identity)
    def activation(self, previous_output, my_activation, my_output, useDropout): 
        self.gpu_func(previous_output, my_output); 
        
class Softmax(ActivationFunc):
    def __init__(self):   
        super(Softmax, self).__init__(0.0, gpu.softmax, gpu.identity)
    def activation(self, previous_output, my_activation, my_output, useDropout): 
        self.gpu_func(previous_output, my_output); 
        
class Code(ActivationFunc):
    def __init__(self):   
        #super(Code, self).__init__(0.0, gpu.double_ReLU, gpu.double_ReLU_grad)
        super(Code, self).__init__(0.0, gpu.identity, gpu.identity)
        #super(Code, self).__init__(0.0, gpu.logistic, gpu.logistic_grad)
    def activation(self, previous_output, my_activation, my_output, useDropout): 
        self.gpu_func(previous_output, my_output); 

'''
class GradientSynchronizer(Thread):
    def __init__(self):
        super(GradientSynchronizer, self).__init__()
        self.daemon = True
        self.cancelled = False          
        self.todo = []    
        self.idx = 0
        self.synchronizing = False
        
        pass
                
    def run(self):     
        while not self.cancel(): 
            if len(self.todo) > 0:
                self.synchronizing = True
                gpu.sync(self.todo[0][0], self.todo[0][1])
                gpu.sync_streams()
                self.todo.pop(0)
                self.idx +=1
                
            sleep(0.001)
                    
    def cancel(self):
        """End this timer thread"""        
        return self.cancelled
    
    
g = GradientSynchronizer()
'''
        
def create_uniform_rdm_weight(input_size,output_size):
    rdm = np.random.RandomState(1234)        
    return rdm.uniform(low=-4*np.sqrt(6./(input_size+output_size)),
                    high=4*np.sqrt(6./(input_size+output_size)),
                    size=(input_size,output_size))
    
def swap(A,B):
    C = A
    A = B
    B = C
    
def log_and_print(message):
    logging.info(message)
    print message    


class Layer(object):
    #def __init__(self, unitcount=0, activation_function=Logistic(), workdir = None, network_name = 'neural_net'):
    def __init__(self, unitcount=0, activation_function=None, workdir = None, network_name = 'neural_net'):        
        self.w_next = None
        self.w_next_sync = None
        self.activation = None
        self.activation_offsize = None         
        self.funcs = activation_function
        if not self.funcs: self.funcs = Input()
        self.unitcount = unitcount
        self.next_layer = None
        self.prev_layer = None
        self.id = 0
        self.target = None
        self.current_error = []
        self.current_SE = []
        self.error_epochs = {}
        self.confidence_interval_epochs = {}
        self.test_buffer = None
        #print 1./unitcount if unitcount > 0.0 else 0.003
        self.config = {'learning_rate' : 0.001,
                       'momentum' : 0.9,
                       'input_dropout': self.funcs.dropout,
                       'dropout' : self.funcs.dropout,
                       'learning_rate_decay' : 1.0,
                       'parallelism' : 'data',
                       'compression' : '8bit',
                       'dropout_decay' : True,
                       'test_for_convergence' : True,
                       'error_evaluation' : 'classification'
                       }        
        self.logger = None
        
        self.workdir = workdir
        self.network_name = network_name
        
        self.init_work_dir()
        self.epoch = 0
        
        self.has_gradients = False
        self.abs_max_grad_value = 0.0
        
    def log(self, msg, print_msg = True, level = logging.INFO):
        logging.log(level, msg)
        if print_msg: print msg
        
    def log_network(self):
        if not self.workdir: return
        i=0
        layer = self
        self.log('\n',False)
        self.log('---------------------------------------', False)
        self.log('            ' + self.network_name, False)
        self.log('---------------------------------------', False)
        self.log('\n',False)
        while True:
            self.log('Layer {0}'.format(i), False)
            self.log('---------------------------------------', False)
            for key in self.config:
                if key == 'dropout' and type(layer.funcs) is Input: continue
                if key == 'input_dropout' and type(layer.funcs) is not Input: continue
                self.log('{0}: {1}'.format(key, layer.config[key]), False)
            self.log('{0}: {1}'.format('unitcount',layer.unitcount), False)
            self.log('{0}: {1}'.format('activation function',layer.funcs.__class__.__name__), False)
            self.log('\n',False)
            i+=1
            if layer.next_layer: layer = layer.next_layer
            else: break
    
    def init_work_dir(self):
        if self.workdir:
            if not os.path.exists(self.workdir): os.mkdir(self.workdir)
            logging.basicConfig(filename=os.path.join(self.workdir,self.network_name+'_log'),format='%(asctime)s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
            
            pass
        
        
    def add(self,next_layer, logger=None):  
        if self.next_layer:    
            self.next_layer.add(next_layer,self.logger)
            return
        
        if type(next_layer) is Layer:            
            self.next_layer = next_layer
            next_layer.prev_layer = self
            self.next_layer.logger = self.logger   
            next_layer.id = self.id +1
            
        else:
            self.funcs = next_layer
    
    def create_weights(self):
        self.log_network()
        if self.next_layer:
            self.w_next = gpu.array(create_uniform_rdm_weight(self.unitcount,self.next_layer.unitcount))
            self.b_next = gpu.zeros((1, self.next_layer.unitcount))
            self.m_next = gpu.zeros((self.unitcount, self.next_layer.unitcount))
            self.w_grad_next = gpu.zeros((self.unitcount, self.next_layer.unitcount))
            self.b_grad_next = gpu.zeros((1, self.next_layer.unitcount))   
            self.w_next_sync = gpu.zeros((self.unitcount,self.next_layer.unitcount))  
            if self.next_layer.config['compression'] == '1bit':
                self.errors = gpu.zeros_like(self.w_grad_next)
                self.posMask = gpu.zeros_like(self.w_grad_next)
                self.negMask = gpu.zeros_like(self.w_grad_next)
                self.w_grad_with_errors = gpu.zeros_like(self.w_grad_next)
                self.posCount = gpu.zeros((self.w_grad_next.shape_tensor[2],))
                self.negCount = gpu.zeros((self.w_grad_next.shape_tensor[2],))
                self.posAvg = gpu.zeros((self.w_grad_next.shape_tensor[2],))
                self.negAvg = gpu.zeros((self.w_grad_next.shape_tensor[2],))
            if self.next_layer.config['compression'] == '8bit':    
                self.max_value_buffer = gpu.empty_like(self.w_grad_next)
                   
            if self.next_layer: self.next_layer.create_weights()
        
    def create_buffers(self, batch_size):
        self.activation = gpu.empty((batch_size,self.unitcount))
        self.out = gpu.empty((batch_size,self.unitcount))
        self.error = gpu.empty((batch_size,self.unitcount))
        self.bias_ones = gpu.ones((batch_size,1))
        #self.bias_ones = gpu.array(np.ones((1,batch_size)),3)
        
    def handle_offsize(self, batch_size):
        if self.activation_offsize == None:
            split_axis = (2 if self.config['parallelism'] == 'data' else -1)
            self.activation_offsize = gpu.empty((batch_size,self.unitcount),split_axis)
            self.out_offsize = gpu.empty((batch_size,self.unitcount),split_axis)
            self.error_offsize = gpu.empty((batch_size,self.unitcount),split_axis)
            self.bias_ones_offsize = gpu.zeros((batch_size,1),split_axis)+1
            swap(self.activation, self.activation_offsize)
            swap(self.out, self.out_offsize)
            swap(self.error, self.error_offsize)
            swap(self.bias_ones, self.bias_ones_offsize)            
        elif self.activation_offsize.shape[2] != batch_size:
            del self.activation
            del self.out
            del self.error
            del self.bias_ones
            self.create_buffers(batch_size)
        else:
            swap(self.activation, self.activation_offsize)
            swap(self.out, self.out_offsize)
            swap(self.error, self.error_offsize)
            swap(self.bias_ones, self.bias_ones_offsize)    
    
    def handle_input_size(self, batch_size):
        if self.w_next==None: self.create_weights()
        if self.activation == None: self.create_buffers(batch_size)
        elif self.activation.shape[2] != batch_size: self.handle_offsize(batch_size)         
        if self.next_layer: self.next_layer.handle_input_size(batch_size)
     
    @property
    def root(self):
        root = self
        while root.next_layer: root = root.next_layer
        return root    
        
    def forward(self, data=None, target=None,inTrainingMode=True):              
        if data is not None:            
            self.unitcount = data.shape[1]
            self.handle_input_size(data.shape[0])           
            self.root.target = target
            self.funcs.activation(data, self.activation, self.out, inTrainingMode)
            #if inTrainingMode: self.handle_parallelism()
        else:
            #if inTrainingMode: self.handle_parallelism()
            #cool
            gpu.dot(self.prev_layer.out,self.prev_layer.w_next,self.activation)   
            #not cool activation, dot problem? -> nope, memory problem (wrong buffer size)?            
            #print self.prev_layer.out.sum()
            #print self.prev_layer.w_next.sum()
            #print self.activation.sum()
            #print 'a'
            #sleep(0.5)    
            gpu.add(self.activation, self.prev_layer.b_next, self.activation)   
            self.funcs.activation(self.activation, self.activation, self.out, inTrainingMode)  
            
        if self.next_layer: self.next_layer.forward(None, None, inTrainingMode)

    
    def predict(self, data):
        self.forward(data, None,False)   
        if type(self.root.funcs) == Softmax:            
            if self.config['error_evaluation'] == 'classification':
                return gpu.argmax(self.root.out)              
            elif self.config['error_evaluation'] == 'logloss':
                return self.root.out
        else: return self.root.out 
        
    def backward(self):
        self.backward_errors()
        self.backward_grads()  
              
    def backward_errors(self):
        if self.next_layer: self.next_layer.backward_errors()
        else: 
            gpu.subtract(self.out,self.target,self.error)
            return
        
        if type(self.funcs) is Input: return
        
        self.funcs.grad(self.activation,self.out)
        gpu.dot(self.next_layer.error, self.w_next.T, self.error)
        gpu.multiply(self.error, self.out, self.error)
        
    def backward_grads(self):
        if self.target: return
        gpu.dot(self.activation.T, self.next_layer.error, self.w_grad_next)
        
        if self.next_layer: self.next_layer.backward_grads()        
        
        gpu.dot(self.bias_ones.T, self.next_layer.error, self.b_grad_next)
        
    def accumulate_error(self):
        if self.config['error_evaluation'] == 'classification':
            predicted_labels = gpu.argmax(self.root.out) 
            target_labels = gpu.argmax(self.root.target)
            gpu.equal(predicted_labels, target_labels, target_labels)
            #print target_labels.sum(), target_labels.shape_tensor
            
            error = 1.0-(target_labels.sum()/self.out.shape[2])
          
        elif self.config['error_evaluation'] == 'logloss':
            size =  self.root.out.shape_tensor[2]
            error = (self.root.target*gpu.log(self.root.out+1e-15)).sum()/np.float32(-size)
        elif self.config['error_evaluation'] == 'regression':
            error = gpu.sum((self.activation-self.root.out)**2)/np.float32(self.root.out.shape_tensor[2]*self.root.out.shape_tensor[3])
            
        self.current_error.append(error) 
        self.current_SE.append(np.array(self.current_error).std()/len(self.current_error))        
        
        
    def print_reset_error(self, error_name='Train'):
        error = np.array(self.current_error).mean()
        if error_name not in self.error_epochs:
            self.error_epochs[error_name] = []
            self.confidence_interval_epochs[error_name] = []
        CI_lower = error-(self.current_SE[-1]*1.96)
        CI_upper = error+(self.current_SE[-1]*1.96)        
        self.error_epochs[error_name].append(error)
        self.confidence_interval_epochs[error_name].append([CI_lower, CI_upper])
        log_and_print('{1} error: {0}\t ({2},{3})'.format(np.round(error,4),error_name, np.round(CI_lower,4),np.round(CI_upper,4)))        
        del self.current_error
        del self.current_SE
        self.current_error = []
        self.current_SE = []
        return error
        
        
    def weight_update(self):
        if self.next_layer:    
            #batch_size = ((self.out.shape[2]*gpu.gpu_count()) if self.config['parallelism'] == 'data' else self.out.shape[2])
            batch_size = self.out.shape[2]
            
            if self.has_gradients:
                #x = self.w_grad_next.tocpu()
                #x[np.abs(x) < 0.005] = 0
                #del self.w_grad_next                 
                #self.w_grad_next = gpu.array(x)
                lib.funcs.inp_RMSProp(self.m_next.pt, self.w_grad_next.pt, ct.c_float(self.config['momentum']),ct.c_float(self.config['learning_rate']), batch_size)                
                gpu.subtract(self.w_next, self.w_grad_next, self.w_next)
                
                if not self.test_buffer:
                    self.test_buffer = gpu.empty_like(self.w_next)
                
                gpu.fill(self.test_buffer, 0.005)
                gpu.greater(self.w_next, self.test_buffer, self.test_buffer)
                gpu.multiply(self.w_next, self.test_buffer, self.w_next)
                
                
            #apply grad only after initializing RMSProp with the first gradient
            if not self.has_gradients: 
                self.has_gradients = True
                #TODO: this should work
                #gpu.div(self.w_grad_next, batch_size, self.m_next)
                
                     
            if self.config['parallelism'] != 'data':
                self.next_layer.weight_update()
        
    def test_for_no_convergence(self):
        if not self.config['test_for_convergence']: return False
        if len(self.error_epochs['CV']) < 20: return False
        #this is a simple 99% confidence interval plot_data 
        #for the null hypothesis that the CV error
        #for the last 20 epochs is the same as for the last 10 epochs
        #In short: It tests if the neural net has converged
        null_hypo_mean = np.mean(self.error_epochs['CV'][-20:])
        null_hypo_SE = np.std(self.error_epochs['CV'][-20:])/20.
        null_CI = [null_hypo_mean+(2.57*null_hypo_SE),null_hypo_mean-(2.57*null_hypo_SE)]
        data_mean = np.mean(self.error_epochs['CV'][-10:])
        data_SE = np.std(self.error_epochs['CV'][-10:])/10.
        data_CI = [data_mean+(2.57*data_SE),data_mean-(2.57*data_SE)]
        if data_CI[0] > null_CI[1]: 
            self.log('Convergence detected: null hypothesis vs data: {0} vs {1}'.format(null_CI, data_CI), True)
            self.config['test_for_convergence'] = False
        return data_CI[0] > null_CI[1]
        
        
    def end_epoch(self):
        self.set_config_value('learning_rate', 0.0, 'learning_rate_decay', lambda a,b: a*b)   
        self.abs_max_grad_value = 0.0
        if self.id == 0 and self.test_for_no_convergence(): self.dropout_decay()
        if self.next_layer: self.next_layer.end_epoch()
        
    def dropout_decay(self):
        if not self.config['dropout_decay']: return
        self.config['learning_rate_decay'] = 0.85
        self.config['dropout'] *=0.5
        self.config['input_dropout'] *=0.5
        if self.prev_layer: self.funcs.dropout = self.config['dropout']
        else: 
            self.funcs.dropout = self.config['input_dropout']
            self.log("Appling dropout decay", True)
            
        if self.next_layer: self.next_layer.dropout_decay()
        
        self.log_network()
        
        
    def set_config_value(self, key, value, key2=None, func=None):
        if func and key2: self.config[key] = func(self.config[key], self.config[key2])
        else: self.config[key] = value
        if key == 'dropout' and self.prev_layer: self.funcs.dropout = value
        if key == 'input_dropout': 
            self.funcs.dropout = value
            return
            
        if self.next_layer:
            self.next_layer.set_config_value(key, value, key2, func)
            
    def save_config(self):
        self.check_work_dir()
        configs = []
        layer = self
        configs.append(layer.config)
        while layer.next_layer:
            layer = layer.next_layer
            configs.append(layer.config)
            
        pickle.dump(configs, open(os.path.join(self.workdir,'parameters.config'),'w'))
        return configs
                    
    def load_config(self):
        self.check_work_dir()
        if not os.path.exists(os.path.join(self.workdir, 'parameters.config')): 
            logging.error('Cannot load config: No config exists!')
            return
        
        configs = pickle.load(open(os.path.join(self.workdir, 'parameters.config'),'r'))
        layer = self
        layer.config = configs[0]
        i=1
        while layer.next_layer:
            layer = layer.next_layer
            layer.config = configs[i]
            i+=1
            
        return configs
        
    def check_work_dir(self):        
        if not self.workdir: 
            logging.error('Need working directory to perform this action!')
            return       
        
