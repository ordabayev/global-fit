import numpy as np
import matplotlib
import matplotlib.pyplot as plt
if not plt.get_backend():
    matplotlib.use('tkagg')
#from matplotlib.widgets import TextBox
#from matplotlib.widgets import Button
#%matplotlib inline
import lmfit
from laplace import Talbot
import time
import inspect

def help():
    print('Create a fitting model: mymodel = GlobalFit()')
    print('Print model parameters: mymodel.params()')
    print('Plot the data: mymodel.plot()')
    print('Fit the data: mymodel.fit()')
    print('Update model parameters: mymodel.update()')
    print('Save model parameters: mymodel.save()')


def doc():
    print('class GlobalFit(models_list=models)')
    print('      Parameters: models_list - dict containing modeling functions')
    print('')
    print('      data(data="data.csv")')
    print('        Load data saved in csv format.')
    print('      plot(plot_sim=False)')
    print('        Plot data and simulation curves.')
    print('')
    print('      Attribute Name  Description')
    print('      t               time')
    print('      y               data')
    print('      N               number of experiments')

# Define models here ...
models = {}

# Use 's' as a Laplace variable and don't use 'b' (it is later used for offset)
#
#

### TRANSLOCATION MODELS ###
def twostepdiss(s, A, n, kt, kd, kc, kend, r, C):
    return A/(1+n*r) * (1/(s+kc) * (1+kt*r/(s+kd)*(1-(kt/(s+kt+kd))**n))*(1+C*kc/(s+kend))) 
models['twostepdiss'] = twostepdiss

def BCDtrans(s, A, nt, nu, kt, ku, kend):
    return A*(kt**(nt))*(ku**(nu))/((s+kt)**(nt)*(s+ku)**(nu)*(s+kend))
    #return A*(kt**(L/mt))*(ku**(L/mu))/((s+kt)**(L/mt)*(s+ku)**(L/mu)*(s+kend))
models['BCDtrans'] = BCDtrans

### UNWINDING MODELS ###
def nstep(s, n, k, A):
    return A * k**n / (s * (k + s)**n)
models['nstep'] = nstep

def knp_nstep(s, n, k, A, knp, xp):
    # Extract parameters
    return A * k**n / (s * (k + s)**n) * (knp + s*xp) / (knp + s)
models['knp_nstep'] = knp_nstep

def twophase(s, n1, k1, A1, n2, k2, A2):
    return (A1 * k1**n1 / (s * (k1 + s)**n1) + A2 * k2**n2 / (s * (k2 + s)**n2))
models['twophase'] = twophase

def kc_nstep(s, n, k, kc, A):
    return A * k**n * kc / (s * (k + s)**n * (kc + s))
models['kc_nstep'] = kc_nstep

class GlobalFit(object):
    """
    Comments here
    """
    def __init__(self, data='data.csv', model='twostepdiss', models_list=models):
        self.load_data(data=data)
        self.select_model(model=model)
        self.set_global()
        self.initialize()
        self.simulate()
        self.plot()

    def load_data(self, data='data.csv'):
        print('')
        print('Load your data from a csv file.')
        print('Order of columns must be: time, y1, y2, ...')
        data_input = input('Data filename [default={}]: '.format(data))
        if data_input: data = data_input
        self.filename = data.split('.')[0]
        ty = np.loadtxt(data, dtype='float', delimiter=',')
        self.t = ty[:,0]
        self.y = ty[:,1:]
        self.N = self.y.shape[1]

    def select_model(self, model='twostepdiss', models_list=models):
        print('')
        print('Select a model to fit your data.')
        print('List of all available models:')
        for m in models_list:
            print(m)
        model_input = input('Type the name of your model [default={}]: '.format(model))
        if model_input: model = model_input
        self.model = models_list[model]
        self.params_list = [p for p in inspect.signature(self.model).parameters.keys()]
        self.params_list.remove('s')
        self.params_list.append('b')
        self.global_params = []
        self.fixed_params = []
        
    def set_global(self):
        print('')
        print('List of parameters for the selected model: {}'.format(self.params_list))
        params_input = input('Type global parameters comma separated: ')
        if params_input: self.global_params = [p.strip() for p in params_input.split(',')]
        
    def plot(self, plot_sim=True):
        plt.cla()
        plt.plot(self.t, self.y, 'o')
        if plot_sim:
            plt.plot(self.t_sim, self.y_sim, 'k-', lw=1.5)
        plt.pause(0.01)
        
    def params(self):
        self.init_params.pretty_print()
    
    def initialize(self):
        self.init_params = lmfit.Parameters()
        for p in self.params_list:
            if p in self.global_params:
                param = 0
                param_input = input('{} [default={}]: '.format(p, param))
                if param_input: param = float(param_input)
                self.init_params.add('{}_{}'.format(p, 1), value=param)
                for i in range(1,self.N):
                    self.init_params.add('{}_{}'.format(p, i+1), expr='{}_1'.format(p))
            else:
                for i in range(self.N):
                    param = 0
                    param_input = input('{}_{} [default={}]: '.format(p, i+1, param))
                    if param_input: param = float(param_input)
                    self.init_params.add('{}_{}'.format(p, i+1), value=param)
        self.simulate()
        self.plot()
                    
    
    def set_params(self):
        for p in self.params_list:
            if p in self.global_params:
                param = self.init_params['{}_{}'.format(p, 1)].value
                param_input = input('{} [default={}]: '.format(p, param))
                if param_input: param = float(param_input)
                self.init_params.add('{}_{}'.format(p, 1), value=param)
                for i in range(1,self.N):
                    self.init_params.add('{}_{}'.format(p, i+1), expr='{}_1'.format(p))
            else:
                for i in range(self.N):
                    param = self.init_params['{}_{}'.format(p, i+1)].value
                    param_input = input('{}_{} [default={}]: '.format(p, i+1, param))
                    if param_input: param = float(param_input)
                    self.init_params.add('{}_{}'.format(p, i+1), value=param)
        self.simulate()
        self.plot()
        
    
    def set_fixed(self):
        print('')
        print('List of parameters for the selected model: {}'.format(self.params_list))
        params_input = input('Type fixed parameters comma separated: ')
        if params_input: self.fixed_params = [p.strip() for p in params_input.split(',')]
        for p in self.params_list:
            if p in self.fixed_params:
                for i in range(self.N):
                    self.init_params['{}_{}'.format(p,i+1)].set(vary=False)
            elif p in self.global_params:
                self.init_params['{}_{}'.format(p,1)].set(vary=True)
            else:
                for i in range(self.N):
                    self.init_params['{}_{}'.format(p,i+1)].set(vary=True)
                    
    def set_min(self, x=0):
        print('')
        print('List of parameters for the selected model: {}'.format(self.params_list))
        params_input = input('Set min={} for: '.format(x))
        min_params = []
        if params_input: min_params = [p.strip() for p in params_input.split(',')]
        for p in min_params:
                for i in range(self.N):
                    self.init_params['{}_{}'.format(p,i+1)].set(min=x)
    
    def set_max(self, x=0):
        print('')
        print('List of parameters for the selected model: {}'.format(self.params_list))
        params_input = input('Set max={} for: '.format(x))
        max_params = []
        if params_input: max_params = [p.strip() for p in params_input.split(',')]
        for p in max_params:
                for i in range(self.N):
                    self.init_params['{}_{}'.format(p,i+1)].set(max=x)

    def simulate(self, t=None, params=None):
        if t is None: self.t_sim = self.t
        if params is None: params = self.init_params
        self.y_sim = 0.0*self.y
        for i in range(self.N):
            p_list = []
            for p in self.params_list:
                if p != 'b': p_list.append(params['{}_{}'.format(p, i+1)].value)
                else: b = params['{}_{}'.format(p, i+1)].value
            params_tuple = tuple(p_list)
            F = lambda s: self.model(s, *params_tuple)
            self.y_sim[:,i] = Talbot(F,self.t_sim,N=50) + b
        return self.y_sim
    
    def residuals(self, params):
        self.simulate(params=params)
        resid = self.y - self.y_sim
        return resid.flatten()
    
    def iteration(self, params, it, resid):
        if (it % 10) == 0:
            self.plot(plot_sim=True)
    
    def fit(self, live=False):
        self.set_fixed()
        t1 = time.time()
        if live:
            self.result = lmfit.minimize(self.residuals, self.init_params, iter_cb=self.iteration)
        else:
            self.result = lmfit.minimize(self.residuals, self.init_params)
        t2 = time.time()
        dt = t2 - t1
        self.result.time = dt
        self.result.params.pretty_print()
        print('{} ({:.2f} seconds)'.format(self.result.message, self.result.time))
        print('Chi-square_{}'.format(self.result.chisqr))
        self.plot()
        return self.result
    
    def update(self):
        self.init_params = self.result.params
    
    def read_params(self):
        fj = open('{}.json'.format(self.filename), 'r')
        self.init_params.load(fj)
        fj.close()
        self.simulate()
        self.plot()
    
    def save(self):
        fj = open('{}.json'.format(self.filename), 'w+')
        self.result.params.dump(fj)
        fj.close()
        print('Saved parameters into a file: {}.json'.format(self.filename))
        CSV ="\n".join([k+','+str(self.result.params[k].value)+','+str(self.result.params[k].stderr) for k in self.result.params])
        fp = open('{}_params.csv'.format(self.filename), 'w+')
        fp.write(CSV)
        fp.close()
        print('Saved parameters into a file: {}_params.csv'.format(self.filename))
        self.simulate()
        sim_data = np.hstack((self.t_sim[:,np.newaxis], self.y_sim))
        np.savetxt('{}_sim.csv'.format(self.filename), sim_data, delimiter=',')
        print('Saved simulated traces into a file: {}_sim.csv'.format(self.filename))

print('Type "help()" if you need a help.')
