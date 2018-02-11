import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
#from matplotlib.widgets import Button
#%matplotlib inline
#print(plt.get_backend())
import lmfit
from laplace import Talbot
import time
import inspect

def help():
    print('Creat model: mymodel = GlobalFit()')
    print('Load data: mymodel.data()')
    print('Initialize parameters: mymodel.initialize()')
    print('Fit data: mymodel.fit()')
    print('Update parameters: mymodel.write()')
    print('Plot data: mymodel.plot(plot_sim=True)')

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
    def __init__(self, models_list=models):
        print('List of all available models:')
        for m in models_list:
            print(m)
        model = 'nstep' # default model
        model_input = input('Type the name of your model [default={}]: '.format(model))
        if model_input: model = model_input
        self.model = models_list[model]
        self.params_list = [p for p in inspect.signature(self.model).parameters.keys()]
        self.params_list.remove('s')
        self.params_list.append('b')
        self.global_params = []
        self.fixed_params = []
        self.set_global()
    
    def data(self, data='data.csv'):
        data_input = input('Data filename [default={}]: '.format(data))
        if data_input: data = data_input
        self.filename = data.split('.')[0]
        data = np.loadtxt(data, dtype='float', delimiter=',')
        self.t = data[:,0]
        self.y = data[:,1:]
        self.N = self.y.shape[1]
    
    def plot(self, plot_sim=False):
        plt.cla()
        plt.plot(self.t, self.y, 'o')
        if plot_sim:
            plt.plot(self.t_sim, self.y_sim, 'k-', lw=1)
        plt.pause(0.01)
    
    def initialize(self):
        self.init_params = lmfit.Parameters()
        for p in self.params_list:
            for i in range(self.N):
                self.init_params.add('{}_{}'.format(p, i+1), value=1)
        for p in self.global_params:
            for i in range(1,self.N):
                self.init_params['{}_{}'.format(p,i+1)].expr = '{}_1'.format(p)
        self.simulate()
        self.open_figure()
        self.tune_params()
    
    def submit(self, txt):
        for p in self.params_list:
            if p in self.global_params:
                self.init_params['{}_{}'.format(p,1)].set(value=float(self.text_box[p].text))
            else:
                for i in range(self.N):
                    self.init_params['{}_{}'.format(p,i+1)].set(value=float(self.text_box[p].text))
        self.simulate()
        self.tune_params()
    
    def open_figure(self):
        self.fig, self.ax = plt.subplots()
        self.axbox, self.text_box = {}, {}
        
    def tune_params(self):
        plt.subplots_adjust(bottom=0.2)
        self.ax.clear()
        self.ax.plot(self.t, self.y)
        self.ax.plot(self.t_sim, self.y_sim, 'k-', lw=1)
    
        count = 1
        for p in self.params_list:
            self.axbox[p] = plt.axes([0.1*count, 0.075, 0.075, 0.075])
            self.axbox[p].clear()
            self.text_box[p] = TextBox(self.axbox[p], p, initial=str(self.init_params['{}_1'.format(p)].value))
            self.text_box[p].on_submit(self.submit)
            count += 1
        plt.pause(0.01)
    
    def set_global(self):
        print('')
        print('List of parameters for the selected model: {}'.format(self.params_list))
        params_input = input('Type global parameters comma separated: ')
        if params_input: self.global_params = [p.strip() for p in params_input.split(',')]
        
    
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
            self.y_sim[:,i] = Talbot(F,self.t_sim,N=24) + b
        return self.y_sim
    
    def residuals(self, params):
        self.simulate(params=params)
        resid = self.y - self.y_sim
        return resid.flatten()
    
    def iteration(self, params, it, resid):
        if (it % 10) == 0:
            self.plot(plot_sim=True)
    
    def fit(self):
        self.set_fixed()
        t1 = time.time()
        self.result = lmfit.minimize(self.residuals, self.init_params, iter_cb=self.iteration)
        t2 = time.time()
        dt = t2 - t1
        self.result.time = dt
        print('{} ({:.2f} seconds)'.format(self.result.message, self.result.time))
        self.result.params.pretty_print()
        return self.result
    
    def write(self):
        self.init_params = self.result.params
    
    def read_params(self):
        fj = open('{}.json'.format(self.filename), 'r')
        self.init_params.load(fj)
        fj.close()
    
    def save(self):
        fj = open('{}.json'.format(self.filename), 'w+')
        self.result.params.dump(fj)
        fj.close()
        print('Saved parameters into file: {}.json'.format(self.filename))
        CSV ="\n".join([k+','+str(self.result.params[k].value)+','+str(self.result.params[k].stderr) for k in self.result.params])
        fp = open('{}_params.csv'.format(self.filename), 'w+')
        fp.write(CSV)
        fp.close()
        print('Saved parameters into file: {}_params.csv'.format(self.filename))

print('Type "help()" if you need a help.')
