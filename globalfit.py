'''
GLOBAL NON-LINEAR LEAST-SQUARES MINIMIZATION PROGRAM

Globalfit is a wrapper around lmfit (https://lmfit.github.io/lmfit-py) providing an interface
for multiple curves fitting with global parameters.

version: 1.1
last-update: 2018-Aug-18
author: Yerdos Ordabayev
        Dept. of Biochemistry and Molecular Biophysics
        Washington University School of Medicine
        Saint Louis, MO 63110
'''

import numpy as np
import matplotlib.pyplot as plt
import lmfit
import inspect
import itertools
import sys
import corner
from multiprocessing import Pool
import os

from models import models

class GlobalModel:
    
    def __init__(self, func=None, independent_vars=None, global_params=None, data=None):
        # choose a user-defined model
        if func is None:
            self.func = self._select_func()
        else:
            self.func = func
        self._name = self.func.__name__

        # load data
        if data is None:
            self.data = self._load_data()
        else:
            self.data = data
        self.N = self.data.shape[1] - 1
        
        # create parameters
        self.independent_vars = independent_vars
        self._param_names = []
        self._parse_params()
        if global_params is None:
            self.global_params = self._set_global()
        else:
            self.global_params = list(global_params)
        
        if os.path.isfile(os.path.join(self.path, 'params.gss')):
            self.params = self.read()
        else:
            self.params = self._make_params()
        
        # initiate program
        self.rss = []
        self._eval()
        self.plot()
        self._menu()
        
    def __repr__(self):
        '''Return representation of GlobalModel.'''
        return '{}.{}(func={})'.format(self.__module__, self.__class__.__name__, self._name)
    
    def _menu(self):
        print('----------------------------------------------------------------------------')
        print('| .fit() | .write() | .read() | .plot() | | .report() | .save() | .emcee() |')
        print('----------------------------------------------------------------------------')

    def _select_func(self):
        '''Select a function from the models list.'''
        print('User-built models:')
        for i, m in enumerate(models):
            print('{}. {}'.format(i+1, m))
        func_input = input('Type the name of your model: ')
        if func_input in models:
            return models[func_input]
        else:
            raise KeyError('{!r} is not found in the list of models'.format(func_input))
            
    def _load_data(self):
        data_input = input('DATA folder name: ')
        self.path = data_input
        if data_input:
            return np.loadtxt(os.path.join(self.path, 'data.csv'), dtype='float', delimiter=',')
        
    def _parse_params(self):
        '''Build parameters from function argumets.'''
        if self.func is None:
            return
        self._param_names = list(inspect.signature(self.func).parameters)
        if self.independent_vars is None:
            self.independent_vars = self._param_names[0]
        self._param_names.remove(self.independent_vars)

    def _set_global(self):
        '''Set global parameters for the Model.'''
        print('MODEL parameters: {}'.format(self._param_names))
        params_input = input('GLOBAL parameters (comma separated): ')
        return [p.strip() for p in params_input.split(',') if p.strip() in self._param_names]
    
    def _make_params(self):
        '''Create a Parameters object for a Model.'''
        params = lmfit.Parameters()
        for name in self._param_names:
            if name in self.global_params:
                param = 0
                param_input = input('{} [default={}]: '.format(name, param))
                if param_input: param = float(param_input)
                params.add('{}_{}'.format(name, 1), value=param)
                for i in range(1,self.N):
                    params.add('{}_{}'.format(name, i+1), expr='{}_1'.format(name))
            else:
                for i in range(self.N):
                    param = 0
                    param_input = input('{}_{} [default={}]: '.format(name, i+1, param))
                    if param_input: param = float(param_input)
                    params.add('{}_{}'.format(name, i+1), value=param)
        
        PARAMS = '{:<10}{:<16}{:<16}{:<16}\n'.format('Name', 'Value', 'Min', 'Max') \
              + '\n'.join(['{:<10}'.format(k)+'{:<16.8e}'.format(params[k].value)+'{:<16.8e}'.format(params[k].min)+'{:<16.8e}'.format(params[k].max) for k in params])
        fp = open(os.path.join(self.path, 'params.gss'), 'w+')
        fp.write(PARAMS)
        fp.close()
        print('Created guess parameters file: params.gss')
        return params

    '''Fit'''
    def fit(self, verbose=False):
        '''Fit the model to the data.'''
        if self.params is None:
            self.params = self._make_params()
        self._set_fixed()
        self.thinking = itertools.cycle(['.', '..', '...', '....', '.....'])
        self.result = lmfit.minimize(self._residual, self.params, method='leastsq', nan_policy='omit', iter_cb=self._iteration)
        print('\nParameters fit values:')
        self.result.params.pretty_print()
        print('Chi2 {:.8e}'.format(self.result.chisqr))
        self._eval(params=self.result.params)
        self.plot()
        self._menu()
        
    def _eval(self, params=None):
        '''Evaluate the model with supplied parameters.'''
        if params is None:
            params = self.params
        self.y_sim = np.zeros_like(self.data[:,1:])
        for i in range(self.N):
            kwargs = {name.split('_')[0]: par.value for name, par in params.items() if name.endswith('_{}'.format(i+1))}
            kwargs[self.independent_vars] = self.data[:,0]
            self.y_sim[:,i] = self.func(**kwargs)
        #return self.y_sim
    
    def _set_fixed(self):
        '''Set fixed parameters for the fitting.'''
        print('MODEL parameters: {}'.format(self._param_names))
        params_input = input('FIXED parameters (comma separated): ')
        self.fixed_params = [p.strip() for p in params_input.split(',') if p.strip() in self._param_names]
        for name in self._param_names:
            if name in self.fixed_params:
                for i in range(self.N):
                    self.params['{}_{}'.format(name,i+1)].set(vary=False)
            elif name in self.global_params:
                self.params['{}_{}'.format(name,1)].set(vary=True)
            else:
                for i in range(self.N):
                    self.params['{}_{}'.format(name,i+1)].set(vary=True)
        
    def _residual(self, params):
        '''Return the residual.'''
        self._eval(params=params)
        diff = self.y_sim - self.data[:,1:]
        return diff.flatten()
    
    def _iteration(self, params, it, resid):
        '''have some fun while fitting'''
        rss = np.sum(resid**2)
        self.rss.append(rss)
        char = next(self.thinking)
        sys.stdout.write('\rFitting ' + char)
        #sys.stdout.write('\rRSS: ' + str(rss))
    
    '''Bayesian credible region estimation using MCMC'''
    def emcee(self, burn=300, steps=1000, thin=20):
        self.params.add('noise', value=self.result.chisqr, min=0.0001, max=2)
        mini = lmfit.Minimizer(self._log_posterior, self.params)
        with Pool() as pool:
            self.posterior = mini.emcee(burn=burn, steps=steps, thin=thin, workers=pool)
        corner.corner(self.posterior.flatchain, quantiles=[0.05, 0.5, 0.95], labels=self.posterior.var_names, truths=list(self.posterior.params.valuesdict().values()), show_titles=True)
        plt.savefig(os.path.join(self.path, 'posterior.png'), dpi=300)
        print("median of posterior probability distribution")
        print('--------------------------------------------')
        lmfit.report_fit(self.posterior.params)
        self._menu()
        
    def _log_posterior(self, params):
        noise = params['noise']
        return -0.5 * np.sum((self._residual(params) / noise)**2 + np.log(2 * np.pi * noise**2))
        
    def write(self):
        '''Update parameters and simulations.'''
        self.params = self.result.params
        self._eval()
        self._menu()
        
    def plot(self, plot_sim=True):
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(self.data[:,0], self.data[:,1:], 'o')
        if plot_sim:
            plt.plot(self.data[:,0], self.y_sim, 'k-', lw=1.5)
            #plt.plot(self.t, self.y-self.y_sim, 'o')
        plt.title('Data')
        plt.subplot(1,2,2)
        plt.plot(self.data[:,0], self.data[:,1:]-self.y_sim, 'o')
        plt.title('Residuals')
        plt.show()
        #plt.pause(0.01)
        
    def report(self):
        print(lmfit.fit_report(self.params))

    def read(self):
        '''Read parameter values from the file'''
        params = lmfit.Parameters()
        with open(os.path.join(self.path, 'params.gss')) as fp:
            content = fp.readlines()
        for c in content[1:]:
            c = c.strip('\n').split()
            params.add(c[0], value=float(c[1]), min=float(c[2]), max=float(c[3]))
        for name in self.global_params:
            for i in range(1,self.N):
                params.add('{}_{}'.format(name, i+1), expr='{}_1'.format(name))
        return params
    
    def save(self):
        PARAMS = '{:<10}{:<16}{:<16}{:<16}{:<16}\n'.format('Name', 'Value', 'Min', 'Max', 'Stderr') \
              + '\n'.join(['{:<10}'.format(k)+'{:<16.8e}'.format(self.result.params[k].value)+'{:<16.8e}'.format(self.result.params[k].min)+'{:<16.8e}'.format(self.result.params[k].max)+'{:<16.8e}'.format(self.result.params[k].stderr)
                           for k in self.result.params]) \
            + '\n{:<10}{:<16.8e}'.format('Chi2', self.result.chisqr)
        fp = open(os.path.join(self.path, 'params.out'), 'w+')
        fp.write(PARAMS)
        fp.close()
            
        np.savetxt(os.path.join(self.path, 'simulation.csv'), np.hstack((self.data[:,0,None], self.y_sim)), fmt='%.8e', delimiter=',')
        np.savetxt(os.path.join(self.path, 'residual.csv'), np.hstack((self.data[:,0,None], self.data[:,1:] - self.y_sim)), fmt='%.8e', delimiter=',')
        print('Saved parameters into file: params.out')
        print('Saved simulation into file: simulation.csv')
        print('Saved residuals into file: residual.csv')
        self._menu()
        
    def help(self):
        return
                    

