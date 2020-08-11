# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 17:00:23 2018

@author: lj
"""

import numpy as np
from sklearn import svm
from sklearn import cross_validation
import random
import matplotlib.pyplot as plt

## 1.loading data
def load_data(data_file):
    '''
    # import training data
    input:  data_file(string):file
    output: data(mat):sample feature
            label(mat):sample label
    '''
    data = []
    label = []
    f = open(data_file)
    for line in f.readlines():
        lines = line.strip().split(' ') # array
        # get label (the first item in each line)
        label.append(float(lines[0]))      
        index = 0 
        tmp = []
        for i in range(1, len(lines)): # the remaining parts of array
            li = lines[i].strip().split(":") 
            if int(li[0]) - 1 == index:
                tmp.append(float(li[1]))
            else:
                while(int(li[0]) - 1 > index):
                    tmp.append(0)
                    index += 1
                tmp.append(float(li[1]))
            index += 1
        while len(tmp) < 13:
            tmp.append(0)
        data.append(tmp)
    f.close()
    return np.array(data), np.array(label).T



## 2. PSO优化算法
class PSO(object):
    def __init__(self,particle_num,particle_dim,iter_num,c1,c2,w,max_value,min_value):
        '''
        # initialize prams
        particle_num(int):number of particles
        particle_dim(int):dimension of particles
        iter_num(int):max iteration
        c1(float):local learning factor, weight of history best (pbest)
        c2(float):global learning factor, weight of global best (gbest)
        w(float):inertia factor, inertia of particles' past direction on present direction
        max_value(float):max pram value
        min_value(float):min pram value
        '''
        self.particle_num = particle_num
        self.particle_dim = particle_dim
        self.iter_num = iter_num
        self.c1 = c1  ## usually 2.0 pbest
        self.c2 = c2  ## usually 2.0 gbest
        self.w = w    
        self.max_value = max_value
        self.min_value = min_value
        
        
### 2.1 particle swarm initialization
    def swarm_origin(self):
        '''
        # initialize particle swarm
        input:self(object):PSO type
        output:particle_loc(list):particle swarm location list
               particle_dir(list):particle swarm direction list
        '''
        particle_loc = []
        particle_dir = []
        for i in range(self.particle_num): # each particle
            tmp1 = []
            tmp2 = []
            for j in range(self.particle_dim): # each dimension
                a = random.random()
                b = random.random()
                tmp1.append(a * (self.max_value - self.min_value) + self.min_value)
                tmp2.append(b)
            particle_loc.append(tmp1) # randnum within min-max range
            particle_dir.append(tmp2) # randnum 0-1
        
        return particle_loc,particle_dir

## 2.2 calculate fitness function list; initialize pbest_parameters & gbest_parameter   
    def fitness(self,particle_loc):
        '''
        # calculate fitness function
        input:self(object):PSO type
              particle_loc(list):particle swarm location list
        output:fitness_value(list):fitness function list
        '''
        fitness_value = []
        ### 1. generate fitness fuction RBF_SVM's 3_fold cross check mean
        for i in range(self.particle_num): # each particle
            rbf_svm = svm.SVC(kernel = 'rbf', C = particle_loc[i][0], gamma = particle_loc[i][1])
            cv_scores = cross_validation.cross_val_score(rbf_svm,trainX,trainY,cv =3,scoring = 'accuracy')
            fitness_value.append(cv_scores.mean())
            # get the list of present particle swarm's fitness values
        ### 2. find present particle swarm's best fitness value and its loc prams
        current_fitness = 0.0
        current_parameter = []
        for i in range(self.particle_num): # each particle
            if current_fitness < fitness_value[i]:
                current_fitness = fitness_value[i]
                current_parameter = particle_loc[i] # find the max fitness value and record the particle's loc

        return fitness_value,current_fitness,current_parameter 
    
    # myself propose a new fitness function, it supposes to calculate energies but I'll just pretend the locs can represent energies
    def cal_fitness(self,particle_loc):
        '''
        # new way of defining a standard fitness value
        input:self(object):PSO type
              particle_loc(list):particle swarm location list
        output: fitness_value(list): a list of fitness values
        '''
        max_energy = 0.0
        min_energy = 1000.0
        fitness_value = []
        energy_list = []
        for list1 in range(self.particle_loc):
            for item in list1:
                sum_ += item*item
            energy = sum_**0.5
            energy_list.append(energy)
            if max_energy < energy:
                max_energy = energy
            if min_energy > energy:
                min_energy = energy
        for ene in energy_list:
            value = (ene-min_energy)/(max_energy-min_energy)
            fitness_value.append(value)
            
        return fitness_value
                
        
        

## 2.3  update particle location 
    def updata(self,particle_loc,particle_dir,gbest_parameter,pbest_parameters):
        '''
        # particle swarm location update
        input:self(object):PSO type
              particle_loc(list):particle swarm location list
              particle_dir(list):particle swarm direction list
              gbest_parameter(list):gbest
              pbest_parameters(list):history best for each particle
        output:particle_loc(list):new particle swarm location list
               particle_dir(list):new particle swarm direction list
        '''
        ## 1.calculate new quantum swarm direction and particle swarm location
        for i in range(self.particle_num): 
            a1 = [x * self.w for x in particle_dir[i]]
            a2 = [y * self.c1 * random.random() for y in list(np.array(pbest_parameters[i]) - np.array(particle_loc[i]))]
            a3 = [z * self.c2 * random.random() for z in list(np.array(gbest_parameter) - np.array(particle_dir[i]))]
            particle_dir[i] = list(np.array(a1) + np.array(a2) + np.array(a3))
#            particle_dir[i] = self.w * particle_dir[i] + self.c1 * random.random() * (pbest_parameters[i] - particle_loc[i]) + self.c2 * random.random() * (gbest_parameter - particle_dir[i])
            particle_loc[i] = list(np.array(particle_loc[i]) + np.array(particle_dir[i]))
            
        ## 2.fixiate new quantum location within [min_value,max_value]
        ### 2.1 each pram's value list
        parameter_list = []
        for i in range(self.particle_dim):
            tmp1 = []
            for j in range(self.particle_num):
                tmp1.append(particle_loc[j][i])
            parameter_list.append(tmp1)
        ### 2.2 each pram's max min and mean value  
        value = []
        for i in range(self.particle_dim):
            tmp2 = []
            tmp2.append(max(parameter_list[i]))
            tmp2.append(min(parameter_list[i]))
            value.append(tmp2)
        
        for i in range(self.particle_num):
            for j in range(self.particle_dim):
                particle_loc[i][j] = (particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (self.max_value - self.min_value) + self.min_value
                
        return particle_loc,particle_dir

## 2.4 draw fitness function moving plot
    def plot(self,results):
        '''
        draw plot
        '''
        X = []
        Y = []
        for i in range(self.iter_num):
            X.append(i + 1)
            Y.append(results[i])
        plt.plot(X,Y)
        plt.xlabel('Number of iteration',size = 15)
        plt.ylabel('Value of CV',size = 15)
        plt.title('PSO_RBF_SVM parameter optimization')
        plt.show() 
        
## 2.5 main function       
    def main(self):
        '''
        main function
        '''
        results = []
        best_fitness = 0.0 
        ## 1、particle swarm initialization
        particle_loc,particle_dir = self.swarm_origin()
        ## 2、initialize gbest_parameter、pbest_parameters、fitness_value list
        ### 2.1 gbest_parameter
        gbest_parameter = []
        for i in range(self.particle_dim):
            gbest_parameter.append(0.0)
        ### 2.2 pbest_parameters
        pbest_parameters = []
        for i in range(self.particle_num):
            tmp1 = []
            for j in range(self.particle_dim):
                tmp1.append(0.0)
            pbest_parameters.append(tmp1)
        ### 2.3 fitness_value
        fitness_value = []
        for i in range(self.particle_num):
            fitness_value.append(0.0)
    
        ## 3.iterations
        for i in range(self.iter_num):
            ### 3.1 calculate current fitness function list
            current_fitness_value,current_best_fitness,current_best_parameter = self.fitness(particle_loc)
            ### 3.2 calculate current gbest_parameter、pbest_parameters & best_fitness
            for j in range(self.particle_num):
                if current_fitness_value[j] > fitness_value[j]:
                    pbest_parameters[j] = particle_loc[j]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                gbest_parameter = current_best_parameter
            
            print('iteration is :',i+1,';Best parameters:',gbest_parameter,';Best fitness',best_fitness)
            results.append(best_fitness)
            ### 3.3 update fitness_value
            fitness_value = current_fitness_value
            ### 3.4 update particle swarm
            particle_loc,particle_dir = self.updata(particle_loc,particle_dir,gbest_parameter,pbest_parameters)
        ## 4.show result
        results.sort()
        self.plot(results)
        print('Final parameters are :',gbest_parameter)
            

if __name__ == '__main__':
    print('----------------1.Load Data-------------------')
    trainX,trainY = load_data('rbf_data')
    print('----------------2.Parameter Seting------------')
    particle_num = 100
    particle_dim = 2
    iter_num = 50
    c1 = 2
    c2 = 2
    w = 0.8
    max_value = 15
    min_value = 0.001
    print('----------------3.PSO_RBF_SVM-----------------')
    pso = PSO(particle_num,particle_dim,iter_num,c1,c2,w,max_value,min_value)
    pso.main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
