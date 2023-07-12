import tensorflow.compat.v1 as tf
import numpy as np


class DATA_HANDLER(object):
  
  def __init__(self, unsupervised_training_data, 
               supervised_training_data, 
               supervised_training_labels, 
               test_data, 
               test_labels):

    self.unsupervised_training_data = unsupervised_training_data
    self.unsupervised_training_labels = []
    self.supervised_training_data = supervised_training_data
    self.supervised_training_labels = supervised_training_labels
    self.test_data = test_data
    self.test_labels = test_labels

    self.num_unsup_training_example = unsupervised_training_data.shape[0]
    self.num_sup_training_example = supervised_training_data.shape[0]
    self.num_test_example     = test_data.shape[0]

    self.whiten               = False
    self.unsupervised_training_index       = 0
    self.supervised_training_index       = 0
    self.test_index           = 0
    
  # needs work

  def do_whiten(self):
    self.whiten         = True
    data_to_be_whitened = np.copy(self.unsupervised_training_data)
    mean                = np.sum(data_to_be_whitened, axis = 0)/self.num_unsup_training_example
    mean                = np.tile(mean,self.num_unsup_training_example)
    mean                = np.reshape(mean,(self.num_unsup_training_example,784))
    centered_data       = data_to_be_whitened - mean                
    covariance          = np.dot(centered_data.T,centered_data)/self.num_unsup_training_example
    U,S,V               = np.linalg.svd(covariance)
    epsilon = 1e-5
    lambda_square       = np.diag(1./np.sqrt(S+epsilon))
    self.whitening_mat  = np.dot(np.dot(U, lambda_square), V)    
    self.whitened_unsupervised_training_data  = np.dot(centered_data,self.whitening_mat)
    
    data_to_be_whitened = np.copy(self.supervised_training_data)
    mean                = np.sum(data_to_be_whitened, axis = 0)/self.num_sup_training_example
    mean                = np.tile(mean,self.num_sup_training_example)
    mean                = np.reshape(mean,(self.num_sup_training_example,784))
    centered_data       = data_to_be_whitened - mean                
    covariance          = np.dot(centered_data.T,centered_data)/self.num_sup_training_example
    U,S,V               = np.linalg.svd(covariance)
    epsilon = 1e-5
    lambda_square       = np.diag(1./np.sqrt(S+epsilon))
    self.whitening_mat  = np.dot(np.dot(U, lambda_square), V)    
    self.whitened_supervised_training_data  = np.dot(centered_data,self.whitening_mat)

    data_to_be_whitened = np.copy(self.test_data)
    mean                = np.sum(data_to_be_whitened, axis = 0)/self.num_test_example
    mean                = np.tile(mean,self.num_test_example)
    mean                = np.reshape(mean,(self.num_test_example,784))
    centered_data       = data_to_be_whitened - mean  
    self.whitened_test_data  = np.dot(centered_data,self.whitening_mat)

    
  def next_batch(self, batch_size, type = 'train'):
    if type == 'train': # unsupervised training
      if self.whiten:
        operand = self.whitened_unsupervised_training_data
      else:
        operand = self.unsupervised_training_data
      operand_bis = self.unsupervised_training_labels
      self.unsupervised_training_index = (batch_size + self.unsupervised_training_index) % self.num_unsup_training_example
      index = self.unsupervised_training_index
      number = self.num_unsup_training_example
    elif type == 'softmax_train': # supervised training
      if self.whiten:
        operand = self.whitened_supervised_training_data
      else:
        operand = self.supervised_training_data
      operand_bis = self.supervised_training_labels
      self.supervised_training_index = (batch_size + self.supervised_training_index) % self.num_sup_training_example
      index = self.supervised_training_index
      number = self.num_sup_training_example
    elif type == 'test':
      if self.whiten:
        operand = self.whitened_test_data
      else:
        operand = self.test_data
      operand_bis = self.test_labels
      self.test_index = (batch_size + self.test_index) % self.num_test_example
      index = self.test_index
      number = self.num_test_example
    if index + batch_size > number:
      part1 = operand[index:,:]
      part2 = operand[:(index + batch_size)% number,:]
      result = np.concatenate([part1, part2])
      part1 = operand_bis[index:,:]
      part2 = operand_bis[:(index + batch_size)% number,:]
      result_bis = np.concatenate([part1, part2])
    else:
      result = operand[index:index + batch_size,:]
      if type == 'train':
        return result,0
      result_bis = operand_bis[index:index + batch_size,:]
    return result, result_bis
