import numpy as np
import torch
from collections import defaultdict


def log_loss(predicted_probs, true_labels, implementation='lib'):
    '''Log loss implementation in pytorch.'''
    if implementation == 'lib':
        loss_value = torch.nn.modules.BCEWithLogitsLoss()(predicted_probs, true_labels)
    elif implementation == 'custom':
        loss_value = (- true_labels.dot(torch.log(predicted_probs + 1e-20)).sum() -\
                     (1. - true_labels).dot(torch.log(1 - predicted_probs + 1e-20)).sum()) *\
                     (1. / true_labels.shape[0])
    return loss_value


def mae_loss(predicted_probs, true_labels):
    '''MAE loss implementation in pytorch.'''
    return (1 / true_labels.shape[0]) * torch.abs(predicted_probs - true_labels).sum()


def normalize_data(x):
    return x / x.mean(axis=0)


def plot_loss(loss_values, loss_name):
    plt.title(loss_name)
    plt.xlabel('iteration')
    plt.ylabel('loss value')
    plt.plot(loss_values)
    plt.show()


class LogisticRegressionTorch(torch.nn.Module):
    '''
    Logistic regression implementation in pytorch.
    '''
    def __init__(self, alpha=1e-3):
        '''
        alpha : float, gradient descent step.
        '''
        super().__init__()
        self.alpha = alpha
        self.losses = defaultdict(list)

    def _initialize_w_b(self, n_features, rnd_lower_bound=-1., rnd_upper_bound=1.):
        '''
        x: np.ndarray, shape: (n_samples, n_features), matrix object-features.
        y: np.ndarray, shape: (n_samples), true labels.
        rnd_lower_bound : float, lower bound for random sampling of weights.
        rnd_upper_bound : float, upper bound for random sampling of weights.
        '''
        self.w = torch.FloatTensor(n_features).uniform_(rnd_lower_bound, rnd_upper_bound).requires_grad_()
        self.b = torch.FloatTensor(1).uniform_(rnd_lower_bound, rnd_upper_bound).requires_grad_()

    def forward(self, x):
        scores = x.mv(self.w).add(self.b)
        predictions = torch.sigmoid(scores) # 1. / (1. + torch.exp(- self.scores))
        return predictions

    def fit(self, x, y, n_iterations=10, verbose=False):
        '''
        x: np.ndarray, shape: (n_samples, n_features), matrix object-features.
        y: np.ndarray, shape: (n_samples), true labels.
        returns fitted instance of LogisticRegressionTorch.
        '''
        self._initialize_w_b(n_features=x.shape[1])
        
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        
        for i_iteration in range(n_iterations):
            predictions = self.forward(x)
            
            loss = log_loss(predictions, y, implementation='custom')
            loss.backward()
            
            self.losses['log_loss'].append(loss)
            self.losses['mae_loss'].append(mae_loss(predictions, y))
            
            if verbose:
                print('predictions', predictions)
                print('loss', loss)
                print('self.w', self.w)
                print('self.b', self.b)
                print('self.w.grad', self.w.grad)
                print('self.b.grad', self.b.grad)
                print('\n')
            
            self.w.data.sub_(self.w.grad.mul(self.alpha))
            self.w.grad.zero_()
            self.b.data.sub_(self.b.grad.mul(self.alpha))
            self.b.grad.zero_()
        
        return self

    def predict_proba(self, x):
        '''
        x: np.ndarray of shape (n_samples, n_features), matrix object-features.
        returns probabilities for the 1 class, shape: (n_samples,).
        '''
        x = torch.FloatTensor(x)
        return self(x).data

def main():
    from sklearn.datasets import load_breast_cancer
    print('Start downloading breast_cancer dataset...')
    dataset = load_breast_cancer()
    print('Downloading finished.')

    x = normalize_data(np.array(dataset.data))
    y = np.array(dataset.target)

    clf = LogisticRegressionTorch(alpha=1e-1)
    print('Fit LogisticRegressionTorch with breast_cancer dataset...')
    clf.fit(x, y, n_iterations=5000, verbose=False)
    print('Fitting finished.')
    print('Start losses: logloss {:.3f}; mae {:.3f}.'.format(clf.losses['log_loss'][0], clf.losses['mae_loss'][0]))
    print('End   losses: logloss {:.3f}; mae {:.3f}.'.format(clf.losses['log_loss'][-1], clf.losses['mae_loss'][-1]))

    print('\nSample predictions: {}'.format(clf.predict_proba(x)[:10]))
    
if __name__ == '__main__':
    main()