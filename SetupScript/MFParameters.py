import json
class MF_Parameters:
    def __init__(self, epochs, ncomponents, lossfunction, mfk, learningrate, learningschedule, useralpha, itemalpha, rho, epsilon, maxsampled, actionsweights, listactions):
        self.epochs = epochs
        self.ncomponents = ncomponents
        self.lossfunction = lossfunction
        self.mfk = mfk
        self.actionsweights = actionsweights
        self.listactions = listactions
        self.learningrate = learningrate
        self.learningschedule = learningschedule
        self.useralpha = useralpha
        self.itemalpha = itemalpha
        self.rho = rho
        self.epsilon = epsilon
        self.maxsampled = maxsampled
    
    def print_params(self):
        print ('#'*20)
        print('     PARAMETERS     ')
        print ('#'*20)
        print('# Epochs: ' + str(self.epochs))
        print('# Components ' + str(self.ncomponents))
        print('Loss function ' + str(self.lossfunction))
        print('K: ' + str(self.mfk))
        print('Weights: ' + json.dumps(self.actionsweights))
        print('Learning rate: ' + str(self.learningrate))
        print('Learning schedule: ' + str(self.learningschedule))
        print('User alpha: ' + str(self.useralpha))
        print('Item alpha: ' + str(self.itemalpha))
        print('Rho: ' + str(self.rho))
        print('Epsilon: ' + str(self.epsilon))
        print('Max sampled: ' + str(self.maxsampled))

