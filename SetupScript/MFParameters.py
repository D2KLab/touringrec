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