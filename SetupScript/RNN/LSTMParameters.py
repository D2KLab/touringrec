class LSTMParameters():
    def __init__(self, train, test, gt, epochs, ncomponents, window, learnrate, iscuda, subname, numthread, batchsize):
        self.train = train
        self.test = test
        self.gt = gt
        self.epochs = epochs
        self.ncomponents = ncomponents
        self.window = window
        self.learnrate = learnrate
        self.iscuda = iscuda
        self.subname = subname
        self.numthread = numthread
        self.batchsize = batchsize