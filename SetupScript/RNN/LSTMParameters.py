class LSTMParameters():
    def __init__(self, encode, meta, traininner, testinner, gtinner, testdev, ismeta, isimpression, isdrop, hiddendim, epochs, ncomponents, window, learnrate, iscuda, subname, numthread, batchsize):
        self.encode = encode
        self.meta = meta
        self.traininner = traininner
        self.testinner = testinner
        self.gtinner = gtinner
        self.testdev = testdev
        self.ismeta = ismeta
        self.isimpression = isimpression
        self.isdrop = isdrop
        self.hiddendim = hiddendim
        self.epochs = epochs
        self.ncomponents = ncomponents
        self.window = window
        self.learnrate = learnrate
        self.iscuda = iscuda
        self.subname = subname
        self.numthread = numthread
        self.batchsize = batchsize