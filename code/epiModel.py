# close area epi-models

class EpiModel:
    def __init__(self):
        self.history = []
        self.timestamp = 0

    def setInitial(self):
        self.timestamp = 0
        self.history = []
    
    def nextState(self, sIn, sOut, iIn, iOut):
        pass

    def makeHistory(self):
        pass

class SI(EpiModel):
    def __init__(self, name, beta, c):
        super(SI, self).__init__()
        self.name = name
        self.beta = beta    # probability of transmission per person 
        self.c = c          # rate of contact per person per timestamp
        
        self.N = 0          # total population
        self.S = 0 
        self.I = 0

    def set(self, N, S, I, timestamp, ratio, history):
        self.N = N
        self.S = S
        self.I = I
        self.timestamp = timestamp
        self.ratio = ratio
        self.history = history

    def setInitial(self, S, I, timestamp):
        self.dS = 0
        self.dI = 0
        self.S = S
        self.I = I
        self.N = S + I
        self.timestamp = timestamp
        self.history.append(self.makeHistory())
    
    def nextState(self, sIn, sOut, iIn, iOut, beta_c=None):
        if beta_c:
            self.ratio = beta_c * self.I / self.N
        else:
            self.ratio = self.beta * self.c * self.I / self.N
        self.dS = -self.ratio * self.S + sIn - sOut
        self.dI = self.ratio * self.S + iIn - iOut
        self.S += self.dS
        self.I += self.dI
        self.N += sIn - sOut + iIn - iOut
        self.timestamp += 1
        self.history.append(self.makeHistory())

    def makeHistory(self):
        historyOb = {
            "name": self.name,
            "timestamp": self.timestamp,
            "S": self.S,
            "I": self.I,
            "dS": self.dS,
            "dI": self.dI
        }
        return historyOb

    def copy(self):
        copy = SI(self.beta, self.c)
        copy.set(self.N, self.S, self.I, self.timestamp, self.ratio, list(self.history))
        return copy

class SIR(EpiModel):
    def __init__(self, name, beta, c, mu_s, mu_i, mu_r, b, v):
        super(SIR, self).__init__()
        self.name = name
        self.beta = beta
        self.c = c
        self.mu_s = mu_s
        self.mu_i = mu_i
        self.mu_r = mu_r
        self.b = b
        self.v = v

        self.N = 0
        self.S = 0
        self.I = 0
        self.R = 0

    def setInitial(self, S, I, R, timestamp):
        self.S = S
        self.I = I
        self.R = R
        self.N = S + I + R
        self.timestamp = timestamp
        self.ratio = self.beta * self.c * self.I / self.N
        self.history.append(self.makeHistory(0, 0, 0))

    def nextState(self, sIn, sOut, iIn, iOut):
        self.dS = - self.ratio * self.S + self.b * self.N - self.mu_s * self.S + sIn - sOut
        self.dI = self.ratio * self.S - self.v * self.I - self.mu_i * self.S + iIn - iOut
        self.dR = self.v * self.I - self.mu_r * self.R
        self.S += dS
        self.I += dI
        self.R += dR
        self.N += sIn - sOut + iIn - iOut
        self.timestamp += 1
        self.history.append(self.makeHistory(dS, dI, dR))
        self.ratio = self.beta * self.c * self.I / self.N
        
    def makeHistory(self, dS, dI, dR):
        historyOb = {
            "name": self.name,
            "timestamp": self.timestamp,
            "S": self.S,
            "I": self.I,
            "R": self.R,
            "dS": dS,
            "dI": dI,
            "dR": dR
        }
        return historyOb
        