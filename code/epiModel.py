# close area epi-models

class EpiModel:
    def __init__(self):
        self.history = []
        self.timestamp = 0

    def setInitial(self):
        self.timestamp = 0
        self.history = []
    
    def nextState(self):
        pass

    def makeHistory(self):
        pass

class SI(EpiModel):
    def __init__(self, beta, c):
        super(SI, self).__init__()
        self.beta = beta    # probability of transmission per person 
        self.c = c          # rate of contact per person per timestamp
        
        self.N = 0          # total population
        self.S = 0 
        self.I = 0

    def setInitial(self, S, I, timestamp):
        self.S = S
        self.I = I
        self.N = S + I
        self.timestamp = timestamp
        self.history.append(self.makeHistory(0, 0))
    
    def nextState(self):
        lamb = self.beta * self.c * I / self.N
        dS = -lamb * S
        dI = lamb * S
        self.S += dS
        self.I += dI
        self.timestamp += 1
        self.history.append(self.makeHistory(dS, dI))

    def makeHistory(self, dS, dI):
        return {
            "timestamp": self.timestamp,
            "S": self.S,
            "I": self.I,
            "dS": dS,
            "dI": dI
        }

class SIR(EpiModel):
    def __init__(self, beta, c, mu_s, mu_i, mu_r, b, v):
        super(SIR, self).__init__()
        self.beta = beta
        self.c = c
        self.mu_s = mu_s
        self.mu_i = mu_i
        self.mu_r = mu_r
        self.b
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
        self.history.append(self.makeHistory(0, 0, 0))

    def nextState(self):
        lamb = self.beta * self.c * self.I / self.N
        dS = - lamb * self.S + self.b * self.N - self.mu_s * S
        dI = lamb * self.S - self.v * self.I - self.mu_i * S
        dR = self.v * I - self.mu_r * self.R
        self.S += dS
        self.I += dI
        self.R += dR
        self.timestamp += 1
        self.history.append(self.makeHistory(dS, dI, dR))
        
    def makeHistory(self, dS, dI, dR):
        return {
            "timestamp": self.timestamp,
            "S": self.S,
            "I": self.I,
            "R": self.R,
            "dS": dS,
            "dI": dI,
            "dR": dR
        }
        