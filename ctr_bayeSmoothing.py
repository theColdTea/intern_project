import numpy
import random
import scipy.special as special
from matplotlib import pyplot as plt

class bayeSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        # sample中是点击率ctr
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for clk_rt in sample:
            # I display的次数
            imp = random.random() * imp_upperbound
            # imp = imp_upperbound
            # C click次数
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):
        # epsilon 终止条件
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_alpha-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def predict(self, C, I):
        return (C+self.alpha)*1.0/(I+self.alpha+self.beta)

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i] + alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i] - clks[i] + beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i] + alpha + beta) - special.digamma(alpha + beta))

        return alpha * (numerator_alpha / denominator), beta * (numerator_beta / denominator)


if __name__ == '__main__':
    bs = bayeSmoothing(1,1)
    I, C = bs.sample(100,50,1000,10000)
    # print(I, C)
    bs.update(I,C,1000,0.0000001)
    print(bs.alpha, bs.beta)
    print(bs.predict(10,8))
    print(numpy.random.beta(bs.alpha,bs.beta))
