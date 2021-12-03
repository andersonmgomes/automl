#https://towardsdatascience.com/how-to-use-the-multiprocessing-package-in-python3-a1c808415ec2
from time import sleep
from multiprocessing import Process
from multiprocessing import Pool

class ParalellTest:
    def __init__(self):
        print('init')
        #self.p_feateng = Pool()
        self.p_feateng = Pool(processes=2)
        self.result = self.p_feateng.starmap_async(self.features_eng, [(19, 19), (13, 19), (1, 5), (3, 4), (19, 19), (13, 19), (1, 5), (3, 4)], callback=self.process_result)
        self.ret = None
        print('init end...')
    
    def process_result(self, return_val):
        self.ret = return_val
        #print(return_val)
    
    def limpeza_dados(self):
        for i in range(10):
            print('limpeza_dados()', i)
            sleep(2)

    @staticmethod
    def features_eng(x,y):
        print('features_eng()')
        sleep(10)
        return x+y
            
    def train_models(self):
        for i in range(3):
            print('train_models()', i)
            sleep(2)
    def predict(self):
        print('predict', self.result.get())

if __name__ == "__main__":
    obj = ParalellTest()
    #sleep(5)
    obj.predict()
    obj.train_models()
    obj.predict()
    obj.train_models()
    obj.predict()
    #p_feateng.close()
    #p_feateng.join()
    
    
'''
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
'''