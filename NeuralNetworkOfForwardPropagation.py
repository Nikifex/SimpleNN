import numpy as np

def act(x):  #binary step function
    if x<0.5:
        return 0
    else:
        return 1
def go(small_price, two_seats,auto):
    x=np.array([small_price, two_seats,auto])
    w11=[0.3,0.3,0]
    w12=[0.4,-0.5,1]
    weight1=np.array([w11,w12])
    weight2 = np.array([-1, 1])

    sum_hidden=np.dot(weight1,x)
    print("Values sum on neurons of hidden layer: "+str(sum_hidden))

    out_hidden=np.array([act(x)for x in sum_hidden])
    print("Values sum on out neurons of hidden layer: "+str(out_hidden))

    sum_end=np.dot(weight2,out_hidden)

    y = act(sum_end)
    print("Value NN:"+str(y))

    return y

small_price=1
two_seats=1
auto=0

res=go(small_price,two_seats,auto)
if res==1:
    print("Nice car!")
else:
    print("Сhosen something else...")
