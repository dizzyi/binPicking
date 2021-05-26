import torch

from modules import AttentiveGraphtoGraph

def main():
    ATGG = AttentiveGraphtoGraph()

    x = torch.ones(3,64)

    edge = torch.tensor([
        [0,1,1,2],
        [1,0,2,1]
    ],dtype=torch.long)

    print('ready')
    ATGG(x,edge)
    print('here')
    print("finish")

if __name__=="__main__":
    main()