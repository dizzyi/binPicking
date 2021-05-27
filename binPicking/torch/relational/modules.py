## third-party
import torch
from   torch.nn import Linear, LayerNorm, LeakyReLU
from torch.nn.modules.linear import Linear

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

## modules import
from preprocess import fetch_preprocessing


class FetchInputPreprocessing(torch.nn.Module):
    def __init__(self,
                 object_dim,
                 embedding_dim = 64):
        
        #self.save_init_params(locals())  ## from the rlkit-relational

        super(FetchInputPreprocessing, self).__init__()

        self.FC64 = Linear(object_dim,embedding_dim) #Fully connected 64 Layer
        self.Norm = LayerNorm(embedding_dim)


    def forward(self, obs):
        vertices, edge_index = fetch_preprocessing(obs) 

        return LayerNorm( self.FC64( vertices ) ), edge_index



class AttentiveGraphtoGraph(MessagePassing):
    def __init__(self, 
                 features = 64, 
                 leaky_slope = 0.2):

        super(AttentiveGraphtoGraph, self).__init__(aggr='add')

        self.features = features
        self.FC_qcm   = Linear(features, features * 3)
        self.LeReLU   = LeakyReLU(leaky_slope)
        self.FC_logit = Linear(features, 1)
        self.Norm     = LayerNorm(features)

    # propagate -> LeakyReLU -> normalize
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        #x = self.propagate(edge_index, size= (x.size(0),), x=x, norm = 1)
        #x = self.LeReLU(x)
        #x = LayerNorm(x)
        #return x
        return self.Norm(
            self.LeReLU(
                self.propagate(edge_index,x=x)
            )
        )

    def message(self, x_j):
        x_j = self.FC_qcm(x_j)

        query, key, message = torch.chunk(x_j, 3, dim=1)

        attention = query + key                                         # sum                  

        attention = torch.tanh(query)                                   # tanh layer

        attention = self.FC_logit(query)                                # Fully conncected layer to Logit

        attention = torch.tanh(attention)                               # squish the attention to -1 and 1

        return attention * message 

