"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
from layers.att_layers import GraphAttentionLayer
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act
from layers.hyp_layers import HypLinear,HypAct


from geoopt_plusplus.modules.linear import PoincareLinear
from HyperbolicNN.modules import Gyrobias, HyperbolicLinear
from lib.lorentz.layers.LFC import LorentzFullyConnected
from lib.lorentz.manifold import CustomLorentz
from NestedHyperbolicNN.modules import HyperbolicFeatureTransformation

class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class MLP(Encoder):
    """
    Multi-layer perceptron.
    """

    def __init__(self, c, args):
        super(MLP, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False


class HNN(Encoder):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, c, args):
        super(HNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        hnn_layers = []
        use_bias = bool(args.bias)
        if args.manifold=='Hyperboloid':
            dims[0]=dims[0]-1
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            if args.transform_mode in ["Mobius", "Einstein", "LorentzTan"]:
                hnn_layers.append(HypLinear(args.manifold, in_dim, out_dim, self.c, args.dropout, use_bias))
            elif args.transform_mode == 'PoincareFC':
                hnn_layers.append(
                    PoincareLinear(in_dim, out_dim, ball=self.manifold, bias=use_bias, c=self.c, dropout=args.dropout))
                if args.gyrobias:
                    hnn_layers.append(Gyrobias(out_dim, metric='poincare', K=-self.c))
            elif args.transform_mode == 'HFC':
                # The main HNN code uses c = |K| > 0, while HyperbolicLinear uses K = -c < 0.
                hnn_layers.append(
                    HyperbolicLinear(in_dim, out_dim, metric=args.manifold, gyro_bias=use_bias, K=-self.c,
                                     dropout=args.dropout, drop_mode=args.drop_mode, normalize_v=getattr(args, 'normalize_v', False)))
            elif args.transform_mode == 'LFC':
                # The official LorentzFullyConnected code requires the ambient dimesnions
                hnn_layers.append(LorentzFullyConnected(
                    manifold=CustomLorentz(k=self.c),
                    in_features=in_dim+1,
                    out_features=out_dim+1,
                    dropout=args.dropout,
                    bias=use_bias))
                if args.gyrobias:
                    hnn_layers.append(Gyrobias(out_dim, metric='lorentz', K=-self.c))
            elif args.transform_mode == 'NestFC':
                if args.manifold != 'Hyperboloid':
                    raise ValueError(
                        "NestFC requires manifold=Hyperboloid."
                    )
                hnn_layers.append(HyperbolicFeatureTransformation(
                    manifold=self.manifold,
                    in_features=in_dim+1,
                    out_features=out_dim+1,
                    c=self.c,
                    use_bias=use_bias,
                    dropout=args.dropout))
            else:
                raise ValueError(
                    f"Unsupported HNN transform_mode: {args.transform_mode!r}."
                )

            if args.act:
                hnn_layers.append(HypAct(self.manifold, self.c, self.c, act))

        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return super(HNN, self).encode(x_hyp, adj)

class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, c, args):
        super(GCN, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True


class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, args):
        super(HGCN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGCN, self).encode(x_hyp, adj)


class GAT(Encoder):
    """
    Graph Attention Networks.
    """

    def __init__(self, c, args):
        super(GAT, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(
                    GraphAttentionLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat))
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True


class Shallow(Encoder):
    """
    Shallow Embedding method.
    Learns embeddings or loads pretrained embeddings and uses an MLP for classification.
    """

    def __init__(self, c, args):
        super(Shallow, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.use_feats = args.use_feats
        weights = torch.Tensor(args.n_nodes, args.dim)
        if not args.pretrained_embeddings:
            weights = self.manifold.init_weights(weights, self.c)
            trainable = True
        else:
            weights = torch.Tensor(np.load(args.pretrained_embeddings))
            assert weights.shape[0] == args.n_nodes, "The embeddings you passed seem to be for another dataset."
            trainable = False
        self.lt = manifolds.ManifoldParameter(weights, trainable, self.manifold, self.c)
        self.all_nodes = torch.LongTensor(list(range(args.n_nodes)))
        layers = []
        if args.pretrained_embeddings is not None and args.num_layers > 0:
            # MLP layers after pre-trained embeddings
            dims, acts = get_dim_act(args)
            if self.use_feats:
                dims[0] = args.feat_dim + weights.shape[1]
            else:
                dims[0] = weights.shape[1]
            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i + 1]
                act = acts[i]
                layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False

    def encode(self, x, adj):
        h = self.lt[self.all_nodes, :]
        if self.use_feats:
            h = torch.cat((h, x), 1)
        return super(Shallow, self).encode(h, adj)
