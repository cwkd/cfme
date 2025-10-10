import torch.nn as nn
import torch_geometric.nn as geom_nn
import model.Twitter.BiGCN_Twitter as bigcn
import model.Twitter.EBGCN as ebgcn
import model.Twitter.BiGAT_Twitter as chgat
from lrp_pytorch.modules.linear import LRPLinear, Linear_Epsilon_Autograd_Fn, Linear_Zero_Autograd_Fn, \
    Guided_LRP_Autograd_Fn, CAM_Linear_Autograd_Fn, EB_Linear_Autograd_Fn, DeepLIFT_Linear_Autograd_Fn
from lrp_pytorch.modules.gcn_conv import LRPGCNConv, GCNConv_Autograd_Fn, CAM_GCNConv_Autograd_Fn, \
    EB_GCNConv_Autograd_Fn, DeepLIFT_GCNConv_Autograd_Fn
from lrp_pytorch.modules.gat2_conv import LRPGATv2Conv, GATv2Conv_Autograd_Fn, CAM_GATv2Conv_Autograd_Fn, \
    EB_GATv2Conv_Autograd_Fn
from lrp_pytorch.modules.bigcn import LRPBiGCNRumourGCN, BiGCNRumourGCN_Autograd_Fn, LRPBiGCN, LRP_Zero_BiGCN, \
    Guided_LRPBiGCN, CAM_BiGCN, EB_BiGCN, DeepLIFT_BiGCN
from lrp_pytorch.modules.ebgcn import LRPEBGCNRumourGCN, EBGCNRumourGCN_Autograd_Fn, LRPEBGCN, LRP_Zero_EBGCN, \
    Guided_LRPEBGCN, CAM_EBGCN, EB_EBGCN
from lrp_pytorch.modules.chgat import LRPCHGAT, LRP_Zero_CHGAT, Guided_LRPCHGAT, CAM_CHGAT, EB_CHGAT, \
    LRPPostLevelAttention, LRPPostLevelAttention2, \
    LRPPostLevelAttention_Autograd_Fn, CAM_PostLevelAttention_Autograd_Fn, EB_PostLevelAttention_Autograd_Fn
from copy import deepcopy


class Constants:

    key2class = {'nn.Linear': LRPLinear,
                 'nn.Linear(CAM)': LRPLinear,
                 'nn.Linear(EB)': LRPLinear,
                 'nn.Linear(DeepLIFT)': LRPLinear,
                 'geom_nn.GCNConv': LRPGCNConv,
                 'geom_nn.GCNConv(CAM)': LRPGCNConv,
                 'geom_nn.GCNConv(EB)': LRPGCNConv,
                 'geom_nn.GCNConv(DeepLIFT)': LRPGCNConv,
                 'bigcn.TDrumorGCN': LRPBiGCNRumourGCN,
                 'bigcn.BUrumorGCN': LRPBiGCNRumourGCN,
                 'ebgcn.TDrumorGCN': LRPEBGCNRumourGCN,
                 'ebgcn.BUrumorGCN': LRPEBGCNRumourGCN,
                 'bigcn.BiGCN': LRPBiGCN,
                 'bigcn.BiGCN(LRP_Zero)': LRP_Zero_BiGCN,
                 'bigcn.BiGCN(Guided_LRP)': Guided_LRPBiGCN,
                 'bigcn.BiGCN(CAM)': CAM_BiGCN,
                 'bigcn.BiGCN(EB)': EB_BiGCN,
                 'bigcn.BiGCN(DeepLIFT)': DeepLIFT_BiGCN,
                 'ebgcn.EBGCN': LRPEBGCN,
                 'ebgcn.EBGCN(LRP_Zero)': LRP_Zero_EBGCN,
                 'ebgcn.EBGCN(Guided_LRP)': Guided_LRPEBGCN,
                 'ebgcn.EBGCN(CAM)': CAM_EBGCN,
                 'ebgcn.EBGCN(EB)': EB_EBGCN,
                 'geom_nn.GATv2Conv': LRPGATv2Conv,
                 'geom_nn.GATv2Conv(CAM)': LRPGATv2Conv,
                 'geom_nn.GATv2Conv(EB)': LRPGATv2Conv,
                 'chgat.PostLevelAttention': LRPPostLevelAttention,
                 'chgat.PostLevelAttention2': LRPPostLevelAttention2,
                 'chgat.CHGAT': LRPCHGAT,
                 'chgat.CHGAT(LRP_Zero)': LRP_Zero_CHGAT,
                 'chgat.CHGAT(Guided_LRP)': Guided_LRPCHGAT,
                 'chgat.CHGAT(CAM)': CAM_CHGAT,
                 'chgat.CHGAT(EB)': EB_CHGAT,
                 }

    key2autograd_fn = {'nn.Linear': Linear_Epsilon_Autograd_Fn,
                       'nn.Linear(CAM)': CAM_Linear_Autograd_Fn,
                       'nn.Linear(EB)': EB_Linear_Autograd_Fn,
                       'nn.Linear(DeepLIFT)': DeepLIFT_Linear_Autograd_Fn,
                       'geom_nn.GCNConv': GCNConv_Autograd_Fn,
                       'geom_nn.GCNConv(CAM)': CAM_GCNConv_Autograd_Fn,
                       'geom_nn.GCNConv(EB)': EB_GCNConv_Autograd_Fn,
                       'geom_nn.GCNConv(DeepLIFT)': DeepLIFT_GCNConv_Autograd_Fn,
                       'bigcn.TDrumorGCN': BiGCNRumourGCN_Autograd_Fn,
                       'bigcn.BUrumorGCN': BiGCNRumourGCN_Autograd_Fn,
                       'ebgcn.TDrumorGCN': EBGCNRumourGCN_Autograd_Fn,
                       'ebgcn.BUrumorGCN': EBGCNRumourGCN_Autograd_Fn,
                       'chgat.PostLevelAttention': LRPPostLevelAttention_Autograd_Fn,
                       'chgat.PostLevelAttention(CAM)': CAM_PostLevelAttention_Autograd_Fn,
                       'chgat.PostLevelAttention(EB)': EB_PostLevelAttention_Autograd_Fn,
                       'chgat.PostLevelAttention2': LRPPostLevelAttention_Autograd_Fn,
                       'chgat.PostLevelAttention2(CAM)': CAM_PostLevelAttention_Autograd_Fn,
                       'chgat.PostLevelAttention2(EB)': EB_PostLevelAttention_Autograd_Fn,
                       'geom_nn.GATv2Conv': GATv2Conv_Autograd_Fn,
                       'geom_nn.GATv2Conv(CAM)': CAM_GATv2Conv_Autograd_Fn,
                       'geom_nn.GATv2Conv(EB)': EB_GATv2Conv_Autograd_Fn,
                       }


def get_lrpwrappermodule(module, lrp_params, is_contrastive=False):
    if lrp_params.get('mode') == 'lrp':
        if isinstance(module, nn.Linear) or isinstance(module, geom_nn.Linear):
        # if isinstance(module, nn.Linear):
            key = 'nn.Linear'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, geom_nn.GCNConv):
            key = 'geom_nn.GCNConv'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, geom_nn.GATv2Conv) or isinstance(module, geom_nn.conv.gat_conv.GATConv):
            key = 'geom_nn.GATv2Conv'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.TDrumorGCN):
            key = 'bigcn.TDrumorGCN'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.BUrumorGCN):
            key = 'bigcn.BUrumorGCN'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.BiGCN):
            key = 'bigcn.BiGCN'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.BiGCNv2):
            key = 'bigcn.BiGCN'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.TDrumorGCN):
            key = 'ebgcn.TDrumorGCN'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.BUrumorGCN):
            key = 'ebgcn.BUrumorGCN'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.EBGCN):
            key = 'ebgcn.EBGCN'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.EBGCNv2):
            key = 'ebgcn.EBGCN'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.PostLevelAttention):
            key = 'chgat.PostLevelAttention'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.PostLevelAttention2):
            key = 'chgat.PostLevelAttention2'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.CHGAT):
            key = 'chgat.CHGAT'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.CHGATv2):
            key = 'chgat.CHGAT'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        else:
            print('Unknown module', module)
            return None
    elif lrp_params.get('mode') == 'lrp_zero':
        if isinstance(module, nn.Linear) or isinstance(module, geom_nn.Linear):
        # if isinstance(module, nn.Linear):
            key = 'nn.Linear'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, geom_nn.GCNConv):
            key = 'geom_nn.GCNConv'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, geom_nn.GATv2Conv) or isinstance(module, geom_nn.conv.gat_conv.GATConv):
            key = 'geom_nn.GATv2Conv'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.TDrumorGCN):
            key = 'bigcn.TDrumorGCN'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.BUrumorGCN):
            key = 'bigcn.BUrumorGCN'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.BiGCN):
            key = 'bigcn.BiGCN(LRP_Zero)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.TDrumorGCN):
            key = 'ebgcn.TDrumorGCN'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.BUrumorGCN):
            key = 'ebgcn.BUrumorGCN'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.EBGCN):
            key = 'ebgcn.EBGCN(LRP_Zero)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.PostLevelAttention):
            key = 'chgat.PostLevelAttention'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.PostLevelAttention2):
            key = 'chgat.PostLevelAttention2'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.CHGAT):
            key = 'chgat.CHGAT(LRP_Zero)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        else:
            print('Unknown module', module)
            return None
    elif lrp_params.get('mode') == 'guided_lrp':
        if isinstance(module, nn.Linear) or isinstance(module, geom_nn.Linear):
        # if isinstance(module, nn.Linear):
            key = 'nn.Linear'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, geom_nn.GCNConv):
            key = 'geom_nn.GCNConv'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, geom_nn.GATv2Conv) or isinstance(module, geom_nn.conv.gat_conv.GATConv):
            key = 'geom_nn.GATv2Conv'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.TDrumorGCN):
            key = 'bigcn.TDrumorGCN'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.BUrumorGCN):
            key = 'bigcn.BUrumorGCN'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.BiGCN):
            key = 'bigcn.BiGCN(Guided_LRP)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.TDrumorGCN):
            key = 'ebgcn.TDrumorGCN'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.BUrumorGCN):
            key = 'ebgcn.BUrumorGCN'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.EBGCN):
            key = 'ebgcn.EBGCN(Guided_LRP)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.PostLevelAttention):
            key = 'chgat.PostLevelAttention'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.PostLevelAttention2):
            key = 'chgat.PostLevelAttention2'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.CHGAT):
            key = 'chgat.CHGAT(Guided_LRP)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        else:
            print('Unknown module', module)
            return None
    elif lrp_params.get('mode') == 'cam':
        if isinstance(module, nn.Linear) or isinstance(module, geom_nn.Linear):
            # if isinstance(module, nn.Linear):
            key = 'nn.Linear(CAM)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, geom_nn.GCNConv):
            key = 'geom_nn.GCNConv(CAM)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, geom_nn.GATv2Conv) or isinstance(module, geom_nn.conv.gat_conv.GATConv):
            key = 'geom_nn.GATv2Conv(CAM)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.TDrumorGCN):
            key = 'bigcn.TDrumorGCN(CAM)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.BUrumorGCN):
            key = 'bigcn.BUrumorGCN(CAM)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.BiGCN):
            key = 'bigcn.BiGCN(CAM)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.TDrumorGCN):
            key = 'ebgcn.TDrumorGCN(CAM)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.BUrumorGCN):
            key = 'ebgcn.BUrumorGCN(CAM)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.EBGCN):
            key = 'ebgcn.EBGCN(CAM)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.PostLevelAttention):
            key = 'chgat.PostLevelAttention(CAM)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.PostLevelAttention2):
            key = 'chgat.PostLevelAttention2(CAM)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.CHGAT):
            key = 'chgat.CHGAT(CAM)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        else:
            print('Unknown module', module)
            return None
    elif lrp_params.get('mode') == 'eb':
        if isinstance(module, nn.Linear) or isinstance(module, geom_nn.Linear):
            # if isinstance(module, nn.Linear):
            key = 'nn.Linear(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, geom_nn.GCNConv):
            key = 'geom_nn.GCNConv(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, geom_nn.GATv2Conv) or isinstance(module, geom_nn.conv.gat_conv.GATConv):
            key = 'geom_nn.GATv2Conv(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.TDrumorGCN):
            key = 'bigcn.TDrumorGCN(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.BUrumorGCN):
            key = 'bigcn.BUrumorGCN(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.TDrumorGCN):
            key = 'ebgcn.TDrumorGCN(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.BUrumorGCN):
            key = 'ebgcn.BUrumorGCN(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.PostLevelAttention):
            key = 'chgat.PostLevelAttention(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.PostLevelAttention2):
            key = 'chgat.PostLevelAttention2(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.CHGAT):
            key = 'chgat.CHGAT(EB)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params, is_contrastive=is_contrastive)

        if isinstance(module, bigcn.BiGCN):
            key = 'bigcn.BiGCN(EB)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params, is_contrastive=is_contrastive)

        elif isinstance(module, ebgcn.EBGCN):
            key = 'ebgcn.EBGCN(EB)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params, is_contrastive=is_contrastive)

        else:
            print('Unknown module', module)
            return None
    elif lrp_params.get('mode') == 'deeplift':
        if isinstance(module, nn.Linear) or isinstance(module, geom_nn.Linear):
            # if isinstance(module, nn.Linear):
            key = 'nn.Linear(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, geom_nn.GCNConv):
            key = 'geom_nn.GCNConv(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, geom_nn.GATv2Conv) or isinstance(module, geom_nn.conv.gat_conv.GATConv):
            key = 'geom_nn.GATv2Conv(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.TDrumorGCN):
            key = 'bigcn.TDrumorGCN(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, bigcn.BUrumorGCN):
            key = 'bigcn.BUrumorGCN(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.TDrumorGCN):
            key = 'ebgcn.TDrumorGCN(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, ebgcn.BUrumorGCN):
            key = 'ebgcn.BUrumorGCN(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.PostLevelAttention):
            key = 'chgat.PostLevelAttention(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.PostLevelAttention2):
            key = 'chgat.PostLevelAttention2(EB)'

            autograd_fn = Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params)

        elif isinstance(module, chgat.CHGAT):
            key = 'chgat.CHGAT(EB)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params, is_contrastive=is_contrastive)

        if isinstance(module, bigcn.BiGCN):
            key = 'bigcn.BiGCN(EB)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params, is_contrastive=is_contrastive)

        elif isinstance(module, ebgcn.EBGCN):
            key = 'ebgcn.EBGCN(EB)'

            autograd_fn = None  # Constants.key2autograd_fn[key]
            custom_class = Constants.key2class[key]
            return custom_class(module, autograd_fn, params=lrp_params, is_contrastive=is_contrastive)

        else:
            print('Unknown module', module)
            return None
    else:
        print('Explainability method not specified')
        raise Exception
