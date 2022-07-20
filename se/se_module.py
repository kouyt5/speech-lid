import sys, os
sys.path.append(".")
sys.path.append("..")

from ccml.ccml_module import CCMLModule
from se.models.FaSNet import FaSNet_origin


class SEModule(CCMLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = FaSNet_origin(enc_dim=args.enc_dim, feature_dim=args.feature_dim,
                              hidden_dim=args.hidden_dim, layer=args.layer,
                              segment_size=args.segment_size, nspk=args.nspk,
                              win_len=args.win_len, context_len=args.context_len,
                              sr=args.sr)