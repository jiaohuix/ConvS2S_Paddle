from .paddleseq_dropout import PaddleseqDropout
from .grad_multiply import GradMultiply
from .learned_positional_embedding import LearnedPositionalEmbedding
#未完成
# from .quant_noise import quant_noise
from .adaptive_softmax import AdaptiveSoftmax
from .beamable_mm import BeamableMM # fold unfold
from .conv_tbc import ConvTBC # 以后可以添加c++算子
from .conv_nlc import ConvNLC # 卷积、注意力不需要频繁转置
from .linearized_convolution import LinearizedConvolution

'''
可改
softmax
quant noise
# c，有点难
beamable
learned pos √
linearized conv
'''