'''
在反向传播时对梯度缩放
'''
from paddle.autograd import PyLayer

class GradMultiply(PyLayer):
    @staticmethod #必须静态方法，（不初始化就可调用）
    def forward(ctx, x,scale):
        # ctx is a context object that store some objects for backward.
        ctx.scale = scale  # 注册变量，back中可以用
        return x

    @staticmethod
    # forward has only one output, so there is only one gradient in the input of backward.
    def backward(ctx, grad):
        # forward has only one input, so only one gradient tensor is returned.
        return grad * ctx.scale
