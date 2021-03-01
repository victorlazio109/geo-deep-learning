import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def soft_skeletonize(x):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    for i in range(10):
        min_pool_x = torch.nn.functional.max_pool2d(x * -1, (3, 3), 1, 1) * -1
        contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x


# def norm_intersection(center_line, vessel):
#     '''
#     inputs shape  (batch, channel, height, width)
#     intersection formalized by first ares
#     x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
#     '''
#     smooth = 1.
#     clf = center_line.view(*center_line.shape[:2], -1)
#     vf = vessel.view(*vessel.shape[:2], -1)
#     intersection = (clf * vf).sum(-1)
#
#     return (intersection + smooth) / (clf.sum(-1) + smooth)


def norm_intersection(
        centerline: torch.Tensor,
        label: torch.Tensor,
        smooth: float = 0.0,
        eps: float = 1e-7,
        dims=None) -> torch.Tensor:
    assert centerline.size() == label.size()
    if dims is not None:
        intersection = torch.sum(centerline * label, dim=dims)
        cardinality = torch.sum(label, dim=dims)
        # cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(centerline * label)
        cardinality = torch.sum(label)
        # cardinality = torch.sum(output + target)
    norm = (intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return norm


class clDiceLoss(nn.Module):

    def __init__(self, smooth=1.0, eps=1e-7, ignore_index=None, weight=None):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=self.weight)

    def forward(self, output, target, target_skeleton):
        # print('cl_ouput_shape:', cl_output.shape, 'target_skeleton:', target_skeleton.shape)
        # print('output_shape:', output.shape, 'target_shape:', target.shape)

        CE = self.ce(output, target)
        bs = target.size(0)
        num_classes = output.size(1)
        dims = (0, 2)

        cl_output = soft_skeletonize(output)
        output = output.log_softmax(dim=1).exp()
        cl_output = cl_output.log_softmax(dim=1).exp()

        output = output.view(bs, num_classes, -1)
        cl_output = cl_output.view(bs, num_classes, -1)

        target = target.view(bs, -1)
        target_skeleton = target_skeleton.view(bs, -1)

        target = F.one_hot(target, num_classes)  # N,H*W -> N,H*W, C
        target = target.permute(0, 2, 1)  # H, C, H*W
        target_skeleton = F.one_hot(target_skeleton, num_classes)  # N,H*W -> N,H*W, C
        target_skeleton = target_skeleton.permute(0, 2, 1)  # H, C, H*W

        # print('cl_ouput_shape:', cl_output.shape, 'target_skeleton:', target_skeleton.shape)
        # print('output_shape:', output.shape, 'target_shape:', target.shape)

        iflat = norm_intersection(cl_output, target.type_as(cl_output),
                                  smooth=self.smooth, eps=self.eps, dims=dims)
        tflat = norm_intersection(target_skeleton, output.type_as(target_skeleton),
                                  smooth=self.smooth, eps=self.eps, dims=dims)

        intersection = iflat * tflat

        # print('iflat', iflat, 'tflat', tflat, 'intersection', intersection)

        scores = (2.0 * intersection) / (iflat + tflat)
        # print('scores', scores)
        loss = 1.0 - scores
        mask = target.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        # scores = soft_dice_score(output, target.type_as(output), smooth=self.smooth, eps=self.eps, dims=dims)
        # loss = 1.0 - scores
        #
        # mask = target.sum(dims) > 0
        # loss *= mask.to(loss.dtype)
        CL = loss.mean()

        to_loss = 0.5 * (CE + CL)

        return to_loss
