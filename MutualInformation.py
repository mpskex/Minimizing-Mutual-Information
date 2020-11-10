import torch.nn as nn
import torch
from torch.autograd import Function
"""
Mutual Information

Fangrui Liu @ University of British Columbia
fangrui.liu@ubc.ca

Copyright reserved 2020
"""


class approximate_joint_prob(Function):
    @staticmethod
    def forward(ctx, p_ij, bits):
        """
        :param ctx:
        :param bits:
        :return:
        """
        batch_size = bits.size(0)
        num_bits = bits.size(1)
        t_bits = torch.transpose(bits, 0, 1)
        for i in range(num_bits):
            for j in range(num_bits):
                for m in [0, 1]:
                    for n in [0, 1]:
                        p_ij[i, j, m, n] = torch.sum(
                            torch.logical_and(torch.eq(t_bits[i], m),
                                              torch.eq(t_bits[j], n)).float()) / batch_size
        ctx.save_for_backward(bits, p_ij)
        return p_ij

    @staticmethod
    def backward(ctx, grad_output):
        bits, p_ij = ctx.saved_tensors
        grad = torch.zeros(bits.size()).cuda(
        ) if torch.cuda.is_available() else torch.zeros(bits.size()).cpu()
        num_bits = bits.size(1)
        batch_size = bits.size(0)
        t_p_ij = torch.transpose(p_ij, 1, 2) / num_bits

        for j in range(num_bits):
            for n in [0, 1]:
                marginal = t_p_ij[j, n].sum()
                for m in [0, 1]:
                    for i in range(num_bits):
                        g_d = grad_output[i, j, m, n] * \
                            marginal * \
                            (m * 2 - 1)
                        grad[:, i] += g_d
        return None, grad


class MutualInformation(nn.Module):
    """
    Mutual Information
    """

    def __init__(self, entropy_sorting=False, triangle=True):
        super(MutualInformation, self).__init__()
        self.entropy_sorting = entropy_sorting
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.triangle = triangle

    def forward(self, bits):
        """
        :param p_i:     unconditional prob has shape [nbits, 2]
        :param p_ij:    joint prob has shape [nbits, nbits, 2, 2]
        :return:
        """
        bits = bits / 2 + 0.5
        num_bits = bits.size(1)
        mi = torch.zeros((1,)).to(device=self.device)
        p_ij = torch.zeros((num_bits, num_bits, 2, 2)
                           ).to(device=self.device)
        p_i_pos = bits.mean(0)

        if self.entropy_sorting:
            h = - p_i_pos * torch.log(p_i_pos)
            idx = torch.argsort(h, descending=False)
            bits = bits[idx]
            p_i_pos = p_i_pos[idx]

        p_ij = approximate_joint_prob.apply(p_ij, bits)
        p_i = torch.stack([1 - p_i_pos, p_i_pos],
                          dim=-1).to(device=self.device).detach()
        cnt = 0
        for i in range(num_bits):
            for j in range(num_bits):
                for m in [0, 1]:
                    for n in [0, 1]:
                        if self.triangle:
                            if i <= j:
                                continue
                        _p = p_ij[i, j, m, n]
                        if _p > 0 and p_i[i, m] > 0 and p_i[j, n] > 0:
                            mi += _p * \
                                torch.log(
                                    _p / (p_i[i, m]*p_i[j, n]))
                            cnt += 1
        return mi.squeeze() / cnt

if __name__ == '__main__':
    import time
    from torch.autograd import grad, gradcheck
    runs = 5
    nbits = 8
    batch_size = 50000
    ncorrs = [0, 4, 8, 12]
    time_used = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    MI = MutualInformation().to(device)

    bits = torch.randint(0, 2, size=(batch_size, nbits),
                         requires_grad=True, dtype=torch.float32).to(device)
    corr_bit = torch.randint(0, 2, size=(batch_size,),
                             requires_grad=True, dtype=torch.float32).to(device)
    mi_loss = MI(bits * 2 - 1).to(device)
    #test = gradcheck(MI, bits)

    # with torch.autograd.profiler.profile() as prof:
    if True:
        for ncorr in ncorrs:
            mi_average = 0
            grad_average = 0
            for n in range(runs):
                if ncorr != 0:
                    tbits = torch.cat([
                        torch.stack([corr_bit] * ncorr, dim=-1),
                        bits[:, :nbits-ncorr]
                    ], dim=-1).to(device)
                else:
                    tbits = bits
                tbits = tbits * 2 - 1
                start = time.time()
                mi_loss = MI(tbits)
                mi_average += mi_loss.detach().cpu()
                grad_average += grad(mi_loss, tbits)[0]
                time_used += time.time() - start
            print("Average Mutual Information is : {} Max Grad is : {}".format(mi_average/runs,
                                                                                   torch.max(grad_average)))
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    print("Time Elapsed: %.4fs" % (time_used / (runs * len(ncorrs))))
