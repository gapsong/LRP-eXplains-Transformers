import torch
import torch.fx
from torch.autograd import Function
import torch.nn as nn
from lxt import functional as lfunctional


class WrapModule(nn.Module):
    def __init__(self, module):
        super(WrapModule, self).__init__()
        self.module = module


class IdentityRule(WrapModule):
    def forward(self, input):
        return identity_fn.apply(self.module, input)


def identity(fn, input):
    return identity_fn.apply(fn, input)


class identity_fn(Function):
    @staticmethod
    def forward(ctx, fn, input):
        output = fn(input)
        return output

    @staticmethod
    def backward(ctx, *out_relevance):
        return (None,) + out_relevance


class StopRelevanceRule(WrapModule):
    def forward(self, input):
        return stop_relevance_fn.apply(self.module, input)


class stop_relevance_fn(Function):
    @staticmethod
    def forward(ctx, fn, input):
        output = fn(input)
        return output

    @staticmethod
    def backward(ctx, *out_relevance):
        return (None, None)


class lepsilon_lrp_fn(Function):

    @staticmethod
    def forward(ctx, fn, epsilon, *inputs):
        # create boolean mask for inputs requiring gradients
        # TODO: use ctx.needs_input_grad instead of requires_grad
        requires_grads = [True if inp.requires_grad else False for inp in inputs]
        if sum(requires_grads) == 0:
            # no gradients to compute or gradient checkpointing is used
            return fn(*inputs)

        # detach inputs to avoid overwriting gradients if same input is used as multiple arguments (like in self-attention)
        inputs = tuple(
            inp.detach().requires_grad_() if inp.requires_grad else inp
            for inp in inputs
        )

        with torch.enable_grad():
            outputs = fn(*inputs)

        ctx.epsilon, ctx.requires_grads = epsilon, requires_grads
        # save only inputs requiring gradients
        inputs = tuple(inputs[i] for i in range(len(inputs)) if requires_grads[i])
        ctx.save_for_backward(*inputs, outputs)

        return outputs.detach()


class LEpsilonRule(WrapModule):

    def __init__(self, module, epsilon=1e-8):
        super(LEpsilonRule, self).__init__(module)
        self.epsilon = epsilon

    def forward(self, *inputs):
        return lepsilon_lrp_fn.apply(self.module, self.epsilon, *inputs)


class EpsilonRule(LEpsilonRule):
    def __init__(self, module, epsilon=1e-6, norm_backward=False, inplace=True):
        super(EpsilonRule, self).__init__(module, epsilon)
        self.epsilon = epsilon
        self.norm_backward = norm_backward
        self.inplace = inplace

    def forward(self, *inputs):
        return epsilon_lfp_fn.apply(
            self.module, self.epsilon, self.norm_backward, self.inplace, *inputs
        )


@torch.fx.wrap
def epsilon_lrp(fn, epsilon, *inputs):
    return epsilon_lfp_fn.apply(fn, epsilon, False, True, *inputs)


class epsilon_lfp_fn(lepsilon_lrp_fn):
    @staticmethod
    def forward(ctx, fn, epsilon, norm_backward, inplace, *inputs):
        requires_grads = [True if inp.requires_grad else False for inp in inputs]
        if sum(requires_grads) == 0:
            return fn(*inputs)

        inputs = tuple(
            inp.detach().requires_grad_() if inp.requires_grad else inp
            for inp in inputs
        )
        params = [param for param in fn.parameters(recurse=False)]
        for param in params:
            param.requires_grad_()

        with torch.enable_grad():
            outputs = fn(*inputs)

        ctx.epsilon, ctx.norm_backward, ctx.requires_grads, ctx.inplace = (
            epsilon,
            norm_backward,
            requires_grads,
            inplace,
        )
        inputs = tuple(inputs[i] for i in range(len(inputs)) if requires_grads[i])
        ctx.save_for_backward(*inputs, *params, outputs)
        ctx.n_inputs, ctx.n_params = len(inputs), len(params)
        ctx.fn = fn

        return outputs.detach()

    @staticmethod
    def backward(ctx, *incoming_reward):
        if ctx.norm_backward:
            if isinstance(incoming_reward, tuple):
                incoming_reward_new = []
                for g in incoming_reward:
                    if g is not None:
                        incoming_reward_new.append(
                            g
                            / torch.where(
                                g.abs().max() > 0,
                                g.abs().max(),
                                torch.ones_like(g.abs().max()),
                            )
                        )
                    else:
                        incoming_reward_new.append(None)
                incoming_reward = tuple(incoming_reward_new)
            else:
                if incoming_reward is not None:
                    incoming_reward = incoming_reward / torch.where(
                        incoming_reward.abs().max() > 0,
                        incoming_reward.abs().max(),
                        torch.ones_like(incoming_reward.abs().max()),
                    )
                else:
                    incoming_reward = None

        outputs = ctx.saved_tensors[-1]
        inputs = ctx.saved_tensors[: ctx.n_inputs]
        params = ctx.saved_tensors[ctx.n_inputs: ctx.n_inputs + ctx.n_params]

        normed_reward = incoming_reward[0] / lfunctional._stabilize(
            outputs, ctx.epsilon, inplace=False
        )

        for param in params:
            if not isinstance(param, tuple):
                param = (param,)
            param_grads = torch.autograd.grad(
                outputs, param, normed_reward, retain_graph=True
            )
            if ctx.inplace:
                param_reward = tuple(
                    param_grads[i].mul_(param[i].abs()) for i in range(len(param))
                )
            else:
                param_reward = tuple(
                    param_grads[i] * param[i].abs() for i in range(len(param))
                )
            for i in range(len(param)):
                param[i].feedback = param_reward[i]

        input_grads = torch.autograd.grad(
            outputs, inputs, normed_reward, retain_graph=False
        )
        if ctx.inplace:
            outgoing_reward = tuple(
                input_grads[i].mul_(inputs[i]) if ctx.requires_grads[i] else None
                for i in range(len(ctx.requires_grads))
            )
        else:
            outgoing_reward = tuple(
                input_grads[i] * inputs[i] if ctx.requires_grads[i] else None
                for i in range(len(ctx.requires_grads))
            )

        return (None, None, None, None) + outgoing_reward


class UniformRule(WrapModule):
    def forward(self, *inputs):
        return uniform_rule_fn.apply(self.module, *inputs)


class uniform_rule_fn(Function):
    @staticmethod
    def forward(ctx, fn, *inputs):
        requires_grads = [True if inp.requires_grad else False for inp in inputs]
        if sum(requires_grads) == 0:
            return fn(*inputs)

        inputs = tuple(
            inp.detach().requires_grad_() if inp.requires_grad else inp
            for inp in inputs
        )
        with torch.enable_grad():
            outputs = fn(*inputs)

        ctx.requires_grads = requires_grads
        inputs = tuple(inputs[i] for i in range(len(inputs)) if requires_grads[i])
        ctx.save_for_backward(*inputs, outputs)
        return outputs.detach()

    @staticmethod
    def backward(ctx, *out_relevance):
        inputs, _ = ctx.saved_tensors[:-1], ctx.saved_tensors[-1]
        num_inputs = len(inputs)
        uniform_relevance = out_relevance[0] / num_inputs
        relevance = iter([uniform_relevance.expand_as(inp) for inp in inputs])
        return (None,) + tuple(
            next(relevance) if req_grad else None for req_grad in ctx.requires_grads
        )


class UniformEpsilonRule(EpsilonRule):
    def __init__(self, module, epsilon=1e-6, norm_backward=False, inplace=True):
        super(UniformEpsilonRule, self).__init__(
            module, epsilon, norm_backward, inplace
        )


class RuleGenerator:
    def __init__(self, rule, **kwargs):
        self.rule = rule
        self.rule_kwargs = kwargs

    def __call__(self, module):
        return self.rule(module, **self.rule_kwargs)
